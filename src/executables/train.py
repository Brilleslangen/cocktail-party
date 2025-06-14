import os
import time
import platform

import torch
import hydra
import wandb
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch import nn
from torch.utils.data import DataLoader

from src.data.collate import setup_train_dataloaders
from src.evaluate import Loss, compute_mask, batch_si_snr, batch_snr
from src.helpers import (
    select_device,
    count_parameters,
    count_macs,
    prettify_macs,
    prettify_param_count,
    format_time,
)
from src.data.streaming import Streamer
from src.evaluate.loss import MaskedMSELoss


def setup_device_optimizations():
    """Configure device-specific optimizations."""
    device = select_device()
    device_type = device.type

    # Device-specific settings
    if device_type == "cuda":
        # Enable TF32 on Ampere GPUs for better performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable cudnn benchmarking for consistent input sizes
        torch.backends.cudnn.benchmark = True

        # Get GPU info for logging
        gpu_name = torch.cuda.get_device_name()
        gpu_capability = torch.cuda.get_device_capability()

        print(f"🚀 CUDA device: {gpu_name}")
        print(f"   Compute capability: {gpu_capability[0]}.{gpu_capability[1]}")

        # Check FlashAttention availability
        if gpu_capability[0] >= 8:
            print("   ✓ FlashAttention-2 compatible GPU detected")

        # Mixed precision settings
        use_amp = True
        amp_dtype = torch.float16  # Use bfloat16 if available on Ampere+
        if gpu_capability[0] >= 8:
            amp_dtype = torch.bfloat16
            print("   ✓ Using bfloat16 for mixed precision")
        else:
            print("   ✓ Using float16 for mixed precision")

    elif device_type == "mps":
        # MPS (Apple Silicon) settings
        print(f"🍎 MPS device: {platform.processor()}. Mixed precision disabled.")
        use_amp = False
        amp_dtype = torch.float32

    else:
        # CPU fallback
        print("💻 CPU. Mixed precision disabled")
        use_amp = False
        amp_dtype = torch.float32

    return device, use_amp, amp_dtype


def train_epoch(model: nn.Module, loader: DataLoader, loss_fn: Loss,
                optimizer: torch.optim.Optimizer, device: torch.device,
                use_amp: bool, amp_dtype: torch.dtype):
    """
    Runs one full training epoch over `loader`.
    Returns the average loss.
    """
    model.train()
    total_loss = 0.0
    total_mse_loss = 0.0
    streaming_mode = getattr(model, "streaming_mode", False)
    streamer = Streamer(model) if streaming_mode else None
    mse_loss_fn = MaskedMSELoss()

    # Setup mixed precision
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    pbar = tqdm(loader, total=len(loader), desc="Train", leave=False)

    for i, (mix, refs, lengths) in enumerate(pbar):
        mix, refs, lengths = mix.to(device), refs.to(device), lengths.to(device)

        # Forward pass with mixed precision
        if use_amp and device.type == "cuda":
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                if streaming_mode:
                    ests, refs, lengths = streamer.stream_batch(mix, refs, lengths, trim_warmup=True)
                else:
                    ests = model(mix)
                loss = loss_fn(ests, refs, lengths)
                mse_loss = mse_loss_fn()

            # Backward with gradient scaling
            scaler.scale(loss).backward()

            # Unscale before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # ConvTasNet style gradient clipping

            # Step and update
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            # Standard training (CPU/MPS)
            if streaming_mode:
                ests, refs, lengths = streamer.stream_batch(mix, refs, lengths, trim_warmup=True)
            else:
                ests = model(mix)

            loss = loss_fn(ests, refs, lengths)
            mse_loss = mse_loss_fn(ests, refs, lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad()

        loss_val = loss.item()
        mse_loss_val = mse_loss.item()
        total_loss += loss_val
        total_mse_loss += mse_loss_val

        pbar.set_postfix(avg_loss=f"{total_loss / (i + 1):.4f}", curr_loss=f"{loss_val:.4f}",
                         avg_mse=f"{total_mse_loss / (i + 1):.4f}", curr_mse=f"{mse_loss_val:.4f}")

        if model.separator.stateful:
            model.reset_state()

    return total_loss / len(loader)


def validate_epoch(model: torch.nn.Module, loader: DataLoader, criterion: Loss,
                   device: torch.device, use_amp: bool, amp_dtype: torch.dtype):
    """Validation with mixed precision support."""
    model.eval()
    streaming_mode = getattr(model, "streaming_mode", False)
    streamer = Streamer(model) if streaming_mode else None
    totals = {k: 0.0 for k in ["loss", "snr", "snr_i", "si_snr", "si_snr_i"]}
    total_examples = 0

    with torch.no_grad():
        for mix, refs, lengths in tqdm(loader, desc="Validate", leave=False):
            mix, refs, lengths = mix.to(device), refs.to(device), lengths.to(device)

            if hasattr(model, "reset_state"):
                model.reset_state()
            elif hasattr(model, "separator") and hasattr(model.separator, "reset_state"):
                model.separator.reset_state()

            # Forward pass with mixed precision
            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    if streaming_mode:
                        ests, refs, lengths = streamer.stream_batch(mix, refs, lengths, trim_warmup=True)
                        mix = mix[..., streamer.pad_warmup:]
                    else:
                        ests = model(mix)
                    loss = criterion(ests, refs, lengths)
            else:
                if streaming_mode:
                    ests, refs, lengths = streamer.stream_batch(mix, refs, lengths, trim_warmup=True)
                    mix = mix[..., streamer.pad_warmup:]
                else:
                    ests = model(mix)
                loss = criterion(ests, refs, lengths)

            B = ests.size(0)

            # Compute mask for valid frames
            mask = compute_mask(lengths, ests.size(-1), ests.device)

            # For each channel
            snr_out_L = batch_snr(ests[:, 0, :], refs[:, 0, :], mask)
            snr_out_R = batch_snr(ests[:, 1, :], refs[:, 1, :], mask)
            snr_mix_L = batch_snr(mix[:, 0, :], refs[:, 0, :], mask)
            snr_mix_R = batch_snr(mix[:, 1, :], refs[:, 1, :], mask)

            snr_est = (snr_out_L + snr_out_R) / 2
            snr_mix = (snr_mix_L + snr_mix_R) / 2
            snr_i = snr_est - snr_mix

            si_snr_out_L = batch_si_snr(ests[:, 0, :], refs[:, 0, :], mask)
            si_snr_out_R = batch_si_snr(ests[:, 1, :], refs[:, 1, :], mask)
            si_snr_mix_L = batch_si_snr(mix[:, 0, :], refs[:, 0, :], mask)
            si_snr_mix_R = batch_si_snr(mix[:, 1, :], refs[:, 1, :], mask)

            si_snr_est = (si_snr_out_L + si_snr_out_R) / 2
            si_snr_mix = (si_snr_mix_L + si_snr_mix_R) / 2
            si_snr_i = si_snr_est - si_snr_mix

            # Sum (not mean) to enable averaging at end
            totals["loss"] += loss.item() * B
            totals["snr"] += snr_est.sum().item()
            totals["snr_i"] += snr_i.sum().item()
            totals["si_snr"] += si_snr_est.sum().item()
            totals["si_snr_i"] += si_snr_i.sum().item()
            total_examples += B

    for k in totals:
        totals[k] /= total_examples

    return totals


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    # Setup device and optimizations
    device, use_amp, amp_dtype = setup_device_optimizations()
    torch.manual_seed(cfg.training.params.seed)
    OmegaConf.register_new_resolver("mul", lambda x, y: int(x * y))

    print(f"👨🏻‍🏫 Workers: {os.cpu_count()}")

    # Build model
    model = instantiate(cfg.model_arch).to(device)

    # Standard optimizer creation (keeping it simple)
    optimizer = instantiate(cfg.training.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.training.scheduler, optimizer=optimizer, _recursive_=False)
    loss = instantiate(cfg.training.loss).to(device)

    # Calculate composite figures and init wandb
    param_count = count_parameters(model)
    macs = count_macs(model)
    pretty_macs = prettify_macs(macs)
    pretty_param_count = prettify_param_count(param_count)
    run_name = f"{cfg.name}_{pretty_param_count}"

    print(f"Pretty parameters: {pretty_param_count}")
    print('Pretty MACs:', pretty_macs)

    # Optional: Compile model for additional speedup (PyTorch 2.0+)
    if cfg.training.params.get("compile_model", False) and hasattr(torch, "compile"):
        print("🔥 Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        _ = model(torch.randn(1, 2, cfg.dataset.sample_rate, device=device))  # Warmup compile

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["model_arch"]["param_count"] = param_count
    cfg_dict["model_arch"]["macs"] = macs
    cfg_dict["model_arch"]["pretty_macs"] = pretty_macs
    cfg_dict["device"] = str(device)
    cfg_dict["mixed_precision"] = use_amp
    cfg_dict["amp_dtype"] = str(amp_dtype) if use_amp else None

    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            tags=cfg.wandb.tags,
            job_type='train',
            config=cfg_dict,
            name=run_name,
            reinit='finish_previous'
        )
        wandb.run.summary["model/param_count"] = param_count
        wandb.run.summary["model/macs_per_second"] = macs
        wandb.run.summary["model/macs_pretty"] = pretty_macs

    if cfg.training.print_config:
        print(OmegaConf.to_yaml(cfg))

    # dataloaders
    train_loader, val_loader = setup_train_dataloaders(cfg)

    best_metric_name = "si_snr_i"
    best_metric_value = -float("inf")
    best_ckpt_path = f"{cfg.training.model_save_dir}/{run_name}.pt"

    # Early stopping setup
    patience = cfg.training.early_stopping.patience
    min_delta = cfg.training.early_stopping.min_delta
    best_val_loss = float("inf")
    epochs_no_improve = 0
    epochs_trained = 0

    for epoch in range(1, cfg.training.params.max_epochs + 1):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, loss, optimizer, device, use_amp, amp_dtype)
        val_stats = validate_epoch(model, val_loader, loss, device, use_amp, amp_dtype)
        time_elapsed = format_time(time.time() - start_time)

        scheduler.step()
        epochs_trained = epoch
        print(
            f"\rEpoch {epoch:2d} time={time_elapsed} train_loss={train_loss:.4f} val_loss={val_stats['loss']:.4f} " +
            f"SI-SNR={val_stats['si_snr']:.2f} SI-SNRi={val_stats['si_snr_i']:.2f} SNR={val_stats['snr']:.2f} ")

        # Early stopping check
        if val_stats["loss"] < best_val_loss - min_delta:
            best_val_loss = val_stats["loss"]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

        if cfg.wandb.enabled:
            wandb.log({
                "train/loss": train_loss,
                "val/loss": val_stats["loss"],
                "val/SNR": val_stats["snr"],
                "val/SNRi": val_stats["snr_i"],
                "val/SI-SNR": val_stats["si_snr"],
                "val/SI-SNRi": val_stats["si_snr_i"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }, step=epoch)

        # checkpoint best
        current = val_stats[best_metric_name]
        if current > best_metric_value:
            best_metric_value = current
            os.makedirs(os.path.dirname(best_ckpt_path), exist_ok=True)
            torch.save({
                "epoch": epoch,
                "cfg": cfg_dict,
                "model_state": model.state_dict(),
                "opt_state": optimizer.state_dict(),
                "sched_state": scheduler.state_dict(),
                "val_stats": val_stats,
            }, best_ckpt_path)

    # upload the best‐metric model to W&B
    if cfg.wandb.enabled and best_ckpt_path is not None:
        wandb.run.summary[f"best/{best_metric_name}"] = best_metric_value
        wandb.run.summary["epochs_trained"] = epochs_trained
        art = wandb.Artifact(run_name, type="model")
        art.add_file(best_ckpt_path)
        wandb.log_artifact(art)

    print(f"Training complete. Best {best_metric_name}: {best_metric_value:.4f}")


if __name__ == "__main__":
    main()
