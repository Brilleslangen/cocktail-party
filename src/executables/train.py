import os
import time

import torch
import hydra
import wandb
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch import nn
from torch.utils.data import DataLoader

from src.data.collate import setup_train_dataloaders
from src.evaluate import Loss, compute_validation_metrics, count_parameters, count_macs
from src.evaluate.loss import MaskedMSELoss
from src.helpers import (
    prettify_macs,
    prettify_param_count,
    format_time, setup_device_optimizations,
)
from src.data.streaming import Streamer


def train_epoch(model: nn.Module, loader: DataLoader, loss_fn: Loss,
                optimizer: torch.optim.Optimizer, device: torch.device,
                use_amp: bool, amp_dtype: torch.dtype):
    """
    Runs one full training epoch over `loader`.
    Returns the average loss and MSE loss.
    """
    model.train()
    total_loss = 0.0
    total_mse_loss = 0.0
    streaming_mode = getattr(model, "streaming_mode", False)
    streamer = Streamer(model) if streaming_mode else None
    mse_loss_fn = MaskedMSELoss()

    # Setup mixed precision
    scaler = torch.amp.GradScaler('cuda') if (use_amp and device.type == "cuda") else None
    use_targets_as_input = getattr(model, "use_targets_as_input", False)

    pbar = tqdm(loader, total=len(loader), desc="Train", leave=False)

    for i, (mix, refs, lengths) in enumerate(pbar):
        mix, refs, lengths = mix.to(device), refs.to(device), lengths.to(device)
        model_input = refs if use_targets_as_input else mix
        B, C, T = mix.shape

        if use_targets_as_input:
            # Assert that targets and references are the same
            assert torch.allclose(model_input, refs, atol=1e-6), ("Mix and references must be the same when using "
                                                                  " targets as input.")

        if hasattr(model, "reset_state"):
            model.reset_state(batch_size=B, chunk_len=T)

        # Forward pass with mixed precision

        if use_amp and device.type == "cuda":
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                if streaming_mode:
                    ests, refs, lengths = streamer.stream_batch(model_input, refs, lengths, trim_warmup=True)
                else:
                    ests = model(model_input)

                loss = loss_fn(ests, refs, lengths)
                mse_loss = mse_loss_fn(ests, refs, lengths)

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
                ests, refs, lengths = streamer.stream_batch(model_input, refs, lengths, trim_warmup=True)
            else:
                ests = model(model_input)

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

        pbar.set_postfix(avg_loss=f"{(total_loss / (i + 1)):.4f}", avg_mse=f"{(total_mse_loss / (i + 1)):.4f}")

    return total_loss / len(loader), total_mse_loss / len(loader)


def validate_epoch(model: torch.nn.Module, loader: DataLoader, criterion: Loss,
                   device: torch.device, use_amp: bool, amp_dtype: torch.dtype):
    """Validation with energy-weighted metrics."""
    model.eval()
    streaming_mode = getattr(model, "streaming_mode", False)
    streamer = Streamer(model) if streaming_mode else None

    # Initialize metric accumulators
    totals = {
        "loss": 0.0,
        "mc_sdr": 0.0,
        "mc_si_sdr": 0.0,
        "mc_si_sdr_i": 0.0,
        "ew_mse": 0.0,
        "ew_sdr": 0.0,
        "ew_si_sdr": 0.0,
        "ew_si_sdr_i": 0.0,
    }
    total_examples = 0

    with torch.no_grad():
        for mix, refs, lengths in tqdm(loader, desc="Validate", leave=False):
            mix, refs, lengths = mix.to(device), refs.to(device), lengths.to(device)
            model_input = refs if model.use_targets_as_input else mix
            B, C, T = mix.shape

            if model.use_targets_as_input:
                # Assert that targets and references are the same
                assert torch.allclose(model_input, refs,
                                      atol=1e-6), "Mix and references must be the same when using targets as input."

            # Reset state for stateful models
            if hasattr(model, "reset_state"):
                model.reset_state(batch_size=B, chunk_len=T)

            # Forward pass with mixed precision
            if use_amp and device.type == "cuda":
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    if streaming_mode:
                        ests, refs, lengths = streamer.stream_batch(model_input, refs, lengths, trim_warmup=True)
                        mix = mix[..., streamer.pad_warmup:]
                    else:
                        ests = model(model_input)
                    loss = criterion(ests, refs, lengths)
            else:
                if streaming_mode:
                    ests, refs, lengths = streamer.stream_batch(model_input, refs, lengths, trim_warmup=True)
                    mix = mix[..., streamer.pad_warmup:]
                else:
                    ests = model(model_input)
                loss = criterion(ests, refs, lengths)

            B = ests.size(0)

            # Compute energy-weighted metrics
            metrics = compute_validation_metrics(ests, mix, refs, lengths)

            # Accumulate metrics
            totals["loss"] += loss.item() * B
            for metric_name, metric_values in metrics.items():
                totals[metric_name] += metric_values.sum().item()

            total_examples += B

    # Average over all examples
    for k in totals:
        totals[k] /= total_examples

    return totals


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    # Setup device and optimizations
    device, use_amp, amp_dtype = setup_device_optimizations()
    torch.manual_seed(cfg.training.params.seed)

    print(f"👨🏻‍🏫 Workers: {os.cpu_count()}")

    # Build model
    model = instantiate(cfg.model_arch, device=device).to(device)

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

    print(f"📊 Model Statistics:")
    print(f"   Parameters: {pretty_param_count}")
    print(f"   MACs/s: {pretty_macs}")

    # Optional: Compile model for additional speedup (PyTorch 2.0+)
    if cfg.training.params.get("compile_model", False) and hasattr(torch, "compile"):
        print("🔥 Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        _ = model(torch.randn(1, 2, cfg.dataset.sample_rate, device=device))  # Warmup compile

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["model_arch"]["param_count"] = pretty_param_count
    cfg_dict["model_arch"]["pretty_macs"] = pretty_macs
    cfg_dict["device"] = str(device.type)
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
        wandb.run.summary["model/param_count"] = pretty_param_count
        wandb.run.summary["model/macs_per_second"] = pretty_macs

    if cfg.training.print_config:
        print(OmegaConf.to_yaml(cfg))

    # dataloaders
    train_loader, val_loader = setup_train_dataloaders(cfg)

    # Training configuration
    best_metric_name = "ew_si_sdr_i"
    best_metric_value = -float("inf")
    best_ckpt_path = f"{cfg.training.model_save_dir}/{run_name}.pt"

    # Early stopping setup
    patience = cfg.training.early_stopping.patience
    min_delta = cfg.training.early_stopping.min_delta
    best_val_loss = float("inf")
    epochs_no_improve = 0
    epochs_trained = 0

    # ---- PRE-TRAIN init ----

    if cfg.wandb.enabled:
        wandb.log({
            "val/mc_si_sdr_i": 0,
            "val/ew_si_sdr_i": 0,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch": 0
        })
    # ---- END PRE-TRAIN VALIDATION ----

    print(f"\n🎯 Training for {cfg.training.params.max_epochs} epochs, optimizing {best_metric_name}")
    print(f"   Early stopping: patience={patience}, min_delta={min_delta}")

    for epoch in range(1, cfg.training.params.max_epochs + 1):
        start_time = time.time()
        train_loss, train_mse = train_epoch(model, train_loader, loss, optimizer, device, use_amp, amp_dtype)
        val_stats = validate_epoch(model, val_loader, loss, device, use_amp, amp_dtype)
        time_elapsed = format_time(time.time() - start_time)

        scheduler.step()
        epochs_trained = epoch

        # Print epoch summary
        print(f"\rEpoch {epoch:3d}/{cfg.training.params.max_epochs} | "
              f"Time: {time_elapsed} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train MSE: {train_mse:.4f} | "
              f"Val Loss: {val_stats['loss']:.4f} | "
              f"MC-SI-SDRi: {val_stats['mc_si_sdr_i']:.2f} dB | "
              f"MC-SI-SDR: {val_stats['mc_si_sdr']:.2f} dB | "
              f"MC-SDR: {val_stats['mc_sdr']:.2f} dB")

        # Early stopping check
        if val_stats["loss"] < best_val_loss - min_delta:
            best_val_loss = val_stats["loss"]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\n⏹️  Early stopping triggered at epoch {epoch}.")
            break

        # Log to wandb
        if cfg.wandb.enabled:
            wandb.log({
                "train/loss": train_loss,
                "train/mse": train_mse,
                "val/loss": val_stats["loss"],
                "val/mc_sdr": val_stats["mc_sdr"],
                "val/mc_si_sdr": val_stats["mc_si_sdr"],
                "val/mc_si_sdr_i": val_stats["mc_si_sdr_i"],
                "val/ew_mse": val_stats["ew_mse"],
                "val/ew_sdr": val_stats["ew_sdr"],
                "val/ew_si_sdr": val_stats["ew_si_sdr"],
                "val/ew_si_sdr_i": val_stats["ew_si_sdr_i"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch": epoch
            })

        # Save checkpoint if best
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
            print(f"   💾 Saved new best model (EW-SI-SDRi: {current:.2f} dB)")

    # Upload the best model to W&B
    if cfg.wandb.enabled and best_ckpt_path is not None:
        wandb.run.summary[f"best/{best_metric_name}"] = best_metric_value
        wandb.run.summary["epochs_trained"] = epochs_trained
        art = wandb.Artifact(run_name, type="model")
        art.add_file(best_ckpt_path)
        wandb.log_artifact(art)

    print(f"\n✅ Training complete!")
    print(f"   Best {best_metric_name}: {best_metric_value:.4f} dB")
    print(f"   Total epochs: {epochs_trained}")


if __name__ == "__main__":
    main()
