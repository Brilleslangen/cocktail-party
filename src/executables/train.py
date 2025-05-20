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
from src.evaluate import Loss, compute_mask, batch_si_snr, batch_snr
from src.helpers import select_device, count_parameters, prettify_param_count, format_time
from src.data.streaming import Streamer


# TODO:
#  1. Check what loss should be used. Currently using MSELoss.
#  5. Need 40000, 10000, and 6000 audio samples for training, validation, and test sets respectively, to replicate
#  main relevant work (Han et. al 2021: research/papers/system architecture/binaural speech separation).
#  6. Establish what metrics we need, see if we can use the ones from the original paper. Also I think we should
#  develop a timer_metric that averages in-to-out time during validation. Less than 10<ms is the goal.
#  7. Make dataset use the huggingface datasets library. This will allow us to upload the dataset and enable
#  streaming mode.

def train_epoch(model: nn.Module, loader: DataLoader, loss_fn: Loss, optimizer: torch.optim.Optimizer,
                device: torch.device):
    """
    Runs one full training epoch over `loader`.
    Returns the average loss.
    """
    model.train()
    total_loss = 0.0
    streaming_mode = getattr(model, "streaming_mode", False)
    streamer = Streamer(model) if streaming_mode else None

    for mix, refs, lengths in tqdm(loader, desc="Train", leave=False):
        mix, refs, lengths = mix.to(device), refs.to(device), lengths.to(device)
        B, C, _ = mix.shape

        # reset any stateful separator once per sequence
        model.reset_state()

        # forward
        if streaming_mode:
            ests, refs, lengths = streamer.stream_batch(mix, refs, lengths, trim_warmup=True)
        else:
            ests = model(mix)

        loss = loss_fn(ests, refs, lengths)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate_epoch(
    model: torch.nn.Module,
    loader,
    criterion,
    device: torch.device,
):
    model.eval()
    totals = {k: 0.0 for k in ["loss", "snr", "snr_i", "si_snr", "si_snr_i"]}
    total_examples = 0

    with torch.no_grad():
        for mix, refs, lengths in tqdm(loader, desc="Validate", leave=False):
            mix, refs, lengths = mix.to(device), refs.to(device), lengths.to(device)

            if hasattr(model, "reset_state"):
                model.reset_state()
            elif hasattr(model, "separator") and hasattr(model.separator, "reset_state"):
                model.separator.reset_state(batch_size=mix.size(0))

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
            snr_i = snr_est - snr_mix  # improvement

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
    device = torch.device(select_device())
    torch.manual_seed(cfg.training.params.seed)

    print("Using device:", device)
    print("workers:", os.cpu_count())

    # build model, optimizer, scheduler, loss
    model = instantiate(cfg.model_arch).to(device)
    optimizer = instantiate(cfg.training.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.training.scheduler, optimizer=optimizer, _recursive_=False)
    loss = instantiate(cfg.training.loss).to(device)

    # Calculate composite figures and init wandb
    param_count = count_parameters(model)
    pretty_param_count = prettify_param_count(param_count)
    run_name = f"{cfg.name}_{pretty_param_count}"

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["model_arch"]["param_count"] = param_count

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

    print(OmegaConf.to_yaml(cfg))

    # dataloaders
    train_loader, val_loader = setup_train_dataloaders(cfg)

    best_metric_name = "si_snr_i"
    best_metric_value = -float("inf")
    best_ckpt_path = f"{cfg.training.model_save_dir}/{run_name}.pt"

    for epoch in range(1, cfg.training.params.max_epochs + 1):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, loss, optimizer, device)
        val_stats = validate_epoch(model, val_loader, loss, device)
        time_elapsed = format_time(time.time() - start_time)

        scheduler.step(val_stats["loss"])
        print(
            f"\rEpoch {epoch:2d} time={time_elapsed} train_loss={train_loss:.4f} val_loss={val_stats['loss']:.4f} " +
            f"SI-SNR={val_stats['si_snr']:.2f} SI-SNRi={val_stats['si_snr_i']:.2f} SNR={val_stats['snr']:.2f} ")

        if cfg.wandb.enabled:
            wandb.log({
                "train/loss": train_loss,
                "val/loss": val_stats["loss"],
                "val/SNR": val_stats["snr"],
                "val/SNRi": val_stats["snr_i"],
                "val/SI-SNR": val_stats["si_snr"],
                "val/SI-SNRi": val_stats["si_snr_i"],
                "val/SI-SDR": val_stats["si_sdr"],
                "val/SI-SDRi": val_stats["si_sdr_i"],
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
        art = wandb.Artifact(run_name, type="model")
        art.add_file(best_ckpt_path)
        wandb.log_artifact(art)

    print(f"Training complete. Best {best_metric_name}: {best_metric_value:.4f}")


if __name__ == "__main__":
    main()
