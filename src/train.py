import os

import torch
import torchaudio
import hydra
import wandb
from tqdm import trange, tqdm
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch import nn
from torch.utils.data import DataLoader
from src.data.dataset import AudioDataset
from src.helpers import select_device, count_parameters, prettify_param_count
from src.data.collate import pad_collate
from src.data.bucket_sampler import BucketBatchSampler
from torchmetrics.functional.audio import (
    signal_noise_ratio,
    scale_invariant_signal_distortion_ratio,
    signal_distortion_ratio
)


# TODO:
#  1. Check what loss should be used. Currently using MSELoss.
#  5. Need 40000, 10000, and 6000 audio samples for training, validation, and test sets respectively, to replicate
#  main relevant work (Han et. al 2021: research/papers/system architecture/binaural speech separation).
#  6. Establish what metrics we need, see if we can use the ones from the original paper. Also I think we should
#  develop a timer_metric that averages in-to-out time during validation. Less than 10<ms is the goal.
#  7. Make dataset use the huggingface datasets library. This will allow us to upload the dataset and enable
#  streaming mode.

def setup_train_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    """
    Build train and val DataLoaders with bucketing and padding.
    """
    train_ds = AudioDataset(cfg.dataset.train, cfg.model_arch.sample_rate)
    val_ds = AudioDataset(cfg.dataset.val, cfg.model_arch.sample_rate)

    # compute raw lengths (in samples) for bucketing
    train_lengths = [torchaudio.info(p).num_frames for p in train_ds.mix_files]
    val_lengths = [torchaudio.info(p).num_frames for p in val_ds.mix_files]

    train_sampler = BucketBatchSampler(
        lengths=train_lengths,
        batch_size=cfg.training.params.batch_size,
        n_buckets=cfg.training.params.n_buckets,
        shuffle=True
    )
    val_sampler = BucketBatchSampler(
        lengths=val_lengths,
        batch_size=cfg.training.params.batch_size,
        n_buckets=cfg.training.params.n_buckets,
        shuffle=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=pad_collate,
        num_workers=cfg.training.params.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_sampler,
        collate_fn=pad_collate,
        num_workers=cfg.training.params.num_workers,
    )
    return train_loader, val_loader


def train_epoch(model: nn.Module,
                loader: DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device):
    """
    Runs one full training epoch over `loader`.
    Returns the average loss.
    """
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(loader, desc="Train", leave=False)
    for mix, refL, refR, lengths in progress_bar:
        mix, refL, refR = (mix.to(device),
                           refL.to(device),
                           refR.to(device))
        lengths = lengths.to(device)

        # reset any stateful separator once per sequence
        if hasattr(model.separator, "reset_state"):
            model.separator.reset_state(batch_size=mix.size(0))

        # forward
        estL, estR = model(mix)

        # mask out padded time‐steps in loss
        lossL = criterion(estL, refL)
        lossR = criterion(estR, refR)
        max_T = refL.size(1)
        mask = (torch.arange(max_T, device=device)[None, :] < lengths[:, None]).float()
        loss = ((lossL + lossR) * mask).sum() / mask.sum()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate_epoch(model: nn.Module,
                   loader: DataLoader,
                   criterion: nn.Module,
                   device: torch.device):
    """
    Runs one validation epoch and returns average metrics:
      - loss
      - SNR
      - SNR improvement (SNRi)
      - SI-SNR (scale-invariant)
      - SI-SNR improvement (SI-SNRi)  ← PRIMARY
    """
    model.eval()
    totals = {k: 0.0 for k in ["loss", "snr", "snr_i", "si_snr", "si_snr_i"]}
    total_examples = 0

    with torch.no_grad():
        for mix, refL, refR, lengths in tqdm(loader, desc="Validate", leave=False):
            # move everything to device
            mix, refL, refR, lengths = (
                mix.to(device),
                refL.to(device),
                refR.to(device),
                lengths.to(device),
            )
            if hasattr(model.separator, "reset_state"):
                model.separator.reset_state(batch_size=mix.size(0))
            estL, estR = model(mix)

            # compute masked loss
            max_T = refL.size(1)
            mask = (torch.arange(max_T, device=device)[None] < lengths[:, None]).float()
            loss = ((criterion(estL, refL) + criterion(estR, refR)) * mask).sum() / mask.sum()

            # bring to CPU for metrics
            estL_c, estR_c = estL.cpu(), estR.cpu()
            mixL_c, mixR_c = mix[:, 0, :].cpu(), mix[:, 1, :].cpu()
            refL_c, refR_c = refL.cpu(), refR.cpu()
            B = estL_c.size(0)

            # SNR & SNRi
            snr_L    = signal_noise_ratio(estL_c, refL_c)
            snr_R    = signal_noise_ratio(estR_c, refR_c)
            snr_est  = 0.5 * (snr_L + snr_R)
            snr_mix  = 0.5 * (
                signal_noise_ratio(mixL_c, refL_c) +
                signal_noise_ratio(mixR_c, refR_c)
            )
            snr_i = snr_est - snr_mix

            # SI-SNR & SI-SNRi
            si_snr_L    = scale_invariant_signal_distortion_ratio(
                             estL_c, refL_c, zero_mean=True)
            si_snr_R    = scale_invariant_signal_distortion_ratio(
                             estR_c, refR_c, zero_mean=True)
            si_snr_est  = 0.5 * (si_snr_L + si_snr_R)
            si_snr_mix  = 0.5 * (
                scale_invariant_signal_distortion_ratio(mixL_c, refL_c, zero_mean=True) +
                scale_invariant_signal_distortion_ratio(mixR_c, refR_c, zero_mean=True)
            )
            si_snr_i = si_snr_est - si_snr_mix

            # accumulate
            totals["loss"]     += loss.item() * B
            totals["snr"]      += snr_est.sum().item()
            totals["snr_i"]    += snr_i.sum().item()
            totals["si_snr"]   += si_snr_est.sum().item()
            totals["si_snr_i"] += si_snr_i.sum().item()
            total_examples     += B

    # average over all examples
    for k in totals:
        totals[k] /= total_examples
    return totals



@hydra.main(version_base="1.3", config_path="../configs", config_name="tasnet_baseline")
def main(cfg: DictConfig):
    device = torch.device(select_device())
    torch.manual_seed(cfg.training.params.seed)

    # build model, optimizer, scheduler, loss
    model = instantiate(cfg.model_arch).to(device)
    optimizer = instantiate(cfg.training.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.training.scheduler, optimizer=optimizer, _recursive_=False)
    criterion = instantiate(cfg.training.criterion).to(device)

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
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_stats = validate_epoch(model, val_loader, criterion, device)

        scheduler.step(val_stats["loss"])
        print(
            f"Epoch {epoch:2d} "
            f"train_loss={train_loss:.4f} val_loss={val_stats['loss']:.4f} "
            f"SI-SNRi={val_stats['si_snr_i']:.2f} SNR={val_stats['snr']:.2f}"
        )

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
        art = wandb.Artifact(run_name, type="model")
        art.add_file(best_ckpt_path)
        wandb.log_artifact(art)

    print(f"Training complete. Best {best_metric_name}: {best_metric_value:.4f}")


if __name__ == "__main__":
    main()
