import os
import time

import torch
import torchaudio
import hydra
import wandb
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch import nn
from torch.utils.data import DataLoader
from src.data.dataset import AudioDataset
from src.helpers import select_device, count_parameters, prettify_param_count
from src.data.collate import pad_collate
from src.data.bucket_sampler import BucketBatchSampler
from src.helpers.helpers import format_time
from src.helpers.metrics import compute_SNR, compute_SI_SNR, compute_SI_SDR


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
    run = wandb.run
    dataset_dir = run.use_artifact(cfg.dataset.artifact_name, type="dataset").download()
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")

    train_ds = AudioDataset(train_dir, cfg.model_arch.sample_rate)
    val_ds = AudioDataset(val_dir, cfg.model_arch.sample_rate)

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
      - SI-SNR improvement (SI-SNRi)
      - SI-SDR
      - SI-SDR improvement (SI-SDRi)
    """
    model.eval()
    totals = {k: 0.0 for k in ["loss", "snr", "snr_i", "si_snr", "si_snr_i", "si_sdr", "si_sdr_i"]}
    total_examples = 0

    with torch.no_grad():
        for mix, refL, refR, lengths in tqdm(loader, desc="Validate", leave=False):
            # move batch to device
            mix, refL, refR, lengths = (
                mix.to(device),
                refL.to(device),
                refR.to(device),
                lengths.to(device),
            )

            if hasattr(model.separator, "reset_state"):
                model.separator.reset_state(batch_size=mix.size(0))

            # Forward pass
            estL, estR = model(mix)

            # Masked loss computation
            max_T = refL.size(1)
            mask = (torch.arange(max_T, device=device)[None] < lengths[:, None]).float()
            loss = ((criterion(estL, refL) + criterion(estR, refR)) * mask).sum() / mask.sum()

            # Bring to CPU for metric computation
            estL_c, estR_c = estL.cpu(), estR.cpu()
            mix_c = mix.cpu()
            refL_c, refR_c = refL.cpu(), refR.cpu()

            B = estL_c.size(0)

            # Compute metrics per-example (cropped)
            for b in range(B):
                T = lengths[b].item()

                estL_b = estL_c[b, :T]
                estR_b = estR_c[b, :T]
                mixL_b = mix_c[b, 0, :T]
                mixR_b = mix_c[b, 1, :T]
                refL_b = refL_c[b, :T]
                refR_b = refR_c[b, :T]

                # Metric computations
                snr_est, snr_i = compute_SNR(estL_b.unsqueeze(0), estR_b.unsqueeze(0),
                                             mixL_b.unsqueeze(0), mixR_b.unsqueeze(0),
                                             refL_b.unsqueeze(0), refR_b.unsqueeze(0))
                si_snr_est, si_snr_i = compute_SI_SNR(estL_b.unsqueeze(0), estR_b.unsqueeze(0),
                                                      mixL_b.unsqueeze(0), mixR_b.unsqueeze(0),
                                                      refL_b.unsqueeze(0), refR_b.unsqueeze(0))
                si_sdr_est, si_sdr_i = compute_SI_SDR(estL_b.unsqueeze(0), estR_b.unsqueeze(0),
                                                      mixL_b.unsqueeze(0), mixR_b.unsqueeze(0),
                                                      refL_b.unsqueeze(0), refR_b.unsqueeze(0))

                # Accumulate
                totals["snr"] += snr_est.sum().item()
                totals["snr_i"] += snr_i.sum().item()
                totals["si_snr"] += si_snr_est.sum().item()
                totals["si_snr_i"] += si_snr_i.sum().item()
                totals["si_sdr"] += si_sdr_est.sum().item()
                totals["si_sdr_i"] += si_sdr_i.sum().item()

            totals["loss"] += loss.item() * B
            total_examples += B

    # Average over all examples
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
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_stats = validate_epoch(model, val_loader, criterion, device)
        time_elapsed = format_time(time.time() - start_time)

        scheduler.step(val_stats["loss"])
        print(
            f"\rEpoch {epoch:2d} time={time_elapsed} train_loss={train_loss:.4f} val_loss={val_stats['loss']:.4f} " +
            f"SI-SNR={val_stats['si_snr']:.2f} SI-SNRi={val_stats['si_snr_i']:.2f} SNR={val_stats['snr']:.2f} " +
            f"SI-SDRi={val_stats['si_sdr']:.2f} SI-SDR={val_stats['si_sdr_i']:.2f}"
        )

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
