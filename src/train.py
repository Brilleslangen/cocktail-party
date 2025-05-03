import torch
import torchaudio
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch import nn
from torch.utils.data import DataLoader
from src.data.dataset import AudioDataset
from src.helpers import select_device
from src.data.collate import pad_collate
from src.data.bucket_sampler import BucketBatchSampler


# TODO: Check training loop with dataset.
# TODO: Check if what loss should be used. Currently using MSELoss.
# TODO: Connect wandb for logging.
# TODO: Add naming scheme for the model. Use hydra -> set name property in yaml or use model name + dataset.

def setup_dataloaders(cfg: DictConfig):
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

    for mix, refL, refR, lengths in loader:
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
    Runs one full validation epoch over `loader`.
    Returns the average loss.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for mix, refL, refR, lengths in loader:
            mix, refL, refR = (mix.to(device),
                               refL.to(device),
                               refR.to(device))
            lengths = lengths.to(device)

            if hasattr(model.separator, "reset_state"):
                model.separator.reset_state(batch_size=mix.size(0))

            estL, estR = model(mix)
            lossL = criterion(estL, refL)
            lossR = criterion(estR, refR)
            max_T = refL.size(1)
            mask = (torch.arange(max_T, device=device)[None, :] < lengths[:, None]).float()
            loss = ((lossL + lossR) * mask).sum() / mask.sum()

            total_loss += loss.item()

    return total_loss / len(loader)


@hydra.main(version_base="1.3", config_path="../configs", config_name="tasnet_baseline")
def main(cfg: DictConfig):
    # 1) print and save the config
    print(OmegaConf.to_yaml(cfg))

    # 2) set device
    device = torch.device(select_device())

    # 3) build model, optimizer, scheduler, loss
    model = instantiate(cfg.model_arch).to(device)
    optimizer = instantiate(cfg.training.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.training.scheduler, optimizer=optimizer, _recursive_=False)
    criterion = nn.MSELoss(reduction="none")

    # 4) dataloaders
    train_loader, val_loader = setup_dataloaders(cfg)

    best_val = float("inf")
    for epoch in range(1, cfg.training.params.max_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        print(f"Epoch {epoch: 2d}  train_loss={train_loss: .4f}  val_loss={val_loss: .4f}")

        # checkpoint best
        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "model_state": model.state_dict(),
                "opt_state": optimizer.state_dict(),
                "epoch": epoch,
                "cfg": OmegaConf.to_container(cfg),
            }
            torch.save(ckpt, "outputs/<name>best_model.pt")

    print("Training complete. Best val loss:", best_val)


if __name__ == "__main__":
    main()
