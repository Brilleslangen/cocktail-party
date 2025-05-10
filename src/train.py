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


# TODO:
#  1. Check what loss should be used. Currently using MSELoss.
#  2. Connect wandb for logging. Add wandb config to hydra config.
#  3. Add naming scheme for the model. Use hydra -> set name property in yaml or use model name + dataset.
#  4. Create a function to calculate model size (num params).
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
    Runs one full validation epoch over `loader`.
    Returns the average loss.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():

        progress_bar = tqdm(loader, desc="Validate", leave=False)
        for mix, refL, refR, lengths in progress_bar:
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

    best_val = float("inf")
    best_ckpt_path = None

    for epoch in range(1, cfg.training.params.max_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        print(f"Epoch {epoch: 2d} train_loss={train_loss: .4f} val_loss={val_loss: .4f}")

        # log to wandb
        if cfg.wandb.enabled:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }, step=epoch)

        # checkpoint best
        if val_loss < best_val:
            best_val = val_loss
            best_ckpt_path = f"outputs/{run_name}.pt"
            torch.save({
                "model_state": model.state_dict(),
                "opt_state": optimizer.state_dict(),
                "epoch": epoch,
                "cfg": cfg_dict,
            }, best_ckpt_path)

    # upload file if W&B is enabled
    if cfg.wandb.enabled and best_ckpt_path is not None:
        wandb.run.summary["best/val_loss"] = best_val
        art = wandb.Artifact(run_name, type="model")
        art.add_file(best_ckpt_path)
        wandb.log_artifact(art)

    print("Training complete. Best val loss:", best_val)


if __name__ == "__main__":
    main()
