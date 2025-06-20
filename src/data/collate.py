import os

import torch
import torch.nn.functional as F
import torchaudio
import wandb
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.bucket_sampler import BucketBatchSampler
from src.data.dataset import AudioDataset
from src.helpers import using_cuda


def pad_collate(batch):
    mixes, refs = zip(*batch)
    lengths = torch.tensor([m.shape[1] for m in mixes], dtype=torch.long)
    T_max = lengths.max().item()

    # Pad each [2, T_i] to [2, T_max] along last dimension
    mixes_padded = torch.stack([F.pad(m, (0, T_max - m.size(1))) for m in mixes])
    refs_padded = torch.stack([F.pad(r, (0, T_max - r.size(1))) for r in refs])

    return mixes_padded, refs_padded, lengths


def setup_train_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader]:
    """
    Build train and val DataLoaders with bucketing and padding.
    """
    run = wandb.run
    dataset_dir = run.use_artifact(cfg.dataset.artifact_name, type="dataset").download()
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")
    pin_memory = using_cuda()

    train_ds = AudioDataset(train_dir, cfg.dataset.sample_rate)
    val_ds = AudioDataset(val_dir, cfg.dataset.sample_rate)

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
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_sampler,
        collate_fn=pad_collate,
        num_workers=cfg.training.params.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def setup_test_dataloader(cfg: DictConfig) -> DataLoader:
    """
    Build testDataLoaders with bucketing and padding.
    """
    run = wandb.run
    dataset_dir = run.use_artifact(cfg.dataset.artifact_name, type="dataset").download()
    test_dir = os.path.join(dataset_dir, "test")
    pin_memory = using_cuda()

    test_ds = AudioDataset(test_dir, cfg.dataset.sample_rate)

    # compute raw lengths (in samples) for bucketing
    test_lengths = [torchaudio.info(p).num_frames for p in test_ds.mix_files]

    test_sampler = BucketBatchSampler(
        lengths=test_lengths,
        batch_size=cfg.training.params.batch_size,
        n_buckets=cfg.training.params.n_buckets,
        shuffle=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_sampler=test_sampler,
        collate_fn=pad_collate,
        num_workers=cfg.training.params.num_workers,
        pin_memory=pin_memory,
    )

    return test_loader
