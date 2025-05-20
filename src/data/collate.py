# src/data/collate.py
import torch
from torch.nn import functional as F


def pad_collate(batch):
    mixes, refs = zip(*batch)
    lengths = torch.tensor([m.shape[1] for m in mixes], dtype=torch.long)
    T_max = max(lengths)

    mixes_padded = torch.stack([F.pad(m, (0, T_max - m.size(1))) for m in mixes])
    refs_padded = torch.stack([F.pad(r, (0, T_max - r.size(0))) for r in refs])

    return mixes_padded, refs_padded, lengths
