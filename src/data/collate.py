# src/data/collate.py
import torch
from torch.nn.utils.rnn import pad_sequence


def pad_collate(batch):
    """
    batch: list of tuples (mix:[2,T_i], left:[T_i], right:[T_i])
    Returns:
      mix_p:  [B,2,T_max]
      left_p: [B,T_max]
      right_p:[B,T_max]
      lengths:[T_i list]
    """
    mixes, lefts, rights = zip(*batch)
    # record original lengths
    lengths = [m.shape[1] for m in mixes]
    T_max = max(lengths)

    # pad mixtures → [B,2,T_max]
    # we do pad_sequence on time dimension, so first transpose to [T,2]
    mixes_t = [m.transpose(0, 1) for m in mixes]  # list of [T_i,2]
    mix_p = pad_sequence(mixes_t, batch_first=True)  # [B,T_max,2]
    mix_p = mix_p.transpose(1, 2)  # →[B,2,T_max]

    # pad left/right → [B,T_max]
    lefts_p = pad_sequence([l for l in lefts], batch_first=True)
    rights_p = pad_sequence([r for r in rights], batch_first=True)

    return mix_p, lefts_p, rights_p, torch.tensor(lengths, dtype=torch.long)


def no_pad_collate(batch):
    mixes, lefts, rights = zip(*batch)
    lengths = [m.shape[1] for m in mixes]
    mix_batch = torch.stack(mixes, dim=0)  # [B,2,T_i] ragged
    left_batch = torch.stack(lefts, dim=0)
    right_batch = torch.stack(rights, dim=0)
    return mix_batch, left_batch, right_batch, torch.tensor(lengths)
