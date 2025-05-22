import torch
import torch.nn as nn


def compute_mask(lengths, T, device):
    """
    lengths: [B] (on any device)
    Returns mask of shape [B, T] (on target device)
    """
    return (torch.arange(T, device=device)[None, :] < lengths.to(device)[:, None]).float()


def masked_mse(est, ref, mask):
    """
    est, ref, mask: [B, T]
    Returns average masked MSE.
    """
    loss = (est - ref) ** 2  # [B, T]
    return (loss * mask).sum() / mask.sum()


def masked_snr(est, ref, mask, eps=1e-8):
    """
    est, ref, mask: [B, T]
    Returns average negative SNR (minimize -SNR).
    """
    num = ((ref * mask) ** 2).sum(dim=1)
    den = (((est - ref) * mask) ** 2).sum(dim=1) + eps
    snr = 10 * torch.log10(num / den)  # [B]
    return -snr.mean()  # negative for minimization


class Loss(nn.Module):
    def forward(self, est, ref, lengths):
        raise NotImplementedError


class MaskedMSELoss(Loss):
    def forward(self, est, ref, lengths):
        """
        est, ref: [B, 2, T]
        lengths: [B]
        Returns average (L+R)/2 loss.
        """
        mask = compute_mask(lengths, est.size(-1), est.device)  # [B, T]
        lossL = masked_mse(est[:, 0, :], ref[:, 0, :], mask)
        lossR = masked_mse(est[:, 1, :], ref[:, 1, :], mask)
        return (lossL + lossR) / 2


class MaskedSNRLoss(Loss):
    def forward(self, est, ref, lengths):
        """
        est, ref: [B, 2, T]
        lengths: [B]
        Returns average (L+R)/2 negative SNR loss.
        """
        mask = compute_mask(lengths, est.size(-1), est.device)
        lossL = masked_snr(est[:, 0, :], ref[:, 0, :], mask)
        lossR = masked_snr(est[:, 1, :], ref[:, 1, :], mask)
        return (lossL + lossR) / 2
