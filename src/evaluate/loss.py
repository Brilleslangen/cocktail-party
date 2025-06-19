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


def masked_sdr(est, ref, mask, loss_scale=1.0, eps=1e-8):
    """
    est, ref, mask: [B, T]
    Computes negative scaled SDR loss (minimize -SDR)
    """
    # Compute the numerator as signal power
    signal_power = ((ref * mask) ** 2).sum(dim=1)

    # Compute the denominator as distortion power (includes interference and noise)
    distortion_power = (((est - ref) * mask) ** 2).sum(dim=1)

    sdr = 10 * torch.log10((signal_power + eps) / (distortion_power + eps))  # [B]
    return loss_scale * -sdr  # Negative because we minimize


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


class EnergyWeightedMaskedSDRLoss(nn.Module):
    """
    Scale-dependent SDR loss with time-masking and per-ear energy weighting.
    Intended for binaural (2-channel) target-speaker separation with variable
    sequence lengths. Minimises −SDR for each channel, then combines the
    channel losses using reference‐signal power as weights.
    """
    def __init__(self, loss_scale=1.0):
        super().__init__()
        self.loss_scale = loss_scale

    def forward(self, est, ref, lengths):
        """
        est, ref: [B, 2, T] — stereo estimated and reference signals
        lengths: [B] — number of valid timesteps per sample
        Returns: scalar loss averaged over batch
        """
        B, C, T = est.size()
        assert C == 2, "Expected stereo input with shape [B, 2, T]"
        mask = compute_mask(lengths, T, est.device)  # [B, T]

        # Compute per-channel SDR loss [B]
        lossL = masked_sdr(est[:, 0, :], ref[:, 0, :], mask, self.loss_scale)
        lossR = masked_sdr(est[:, 1, :], ref[:, 1, :], mask, self.loss_scale)

        # Energy-based weighting
        energyL = ((ref[:, 0, :] * mask) ** 2).sum(dim=1)
        energyR = ((ref[:, 1, :] * mask) ** 2).sum(dim=1)
        total_energy = energyL + energyR + 1e-8
        weightL = energyL / total_energy
        weightR = energyR / total_energy

        # Weighted loss per sample
        loss = weightL * lossL + weightR * lossR  # [B]
        return loss.mean()

