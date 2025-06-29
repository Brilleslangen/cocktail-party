import torch
import torch.nn as nn

from src.evaluate.pkg_funcs import compute_mask, compute_energy_weights


def masked_snr_loss(est, ref, mask, loss_scale=1.0, eps=1e-8):
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


def masked_mono_mse(est, ref, mask, eps=1e-8):
    return ((est - ref) ** 2 * mask).sum(dim=1) / (mask.sum(dim=1) + eps)  # [B]


def masked_mse(est, ref, lengths):
    mask = compute_mask(lengths, est.size(-1), est.device)  # [B, T]
    lossL = masked_mono_mse(est[:, 0, :], ref[:, 0, :], mask)
    lossR = masked_mono_mse(est[:, 1, :], ref[:, 1, :], mask)
    per_channel_losses = torch.stack([lossL, lossR], dim=1)  # [B, 2]
    channel_weights = compute_energy_weights(ref, mask)  # [B, 2]

    # Weighted sum over channels, and mean over batch
    weighted_loss = (channel_weights * per_channel_losses).sum(dim=1)  # [B]
    return weighted_loss.mean()  # Scalar


class Loss(nn.Module):
    def forward(self, est, ref, lengths):
        raise NotImplementedError


class MaskedMSELoss(Loss):
    def forward(self, est, ref, lengths):
        return masked_mse(est, ref, lengths)


class EnergyWeightedMaskedSNRLoss(nn.Module):
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
        lossL = masked_snr_loss(est[:, 0, :], ref[:, 0, :], mask, self.loss_scale)
        lossR = masked_snr_loss(est[:, 1, :], ref[:, 1, :], mask, self.loss_scale)

        # Energy-based weighting
        energyL = ((ref[:, 0, :] * mask) ** 2).sum(dim=1)
        energyR = ((ref[:, 1, :] * mask) ** 2).sum(dim=1)
        total_energy = energyL + energyR + 1e-8
        weightL = energyL / total_energy
        weightR = energyR / total_energy

        # Weighted loss per sample
        loss = weightL * lossL + weightR * lossR  # [B]
        return loss.mean()
