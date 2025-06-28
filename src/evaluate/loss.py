import torch
import torch.nn as nn


def compute_mask(lengths, T, device):
    """
    lengths: [B] (on any device)
    Returns mask of shape [B, T] (on target device)
    """
    return (torch.arange(T, device=device)[None, :] < lengths.to(device)[:, None]).float()


def compute_energy_weights(
        ref,  # [B, C, T], or list of [C, L_b], or [C, L]
        mask=None,
        eps=1e-8,
):
    """
    Compute energy-based weights for multi-channel signals.

    Args:
        ref:   [B, C, T] batch, or list of [C, L_b] tensors, or [C, L]
        mask:  [B, T], [L], or None
        eps:   float

    Returns:
        weights: [B, C] if batch/list, [C] if single sample
    """
    if isinstance(ref, list):  # Ragged batch: list of [C, L_b]
        weights = []
        for r in ref:
            channel_energy = (r ** 2).sum(dim=1)  # [C]
            total_energy = channel_energy.sum() + eps
            weights.append(channel_energy / total_energy)  # [C]
        return torch.stack(weights, dim=0)  # [B, C]
    elif ref.ndim == 3:  # [B, C, T]
        if mask is not None:
            channel_energy = ((ref * mask.unsqueeze(1)) ** 2).sum(dim=2)  # [B, C]
        else:
            channel_energy = (ref ** 2).sum(dim=2)  # [B, C]
        total_energy = channel_energy.sum(dim=1, keepdim=True) + eps  # [B, 1]
        return channel_energy / total_energy  # [B, C]
    elif ref.ndim == 2:  # [C, L]
        if mask is not None:
            channel_energy = ((ref * mask.unsqueeze(0)) ** 2).sum(dim=1)  # [C]
        else:
            channel_energy = (ref ** 2).sum(dim=1)  # [C]
        total_energy = channel_energy.sum() + eps
        return channel_energy / total_energy  # [C]
    else:
        raise ValueError(
            f"ref must be [C, L], [B, C, T], or list of [C, L_b]. Got {type(ref)}, shape {getattr(ref, 'shape', None)}."
        )


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
