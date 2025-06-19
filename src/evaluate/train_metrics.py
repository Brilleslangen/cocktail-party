# src/evaluate/metrics.py
import torch


def compute_mask(lengths, T, device):
    """
    Create a binary mask for variable-length sequences.

    Args:
        lengths: [B] - actual sequence lengths
        T: maximum sequence length
        device: torch device

    Returns:
        mask: [B, T] - binary mask (1 for valid, 0 for padding)
    """
    return (torch.arange(T, device=device)[None, :] < lengths.to(device)[:, None]).float()


# ============================================================================
# Energy-Weighted Metrics (for training)
# ============================================================================

def energy_weighted_mse(est, ref, mask, eps=1e-8):
    """
    Energy-weighted Mean Squared Error for stereo signals.

    Args:
        est: [B, 2, T] - estimated stereo signal
        ref: [B, 2, T] - reference stereo signal
        mask: [B, T] - valid samples mask
        eps: small value for numerical stability

    Returns:
        [B] - energy-weighted MSE per sample
    """
    # Compute per-channel MSE
    mse_left = ((est[:, 0, :] - ref[:, 0, :]) ** 2 * mask).sum(dim=1) / (mask.sum(dim=1) + eps)
    mse_right = ((est[:, 1, :] - ref[:, 1, :]) ** 2 * mask).sum(dim=1) / (mask.sum(dim=1) + eps)

    # Compute channel energies
    energy_left = ((ref[:, 0, :] * mask) ** 2).sum(dim=1)
    energy_right = ((ref[:, 1, :] * mask) ** 2).sum(dim=1)
    total_energy = energy_left + energy_right + eps

    # Weight by energy
    weight_left = energy_left / total_energy
    weight_right = energy_right / total_energy

    return weight_left * mse_left + weight_right * mse_right


def energy_weighted_sdr(est, ref, mask, eps=1e-8):
    """
    Energy-weighted Signal-to-Distortion Ratio for stereo signals.

    Args:
        est: [B, 2, T] - estimated stereo signal
        ref: [B, 2, T] - reference stereo signal
        mask: [B, T] - valid samples mask
        eps: small value for numerical stability

    Returns:
        [B] - energy-weighted SDR in dB per sample
    """

    # Compute per-channel SDR
    def channel_sdr(est_ch, ref_ch, mask):
        # Find optimal scaling factor
        num = (est_ch * ref_ch * mask).sum(dim=1, keepdim=True)
        den = ((ref_ch * mask) ** 2).sum(dim=1, keepdim=True) + eps
        alpha = num / den

        # Scaled reference
        s_target = alpha * ref_ch * mask

        # Distortion
        e_distortion = (est_ch - ref_ch) * mask

        # SDR
        target_power = (s_target ** 2).sum(dim=1)
        distortion_power = (e_distortion ** 2).sum(dim=1) + eps

        return 10 * torch.log10(target_power / distortion_power)

    sdr_left = channel_sdr(est[:, 0, :], ref[:, 0, :], mask)
    sdr_right = channel_sdr(est[:, 1, :], ref[:, 1, :], mask)

    # Compute channel energies for weighting
    energy_left = ((ref[:, 0, :] * mask) ** 2).sum(dim=1)
    energy_right = ((ref[:, 1, :] * mask) ** 2).sum(dim=1)
    total_energy = energy_left + energy_right + eps

    # Weight by energy
    weight_left = energy_left / total_energy
    weight_right = energy_right / total_energy

    return weight_left * sdr_left + weight_right * sdr_right


def energy_weighted_si_sdr(est, ref, mask, eps=1e-8):
    """
    Energy-weighted Scale-Invariant Signal-to-Distortion Ratio for stereo signals.

    Args:
        est: [B, 2, T] - estimated stereo signal
        ref: [B, 2, T] - reference stereo signal
        mask: [B, T] - valid samples mask
        eps: small value for numerical stability

    Returns:
        [B] - energy-weighted SI-SDR in dB per sample
    """

    # Compute per-channel SI-SDR
    def channel_si_sdr(est_ch, ref_ch, mask):
        # Zero-mean normalization
        mask_sum = mask.sum(dim=1, keepdim=True) + eps
        est_zm = est_ch * mask - (est_ch * mask).sum(dim=1, keepdim=True) / mask_sum
        ref_zm = ref_ch * mask - (ref_ch * mask).sum(dim=1, keepdim=True) / mask_sum

        # Projection
        dot_product = (est_zm * ref_zm).sum(dim=1, keepdim=True)
        ref_energy = (ref_zm ** 2).sum(dim=1, keepdim=True) + eps
        proj_scale = dot_product / ref_energy

        # Target and noise
        s_target = proj_scale * ref_zm
        e_noise = est_zm - s_target

        # SI-SDR
        target_power = (s_target ** 2).sum(dim=1)
        noise_power = (e_noise ** 2).sum(dim=1) + eps

        return 10 * torch.log10(target_power / noise_power)

    si_sdr_left = channel_si_sdr(est[:, 0, :], ref[:, 0, :], mask)
    si_sdr_right = channel_si_sdr(est[:, 1, :], ref[:, 1, :], mask)

    # Compute channel energies for weighting
    energy_left = ((ref[:, 0, :] * mask) ** 2).sum(dim=1)
    energy_right = ((ref[:, 1, :] * mask) ** 2).sum(dim=1)
    total_energy = energy_left + energy_right + eps

    # Weight by energy
    weight_left = energy_left / total_energy
    weight_right = energy_right / total_energy

    return weight_left * si_sdr_left + weight_right * si_sdr_right


def energy_weighted_si_sdr_i(est, mix, ref, mask, eps=1e-8):
    """
    Energy-weighted Scale-Invariant Signal-to-Distortion Ratio Improvement.

    Args:
        est: [B, 2, T] - estimated stereo signal
        mix: [B, 2, T] - mixture stereo signal
        ref: [B, 2, T] - reference stereo signal
        mask: [B, T] - valid samples mask
        eps: small value for numerical stability

    Returns:
        [B] - energy-weighted SI-SDR improvement in dB per sample
    """
    si_sdr_est = energy_weighted_si_sdr(est, ref, mask, eps)
    si_sdr_mix = energy_weighted_si_sdr(mix, ref, mask, eps)
    return si_sdr_est - si_sdr_mix


def compute_validation_metrics(est, mix, ref, mask):
    """
    Compute all energy-weighted metrics for validation.

    Args:
        est: [B, 2, T] - estimated stereo signal
        mix: [B, 2, T] - mixture stereo signal
        ref: [B, 2, T] - reference stereo signal
        mask: [B, T] - valid samples mask

    Returns:
        dict with metric names and [B] tensor values
    """
    metrics = {
        'ew_mse': energy_weighted_mse(est, ref, mask),
        'ew_sdr': energy_weighted_sdr(est, ref, mask),
        'ew_si_sdr': energy_weighted_si_sdr(est, ref, mask),
        'ew_si_sdr_i': energy_weighted_si_sdr_i(est, mix, ref, mask)
    }

    return metrics
