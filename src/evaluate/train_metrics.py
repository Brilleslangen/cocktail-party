# src/evaluate/metrics.py
import torch
from src.evaluate.pkg_funcs import parallel_batch_metric_with_lengths, compute_energy_weights
from src.evaluate.loss import masked_mse
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio, signal_distortion_ratio

G = torch.Generator()
G.manual_seed(42)


def _perfect_estimate_noise(estimate: torch.Tensor, device: torch.device) -> torch.Tensor:
    return estimate + torch.randn(estimate.shape, generator=G, device=device) * 1e-6


def per_sample_sdr(
        reference: torch.Tensor,  # [C, L]
        estimate: torch.Tensor,  # [C, L]
        multi_channel: bool = True,
        eps: float = 1e-9,
        seed=42
) -> torch.Tensor:
    """
    Compute SDR for a single stereo (or multichannel) pair of signals.

    Args:
        seed:
        reference: [C, L] reference signal (trimmed)
        estimate:  [C, L] estimated signal (trimmed)
        multi_channel: If True, flatten all channels and compute SDR as one signal (MC-SDR).
                       If False, compute SDR per channel and average with energy weights (EW-SDR).
        eps: Small value for numerical stability.

    Returns:
        Scalar SDR value for the sample.
    """
    # Check if the reference and estimate are identical and add some noise to avoid divison by zero, returning nan.

    # This is a workaround for the case when the reference and estimate are identical,

    if torch.allclose(reference, estimate, atol=1e-8):
        estimate = _perfect_estimate_noise(estimate, reference.device)

    if multi_channel:
        # Flatten channels and time
        ref_flat = reference.reshape(1, -1)  # [1, C*L]
        est_flat = estimate.reshape(1, -1)  # [1, C*L]
        sdr = signal_distortion_ratio(est_flat, ref_flat, zero_mean=True)
        return sdr.squeeze(0)  # scalar
    else:
        # Compute energy-weighted SDR
        channel_sdrs = signal_distortion_ratio(
            estimate.unsqueeze(0),  # [1, C, L]
            reference.unsqueeze(0),  # [1, C, L]
            zero_mean=True,
        ).squeeze(0)  # [C]
        channel_weights = compute_energy_weights(reference, mask=None, eps=eps)  # [C]
        return (channel_weights * channel_sdrs).sum()


def compute_SDRs(
        estimate: torch.Tensor,  # [B, C, T]
        reference: torch.Tensor,  # [B, C, T]
        lengths: torch.Tensor,
        multi_channel: bool,
        eps: float = 1e-8,
        n_jobs: int = -1):
    """
    Parallel, batch energy-weighted SDR for stereo/multichannel and variable-length signals.
    Returns [B] (per sample).
    """
    move_to_cpu = estimate.device.type == 'mps'
    return parallel_batch_metric_with_lengths(
        lambda ref_trim, est_trim: per_sample_sdr(ref_trim, est_trim, multi_channel, eps=eps),
        estimate, reference, lengths, n_jobs=n_jobs, move_to_cpu=move_to_cpu, filter_nans=False)


def per_sample_SI_SDR(
        reference: torch.Tensor,  # [C, L]
        estimate: torch.Tensor,  # [C, L]
        multi_channel: bool = True,
        eps: float = 1e-7,
        seed=42
) -> torch.Tensor:
    """
    Compute SI-SDR for a single stereo (or multichannel) pair of signals.

    Args:
        reference: [C, L] reference signal (trimmed)
        estimate:  [C, L] estimated signal (trimmed)
        multi_channel: If True, flatten all channels and compute SI-SDR as one signal (MC-SI-SDR).
                       If False, compute SI-SDR per channel and average with energy weights (EW-SI-SDR).
        eps: Small value for numerical stability.

    Returns:
        Scalar SI-SDR value for the sample.
    """
    if reference.abs().sum() < 1e-8:
        raise ValueError("Reference signal is all zeros, cannot compute SI-SDR.")

    # Check if the reference and estimate are identical and add some noise to avoid divison by zero, returning nan.
    if torch.allclose(reference, estimate, atol=1e-8):
        estimate = _perfect_estimate_noise(estimate, reference.device)

    if multi_channel:
        # Flatten channels and time (treat as one long signal)
        ref_flat = reference.reshape(1, -1)  # [1, C*L]
        est_flat = estimate.reshape(1, -1)  # [1, C*L]
        si_sdr = scale_invariant_signal_distortion_ratio(est_flat, ref_flat, zero_mean=True)
        return si_sdr.squeeze(0)  # scalar
    else:
        # Compute energy-weighted SI-SDR
        channel_si_sdrs = scale_invariant_signal_distortion_ratio(
            estimate.unsqueeze(0),  # [1, C, L]
            reference.unsqueeze(0),  # [1, C, L]
            zero_mean=True,
        ).squeeze(0)  # [C]
        channel_weights = compute_energy_weights(reference, mask=None, eps=eps)  # [C]
        return (channel_weights * channel_si_sdrs).sum()


def compute_SI_SDRs(
        estimate: torch.Tensor,  # [B, C, T]
        reference: torch.Tensor,  # [B, C, T]
        lengths: torch.Tensor,
        multi_channel: bool,
        eps: float = 1e-8,
        n_jobs: int = -1
) -> torch.Tensor:
    """
    Parallel Energy-weighted SI-SDR for variable-length, stereo/multichannel batches.
    Returns [B] (per-sample energy-weighted SI-SDR).
    """
    move_to_cpu = estimate.device.type == 'mps'
    return parallel_batch_metric_with_lengths(
        lambda ref_trim, est_trim: per_sample_SI_SDR(ref_trim, est_trim, multi_channel, eps=eps),
        estimate, reference, lengths, n_jobs=n_jobs, move_to_cpu=move_to_cpu)


def compute_si_sdr_i(est, mix, ref, lengths: torch.Tensor, multi_channel: bool, eps=1e-8):
    """
    Scale-Invariant Signal-to-Distortion Ratio Improvement.

    Args:
        multi_channel:
        lengths:
        est: [B, 2, T] - estimated stereo signal
        mix: [B, 2, T] - mixture stereo signal
        ref: [B, 2, T] - reference stereo signal
        mask: [B, T] - valid samples mask
        eps: small value for numerical stability

    Returns:
        [B] - SI-SDR improvement (est - mix) and SI-SDR of the estimate.
    """

    si_sdr_est = compute_SI_SDRs(est, ref, lengths, multi_channel, eps=eps)
    si_sdr_mix = compute_SI_SDRs(mix, ref, lengths, multi_channel, eps=eps)
    return si_sdr_est - si_sdr_mix, si_sdr_est


def compute_validation_metrics(est, mix, ref, lengths: torch.Tensor):
    """
    Compute all energy-weighted metrics for validation.

    Args:
        lengths:
        est: [B, 2, T] - estimated stereo signal
        mix: [B, 2, T] - mixture stereo signal
        ref: [B, 2, T] - reference stereo signal
        mask: [B, T] - valid samples mask

    Returns:
        dict with metric names and [B] tensor values
    """
    est.float(), mix.float(), ref.float()
    mc_si_sdri, mc_si_sdr_est = compute_si_sdr_i(est, mix, ref, lengths, multi_channel=True)
    ew_si_sdri, ew_si_sdr_est = compute_si_sdr_i(est, mix, ref, lengths, multi_channel=False)

    metrics = {
        'ew_mse': masked_mse(est, ref, lengths),
        'mc_sdr': compute_SDRs(est, ref, lengths, multi_channel=True),
        'mc_si_sdr': mc_si_sdr_est,
        'mc_si_sdr_i': mc_si_sdri,
        'ew_sdr': compute_SDRs(est, ref, lengths, multi_channel=False),
        'ew_si_sdr': ew_si_sdr_est,
        'ew_si_sdr_i': ew_si_sdri,
    }

    return metrics
