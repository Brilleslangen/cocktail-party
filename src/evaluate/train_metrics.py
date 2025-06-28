# src/evaluate/metrics.py
import torch
from src.evaluate.loss import compute_energy_weights, compute_mask, masked_mse
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio, signal_distortion_ratio

from joblib import Parallel, delayed


def batch_metric_with_lengths(
        metric_func,  # Callable: takes (ref_trimmed, est_trimmed)
        est: torch.Tensor,  # [B, C, T]
        ref: torch.Tensor,  # [B, C, T]
        lengths: torch.Tensor,  # [B]
        n_jobs: int = 0,  # 0=serial, -1=all cores, >0=explicit
        move_to_cpu: bool = False,  # Move to CPU if necessary
        device: torch.device = None
) -> torch.Tensor:
    """
    Compute a per-sample metric on [B, C, T] tensors, trimming by lengths.
    metric_func should take (ref_trimmed, est_trimmed), both [C, L], and return a float.
    """
    B, C, T = est.shape
    device = device or est.device

    def _prep_one_wrapper(b):
        L = int(lengths[b])
        ref_trim = ref[b, :, :L]
        est_trim = est[b, :, :L]

        if move_to_cpu:
            ref_trim = ref_trim.cpu()
            est_trim = est_trim.cpu()

        return metric_func(ref_trim, est_trim)

    if n_jobs == 0 or B == 1:
        results = [_prep_one_wrapper(b) for b in range(B)]
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_prep_one_wrapper)(b)
            for b in range(B)
        )

    return torch.tensor(results, dtype=torch.float32, device=device)


# ============================================================================
# Energy-Weighted Metrics (for training)
# ============================================================================

def per_sample_energy_weighted_sdr(
        reference: torch.Tensor,
        estimate: torch.Tensor,
        eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute energy-weighted SDR for a single stereo (or multi-channel) pair of signals.

    Args:
        reference: [C, L] reference signal (trimmed)
        estimate: [C, L] estimated signal (trimmed)
        eps: small value for numerical stability

    Returns:
        Scalar energy-weighted SDR value for the sample.
    """
    channel_sdrs = signal_distortion_ratio(
        estimate.unsqueeze(0),  # [1, C, L]
        reference.unsqueeze(0),  # [1, C, L]
        zero_mean=True,
    ).squeeze(0)  # [C]

    channel_weights = compute_energy_weights(reference, mask=None, eps=eps)  # [C]
    return (channel_weights * channel_sdrs).sum()


def energy_weighted_sdr(
        estimate: torch.Tensor,  # [B, C, T]
        reference: torch.Tensor,  # [B, C, T]
        lengths: torch.Tensor,
        eps: float = 1e-8,
        n_jobs: int = 0):
    """
    Efficient, correct batch energy-weighted SDR for stereo/multi-channel and variable-length signals.
    Returns [B] (per sample).
    """
    move_to_cpu = estimate.device.type == 'mps'
    return batch_metric_with_lengths(
        lambda ref_trim, est_trim: per_sample_energy_weighted_sdr(ref_trim, est_trim, eps=eps),
        estimate, reference, lengths, n_jobs=n_jobs, move_to_cpu=move_to_cpu)


def per_sample_energy_weighted_si_sdr(
        reference: torch.Tensor,  # [C, L]
        estimate: torch.Tensor,   # [C, L]
        eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute energy-weighted SI-SDR for a single stereo (or multi-channel) pair of signals.
    Uses torchmetrics implementation.
    """
    channel_si_sdrs = scale_invariant_signal_distortion_ratio(
        estimate.unsqueeze(0),   # [1, C, L]
        reference.unsqueeze(0),  # [1, C, L]
        zero_mean=True,
    ).squeeze(0)                 # [C]

    channel_weights = compute_energy_weights(reference, mask=None, eps=eps)  # [C]
    return (channel_weights * channel_si_sdrs).sum()


def energy_weighted_si_sdr(
        estimate: torch.Tensor,  # [B, C, T]
        reference: torch.Tensor,  # [B, C, T]
        lengths: torch.Tensor,
        eps: float = 1e-8,
        n_jobs: int = 0
) -> torch.Tensor:
    """
    Energy-weighted SI-SDR for variable-length, stereo/multi-channel batches.
    Returns [B] (per-sample energy-weighted SI-SDR).
    """
    move_to_cpu = estimate.device == 'mps'
    return batch_metric_with_lengths(
        lambda ref_trim, est_trim: per_sample_energy_weighted_si_sdr(ref_trim, est_trim, eps=eps),
        estimate, reference, lengths, n_jobs=n_jobs, move_to_cpu=move_to_cpu)


def energy_weighted_si_sdr_i(est, mix, ref, lengths: torch.Tensor, eps=1e-8):
    """
    Energy-weighted Scale-Invariant Signal-to-Distortion Ratio Improvement.

    Args:
        lengths:
        est: [B, 2, T] - estimated stereo signal
        mix: [B, 2, T] - mixture stereo signal
        ref: [B, 2, T] - reference stereo signal
        mask: [B, T] - valid samples mask
        eps: small value for numerical stability

    Returns:
        [B] - energy-weighted SI-SDR improvement in dB per sample
    """

    si_sdr_est = energy_weighted_si_sdr(est, ref, lengths=lengths, eps=eps)
    si_sdr_mix = energy_weighted_si_sdr(mix, ref, lengths=lengths, eps=eps)
    return si_sdr_est - si_sdr_mix


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
    metrics = {
        'ew_mse': masked_mse(est, ref, lengths),
        'ew_sdr': energy_weighted_sdr(est, ref, lengths),
        'ew_si_sdr': energy_weighted_si_sdr(est, ref, lengths),
        'ew_si_sdr_i': energy_weighted_si_sdr_i(est, mix, ref, lengths)
    }

    return metrics
