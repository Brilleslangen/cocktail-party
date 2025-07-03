import time
from typing import Optional, Sequence, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from torchmetrics.audio import PerceptualEvaluationSpeechQuality, ShortTimeObjectiveIntelligibility
from src.evaluate.train_metrics import compute_si_sdr_i, compute_SI_SDRs, compute_SDRs, \
    per_sample_sdr
from src.evaluate.pkg_funcs import compute_energy_weights, parallel_batch_metric_with_lengths

try:
    from binaqual import calculate_binaqual
except ImportError:
    calculate_binaqual = None


# ============================================================================
# Computational Metrics
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_macs(model: nn.Module, seconds: float = 1.0) -> int:
    device = next(model.parameters()).device
    sr = getattr(model, "sample_rate", 16000)

    streaming = getattr(model, "streaming_mode", False)
    print('Streaming mode:', streaming)

    if streaming:
        # Dummy input shaped [batch=1, channels=2, time]
        dummy = torch.randn(1, 2, model.input_size, device=device)

        with FlopCounterMode(display=False) as fcm:
            model(dummy)

        flops_per_input = fcm.get_total_flops()

        # Number of inference passes required per second
        windows_per_second = sr / model.output_size

        flops_total = flops_per_input * windows_per_second

    else:
        samples = int(sr * seconds)
        dummy = torch.randn(1, 2, samples, device=device)
        with FlopCounterMode(display=False) as fcm:
            model(dummy)
        flops_total = fcm.get_total_flops()

    macs = flops_total / 2  # PyTorch counts each MAC as 2 FLOPs
    return int(macs)


# ============================================================================
# Perceptual Metrics
# ============================================================================

def per_sample_energy_weighted_estoi(
        reference: torch.Tensor,  # [C, L]
        estimate: torch.Tensor,  # [C, L]
        sample_rate: int = 16000,
        eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute energy-weighted ESTOI for a single stereo signal pair.

    For a single sample (trimmed to its true length), compute the ESTOI for each channel,
    weight each channel's score by its relative energy, and return the weighted sum.

    Args:
        reference:   torch.Tensor of shape [C, L]
                     Reference signal, C = number of channels, L = length.
        estimate:    torch.Tensor of shape [C, L]
                     Estimated signal, same shape as reference.
        sample_rate: int
                     Audio sample rate (default: 16000).
        eps:         float
                     Small constant for numerical stability in energy weighting.

    Returns:
        torch.Tensor (scalar)
            Energy-weighted ESTOI score for this sample (float).
    """
    num_channels = reference.size(0)
    channel_estoi_scores = []

    for channel_idx in range(num_channels):
        ref_ch = reference[channel_idx].unsqueeze(0)  # [1, L]
        est_ch = estimate[channel_idx].unsqueeze(0)  # [1, L]

        est_ch = est_ch.to(torch.float32)
        ref_ch = ref_ch.to(torch.float32)

        estoi_metric = ShortTimeObjectiveIntelligibility(sample_rate, extended=True)

        score = estoi_metric(est_ch, ref_ch).item()
        channel_estoi_scores.append(score)
    channel_weights = compute_energy_weights(reference, mask=None, eps=eps)  # [C]
    return (channel_weights * torch.tensor(channel_estoi_scores, device=channel_weights.device)).sum()



def energy_weighted_estoi(
        estimate: torch.Tensor,  # [B, C, T]
        reference: torch.Tensor,  # [B, C, T]
        lengths: torch.Tensor,
        sample_rate: int = 16000,
        eps: float = 1e-8,
        n_jobs: int = -1,
) -> torch.Tensor:
    """
    Compute energy-weighted ESTOI for a batch of variable-length, stereo or multi-channel signals.

    For each sample in the batch, trim the signals to their true length, compute the energy-weighted
    ESTOI (see per_sample_energy_weighted_estoi), and return a tensor of results.

    Args:
        estimate:    torch.Tensor of shape [B, C, T]
                     Estimated signals, where B is batch size, C is number of channels, T is max length.
        reference:   torch.Tensor of shape [B, C, T]
                     Reference signals, same shape as estimate.
        lengths:     torch.Tensor of shape [B]
                     True (unpadded) length for each sample in the batch.
        sample_rate: int
                     Audio sample rate (default: 16000).
        eps:         float
                     Small constant for numerical stability in energy weighting.
        n_jobs:      int
                     Number of parallel jobs to run (0=serial, -1=all cores, >0=explicit).

    Returns:
        torch.Tensor of shape [B]
            Energy-weighted ESTOI scores, one per sample in the batch.
    """
    move_to_cpu = estimate.device.type == 'mps'
    return parallel_batch_metric_with_lengths(
        lambda ref_trim, est_trim: per_sample_energy_weighted_estoi(
            ref_trim, est_trim, sample_rate=sample_rate, eps=eps),
        estimate, reference, lengths, n_jobs=n_jobs, move_to_cpu=move_to_cpu
    )


def per_sample_energy_weighted_pesq(
        reference: torch.Tensor,  # [C, L]
        estimate: torch.Tensor,  # [C, L]
        sample_rate: int = 16000,
        mode: str = 'wb',
        eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute energy-weighted PESQ for a single stereo (or multi-channel) signal pair.

    Args:
        reference:   torch.Tensor of shape [C, L]
        estimate:    torch.Tensor of shape [C, L]
        sample_rate: int, sample rate for PESQ metric
        mode:       str, 'wb' (wideband) or 'nb' (narrowband)
        eps:        float, for numerical stability in energy weighting

    Returns:
        torch.Tensor (scalar): Energy-weighted PESQ score for this sample.
    """
    num_channels = reference.size(0)
    channel_pesq_scores = []

    for channel_idx in range(num_channels):
        ref_ch = reference[channel_idx].unsqueeze(0)  # [1, L]
        est_ch = estimate[channel_idx].unsqueeze(0)  # [1, L]

        pesq_metric = PerceptualEvaluationSpeechQuality(sample_rate, mode)

        score = pesq_metric(est_ch.to(torch.float32), ref_ch.to(torch.float32)).item()

        channel_pesq_scores.append(score)
    channel_weights = compute_energy_weights(reference, mask=None, eps=eps)  # [C]
    return (channel_weights * torch.tensor(channel_pesq_scores, device=channel_weights.device)).sum()



def energy_weighted_pesq(
        estimate: torch.Tensor,  # [B, C, T]
        reference: torch.Tensor,  # [B, C, T]
        lengths: torch.Tensor,
        sample_rate: int = 16000,
        mode: str = 'wb',
        eps: float = 1e-8,
        n_jobs: int = -1,
) -> torch.Tensor:
    """
    Compute energy-weighted PESQ for a batch of variable-length, stereo/multichannel signals.

    Args:
        estimate:    torch.Tensor [B, C, T]
        reference:   torch.Tensor [B, C, T]
        lengths:     torch.Tensor [B]
        sample_rate: int
        mode:       str, 'wb' or 'nb'
        eps:        float
        n_jobs:     int

    Returns:
        torch.Tensor [B]: Energy-weighted PESQ scores for each sample.
    """
    move_to_cpu = estimate.device.type == 'mps'
    return parallel_batch_metric_with_lengths(
        lambda ref_trim, est_trim: per_sample_energy_weighted_pesq(
            ref_trim, est_trim, sample_rate=sample_rate, mode=mode, eps=eps),
        estimate, reference, lengths, n_jobs=n_jobs, move_to_cpu=move_to_cpu
    )


# ============================================================================
# Spatial Metrics
# ============================================================================

def per_sample_binaqual(
        reference: torch.Tensor,  # [2, L]
        estimate: torch.Tensor,  # [2, L]
) -> float:
    """
    Compute BINAQUAL localisation-similarity for a single stereo sample.
    """
    # Energy threshold specifically calibrated for speech

    _, ls = calculate_binaqual(reference.permute(1, 0), estimate.permute(1, 0))

    return ls


def compute_binaqual(
        estimate: torch.Tensor,  # [B, 2, T]
        reference: torch.Tensor,  # [B, 2, T]
        lengths: torch.Tensor,  # [B]
        n_jobs: int = -1  # parallelism
) -> torch.Tensor:
    """
    Batched BINAQUAL localisation-similarity for variable-length, stereo signals.

    Args:
        estimate:  [B, 2, T] - estimated signals
        reference: [B, 2, T] - reference signals
        lengths:   [B]       - true (unpadded) length per sample
        n_jobs:    int       - number of parallel jobs (0=serial, -1=all cores, >0=explicit)

    Returns:
        torch.Tensor [B] (float32, device matches input): BINAQUAL LS score for each sample
    """
    return parallel_batch_metric_with_lengths(
        per_sample_binaqual,
        estimate, reference, lengths,
        n_jobs=n_jobs, move_to_cpu=True  # always move to cpu for binaqual
    )


def per_sample_confusion_rate(
        estimate: torch.Tensor,  # [2, L]
        mixture: torch.Tensor,  # [2, L]
        reference: torch.Tensor,  # [2, L]
        threshold_db: float = 0.0,
        eps: float = 1e-8
) -> float:
    """
    For a single trimmed sample, return 1.0 if confused (interferer SDR > target SDR + threshold), else 0.0.
    """
    # Compute interferer reference
    interferer = mixture - reference

    # SDR with target as reference
    sdr_target = per_sample_sdr(reference, estimate, eps=eps)
    # SDR with interferer as reference
    sdr_interferer = per_sample_sdr(interferer, estimate, eps=eps)

    return float(sdr_interferer > sdr_target + threshold_db)


def compute_confusion_rate(
        estimate: torch.Tensor,  # [B, 2, T]
        mixture: torch.Tensor,  # [B, 2, T]
        reference: torch.Tensor,  # [B, 2, T]
        lengths: torch.Tensor,
        threshold_db: float = 1.0,
        eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute the confusion rate for a batch of estimated stereo signals in a speech separation setting.

    For each sample in the batch, the function determines whether the separation model has
    mistakenly separated the interferer (non-target audio) instead of the target source.

    This is measured by computing the energy-weighted SDR of the model's estimate
    relative to:
      - the true target signal (reference)
      - the interferer signal (mixture - reference)

    If the SDR with respect to the interferer is higher than the SDR with respect to the target
    by more than `threshold_db`, the model is said to be "confused" for that sample.

    Args:
        estimate: [B, 2, T] - Batch of estimated stereo signals.
        mixture:  [B, 2, T] - Batch of original mixture stereo signals.
        reference: [B, 2, T] - Batch of reference (target) stereo signals.
        lengths:   [B] - Lengths of valid samples for each batch item (for variable-length trimming).
        threshold_db: float - Margin (in dB) by which the interferer SDR must exceed the target SDR to count as confusion.
        eps:        float - Numerical stability term.

    Returns:
        confusion: [B] tensor of floats, 1.0 if confused (separated the interferer), else 0.0 for each sample.

    Example:
        confusion = compute_confusion_rate(estimate, mixture, reference, lengths)
        confusion_rate = confusion.mean().item()  # Proportion of confused samples in the batch

    Notes:
        - This function is batch- and length-aware, supporting variable-length and parallelized SDR computation.
        - Useful for evaluating model robustness against speaker confusion in multi-speaker separation tasks.
    """
    B = estimate.size(0)

    # Compute interferer reference
    interferer = mixture - reference

    # Compute SDR for target and interferer
    sdr_target = compute_SDRs(estimate=estimate, reference=reference, lengths=lengths, multi_channel=True, eps=eps)
    sdr_interferer = compute_SDRs(estimate=estimate, reference=interferer, lengths=lengths, multi_channel=True, eps=eps)

    # Only keep indices where both are not NaN to maintain valid comparisons
    # valid_mask = (~torch.isnan(sdr_target)) & (~torch.isnan(sdr_interferer))
    # sdr_target_valid = sdr_target[valid_mask]
    # sdr_interferer_valid = sdr_interferer[valid_mask]

    confusion = torch.zeros(B, device=estimate.device)

    for b in range(B):
        # Defensive: check if both SDRs are nan (shouldn't happen if data is valid!)
        if torch.isnan(sdr_target[b]) and torch.isnan(sdr_interferer[b]):
            print(f"Sample {b}: Both SDRs are NaN! This should not happen.")
            confusion[b] = 0.0
        elif torch.isnan(sdr_interferer[b]):
            confusion[b] = 0.0  # If interferer SDR is NaN, not confused
        elif torch.isnan(sdr_target[b]):
            # This should NOT happen if estimate==reference, but:
            print(f"Sample {b}: Target SDR is NaN with estimate==reference!")
            confusion[b] = 0.0  # Or 1.0, but this should be flagged as a data bug
        else:
            confusion[b] = float(sdr_interferer[b] > sdr_target[b] + threshold_db)

    return confusion


def compute_rtf(model: nn.Module, audio_duration: float, batch_size: int = 1,
                num_runs: int = 10, device: torch.device = None) -> float:
    """
    Compute Real-Time Factor (RTF) - processing_time / audio_duration.
    RTF < 1.0 means faster than real-time.

    Args:
        model: the model to evaluate
        audio_duration: duration of audio to process in seconds
        batch_size: batch size for inference
        num_runs: number of runs for averaging
        device: torch device

    Returns:
        RTF value
    """
    if device is None:
        device = next(model.parameters()).device

    sample_rate = model.sample_rate
    num_samples = int(audio_duration * sample_rate)

    # Create dummy input
    dummy_input = torch.randn(batch_size, 2, num_samples, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_input)

    # Time the inference
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    avg_time = np.mean(times)
    rtf = avg_time / audio_duration

    return rtf


def compute_evaluation_metrics(est: torch.Tensor, mix: torch.Tensor, ref: torch.Tensor,
                               sample_rate: int, lengths: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute all evaluation metrics with energy weighting where appropriate.

    Args:
        est: [B, 2, T] - estimated stereo signal
        mix: [B, 2, T] - mixture stereo signal
        ref: [B, 2, T] - reference stereo signal
        sample_rate: audio sample rate

    Returns:
        dict with metric names and [B] tensor values
    """
    metrics = {
        'mc_si_sdr': compute_SI_SDRs(est, ref, lengths, multi_channel=True),
        'mc_si_sdr_i': compute_si_sdr_i(est, mix, ref, lengths, multi_channel=True),
        'ew_si_sdr': compute_SI_SDRs(est, ref, lengths, multi_channel=False, eps=1e-8),
        'ew_si_sdr_i': compute_si_sdr_i(est, mix, ref, lengths, multi_channel=False, eps=1e-8),
        'ew_estoi': energy_weighted_estoi(est, ref, lengths, sample_rate),
        'ew_pesq': energy_weighted_pesq(est, ref, lengths, sample_rate),
        'binaqual': compute_binaqual(est, ref, lengths),
        'confusion_rate': compute_confusion_rate(est, mix, ref, lengths)
    }

    return metrics


# Export all public functions
__all__ = [
    'count_parameters',
    'count_macs',
    'energy_weighted_estoi',
    'compute_binaqual',
    'compute_confusion_rate',
    'compute_rtf',
    'compute_evaluation_metrics'
]
