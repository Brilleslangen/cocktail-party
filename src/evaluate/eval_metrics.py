import time
from typing import Optional, Sequence, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from torchmetrics.audio import PerceptualEvaluationSpeechQuality, ShortTimeObjectiveIntelligibility
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio
from joblib import Parallel, delayed
from binaqual import calculate_binaqual
from src.evaluate.train_metrics import energy_weighted_si_sdr_i, energy_weighted_si_sdr, energy_weighted_sdr
from src.evaluate.loss import compute_energy_weights


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

def batch_estoi(est, ref, sample_rate):
    """Extended Short-Time Objective Intelligibility (ESTOI)."""
    device = est.device
    B = est.shape[0]

    # Handle MPS by moving to CPU
    if device.type == 'mps':
        est = est.cpu().float()
        ref = ref.cpu().float()

    estoi = ShortTimeObjectiveIntelligibility(sample_rate, extended=True)

    scores = []
    for i in range(B):
        score = estoi(est[i:i + 1], ref[i:i + 1])
        scores.append(score.item())

    return torch.tensor(scores, device=device, dtype=torch.float32)


def batch_pesq(est, ref, sample_rate, mode='wb'):
    """Perceptual Evaluation of Speech Quality (PESQ)."""
    device = est.device
    B = est.shape[0]

    # Handle MPS by moving to CPU
    if device.type == 'mps':
        est = est.cpu().float()
        ref = ref.cpu().float()

    pesq = PerceptualEvaluationSpeechQuality(sample_rate, mode).to(device)

    scores = []
    for i in range(B):
        score = pesq(est[i:i + 1], ref[i:i + 1])
        scores.append(score.item())

    return torch.tensor(scores, device=device)


def energy_weighted_estoi(est, ref, sample_rate, lengths: torch.Tensor, eps=1e-8):
    """
    Energy-weighted Extended Short-Time Objective Intelligibility for stereo signals.

    Args:
        lengths:
        est: [B, 2, T] - estimated stereo signal
        ref: [B, 2, T] - reference stereo signal
        sample_rate: sampling rate
        eps: small value for numerical stability

    Returns:
        [B] - energy-weighted ESTOI per sample
    """

    # Trim to original lengths
    for L in lengths:
        est = est[:, :, :L]
        ref = ref[:, :, :L]

    # Compute per-channel ESTOI
    estoi_left = batch_estoi(est[:, 0, :], ref[:, 0, :], sample_rate)
    estoi_right = batch_estoi(est[:, 1, :], ref[:, 1, :], sample_rate)

    # Weight by energy
    weight_left, weight_right = compute_energy_weights(ref, lengths)

    return weight_left * estoi_left + weight_right * estoi_right


def energy_weighted_pesq(est, ref, sample_rate, lengths: torch.Tensor, mode='wb', eps=1e-8):
    """
    Energy-weighted Perceptual Evaluation of Speech Quality for stereo signals.

    Args:
        lengths:
        est: [B, 2, T] - estimated stereo signal
        ref: [B, 2, T] - reference stereo signal
        sample_rate: sampling rate
        mode: PESQ mode ('wb' or 'nb')
        eps: small value for numerical stability

    Returns:
        [B] - energy-weighted PESQ per sample
    """
    # Trim to original lengths
    for L in lengths:
        est = est[:, :, :L]
        ref = ref[:, :, :L]

    # Compute per-channel PESQ
    pesq_left = batch_pesq(est[:, 0, :], ref[:, 0, :], sample_rate, mode)
    pesq_right = batch_pesq(est[:, 1, :], ref[:, 1, :], sample_rate, mode)

    # Weight by energy
    weight_left, weight_right = compute_energy_weights(ref, lengths)

    return weight_left * pesq_left + weight_right * pesq_right


def _prep_one(
        ref: torch.Tensor,  # [2, T]             (on any device / dtype)
        est: torch.Tensor,  # [2, T]
) -> float:
    """
    Helper that converts a *single* (ref, est) pair to NumPy,
    calls `calculate_binaqual`, and returns the localisation-similarity (LS).
    Runs on CPU so it is safe inside a joblib worker.
    """
    ref_np = ref.permute(1, 0).contiguous().cpu().float().numpy()
    est_np = est.permute(1, 0).contiguous().cpu().float().numpy()

    _, ls = calculate_binaqual(ref_np, est_np)
    return ls


def compute_binaqual(
        est: torch.Tensor,  # [B, 2, T]
        ref: torch.Tensor,  # [B, 2, T]
        lengths: torch.Tensor,
        n_jobs: int = 0  # 0 = serial, -1 = "all cores", >0 = explicit
) -> torch.Tensor:
    """
    Batched BINAQUAL localisation-similarity.

    If `n_jobs` ≠ 0 the B items are processed in parallel with joblib.

    Returns
    -------
    torch.Tensor shaped [B] (float32, on the same device as `est`/`ref`).
    """
    if est.shape != ref.shape:
        raise ValueError("est and ref must have the same shape [B, 2, T].")
    if est.size(1) != 2:
        raise ValueError("BINAQUAL is defined for stereo signals (C == 2).")

    # Trim to original lengths
    for L in lengths:
        est = est[:, :, :L]
        ref = ref[:, :, :L]

    B, _, _ = est.shape

    # ----------------------------------------------------------------------
    # choose serial vs. parallel execution
    # ----------------------------------------------------------------------
    if n_jobs == 0 or B == 1:
        # plain Python loop – minimal overhead, fine for small batches
        ls_values: Sequence[float] = [
            _prep_one(ref[b], est[b])
            for b in range(B)
        ]
    else:
        # parallel worker pool (fork or spawn depending on OS)
        ls_values = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_prep_one)(ref[b], est[b])
            for b in range(B)
        )

    # ----------------------------------------------------------------------
    return torch.tensor(ls_values, dtype=torch.float32, device=est.device)


def compute_confusion_rate(est: torch.Tensor, mix: torch.Tensor, ref: torch.Tensor, lengths: torch.Tensor,
                           threshold_db: float = 0.0) -> torch.Tensor:
    """
    Compute confusion rate based on SDR difference threshold.
    A sample is confused if separating the interferer gives better SDR than separating the target.

    Args:
        est: [B, 2, T] - estimated stereo signal
        mix: [B, 2, T] - mixture stereo signal
        ref: [B, 2, T] - reference stereo signal (target)
        threshold_db: SDR difference threshold for confusion detection

    Returns:
        [B] - binary confusion indicator per sample (1 if confused, 0 if correct)
    """
    B = est.shape[0]
    confused = torch.zeros(B, device=est.device)

    for b in range(B):
        # Trim to true lengths
        L = int(lengths[b])
        mix = mix[b, :, :L]
        est = est[b, :, :L]
        ref = ref[b, :, :L]

        # Compute interferer reference (mix - target)
        interferer = mix - ref

        # Compute SDR with target as reference
        sdr_target = energy_weighted_sdr(est, ref, lengths=None, eps=1e-8)

        # Compute SDR with interferer as reference
        sdr_interferer = energy_weighted_sdr(interferer, ref, lengths=None, eps=1e-8)

        # Check if model separated the interferer instead of target
        if sdr_interferer > sdr_target + threshold_db:
            confused[b] = 1.0

    return confused


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


@torch.no_grad()
def energy_weighted_si_sdr_eval(est, ref, lengths, eps=1e-8):
    """
    Energy-weighted SI-SDR for stereo output, batchwise.

    Args:
        eps:
        est: [B, 2, T] - estimated signals (possibly padded)
        ref: [B, 2, T] - reference signals (possibly padded)
        lengths: [B]   - valid (unpadded) lengths per sample

    Returns:
        [B] tensor of weighted SI-SDRs for each sample
    """
    B, C, T = est.shape
    assert C == 2

    si_sdr_L = []
    si_sdr_R = []

    for b in range(B):
        L = int(lengths[b])
        estL = est[b, 0, :L].float()
        refL = ref[b, 0, :L].float()
        estR = est[b, 1, :L].float()
        refR = ref[b, 1, :L].float()
        # Each returns scalar SI-SDR
        sdr_L = scale_invariant_signal_distortion_ratio(estL, refL, zero_mean=True)
        sdr_R = scale_invariant_signal_distortion_ratio(estR, refR, zero_mean=True)
        si_sdr_L.append(sdr_L)
        si_sdr_R.append(sdr_R)

    si_sdr_L = torch.stack(si_sdr_L)  # [B]
    si_sdr_R = torch.stack(si_sdr_R)  # [B]

    # Compute energy-based weights (using unpadded signals)
    energy_L = torch.tensor([ref[b, 0, :int(lengths[b])].float().pow(2).sum() for b in range(B)], device=est.device)
    energy_R = torch.tensor([ref[b, 1, :int(lengths[b])].float().pow(2).sum() for b in range(B)], device=est.device)
    total_energy = energy_L + energy_R + eps
    wL = energy_L / total_energy
    wR = energy_R / total_energy

    # Weighted SI-SDR per sample
    weighted_sisdr = wL * si_sdr_L + wR * si_sdr_R  # [B]
    return weighted_sisdr


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
        'ew_si_sdr_new': energy_weighted_si_sdr_eval(est, ref, lengths=lengths),
        'ew_si_sdr': energy_weighted_si_sdr(est, ref),
        'ew_si_sdr_i': energy_weighted_si_sdr_i(est, mix, ref, lengths),
        'ew_estoi': energy_weighted_estoi(est, ref, sample_rate, lengths),
        'ew_pesq': energy_weighted_pesq(est, ref, sample_rate),
        'binaqual': compute_binaqual(est, ref, n_jobs=-1),
        'confusion_rate': compute_confusion_rate(est, mix, ref)
    }

    return metrics


# Export all public functions
__all__ = [
    'count_parameters',
    'count_macs',
    'batch_estoi',
    'batch_pesq',
    'energy_weighted_estoi',
    'energy_weighted_pesq',
    'compute_binaqual',
    'compute_confusion_rate',
    'compute_rtf',
    'compute_evaluation_metrics'
]
