import time
from typing import Optional, Sequence, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from joblib import Parallel, delayed

try :
    from binaqual import calculate_binaqual
except ImportError:
    raise ImportError("Please install the 'binaqual' package to use BINAQUAL metrics in full evaluation..")

from src.evaluate.train_metrics import energy_weighted_si_sdr_i


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

def batch_estoi(est, ref, sample_rate=16000):
    """Extended Short-Time Objective Intelligibility (ESTOI)."""
    device = est.device
    B = est.shape[0]

    # Handle MPS by moving to CPU
    if device.type == 'mps':
        est_cpu = est.cpu().float()
        ref_cpu = ref.cpu().float()
        stoi = ShortTimeObjectiveIntelligibility(sample_rate, extended=True)

        scores = []
        for i in range(B):
            score = stoi(est_cpu[i:i + 1], ref_cpu[i:i + 1])
            scores.append(score.item())

        return torch.tensor(scores, device=device, dtype=torch.float32)
    else:
        stoi = ShortTimeObjectiveIntelligibility(sample_rate, extended=True).to(device)

        scores = []
        for i in range(B):
            score = stoi(est[i:i + 1], ref[i:i + 1])
            scores.append(score.item())

        return torch.tensor(scores, device=device)


def batch_pesq(est, ref, sample_rate=16000, mode='wb'):
    """Perceptual Evaluation of Speech Quality (PESQ)."""
    device = est.device
    B = est.shape[0]

    # Handle MPS by moving to CPU
    if device.type == 'mps':
        est_cpu = est.cpu().float()
        ref_cpu = ref.cpu().float()
        pesq = PerceptualEvaluationSpeechQuality(sample_rate, mode)

        scores = []
        for i in range(B):
            score = pesq(est_cpu[i:i + 1], ref_cpu[i:i + 1])
            scores.append(score.item())

        return torch.tensor(scores, device=device, dtype=torch.float32)
    else:
        pesq = PerceptualEvaluationSpeechQuality(sample_rate, mode).to(device)

        scores = []
        for i in range(B):
            score = pesq(est[i:i + 1], ref[i:i + 1])
            scores.append(score.item())

        return torch.tensor(scores, device=device)


def energy_weighted_estoi(est, ref, sample_rate=16000, eps=1e-8):
    """
    Energy-weighted Extended Short-Time Objective Intelligibility for stereo signals.

    Args:
        est: [B, 2, T] - estimated stereo signal
        ref: [B, 2, T] - reference stereo signal
        sample_rate: sampling rate
        eps: small value for numerical stability

    Returns:
        [B] - energy-weighted ESTOI per sample
    """
    # Compute per-channel ESTOI
    estoi_left = batch_estoi(est[:, 0, :], ref[:, 0, :], sample_rate)
    estoi_right = batch_estoi(est[:, 1, :], ref[:, 1, :], sample_rate)

    # Compute channel energies for weighting
    energy_left = (ref[:, 0, :] ** 2).sum(dim=1)
    energy_right = (ref[:, 1, :] ** 2).sum(dim=1)
    total_energy = energy_left + energy_right + eps

    # Weight by energy
    weight_left = energy_left / total_energy
    weight_right = energy_right / total_energy

    return weight_left * estoi_left + weight_right * estoi_right


def energy_weighted_pesq(est, ref, sample_rate=16000, mode='wb', eps=1e-8):
    """
    Energy-weighted Perceptual Evaluation of Speech Quality for stereo signals.

    Args:
        est: [B, 2, T] - estimated stereo signal
        ref: [B, 2, T] - reference stereo signal
        sample_rate: sampling rate
        mode: PESQ mode ('wb' or 'nb')
        eps: small value for numerical stability

    Returns:
        [B] - energy-weighted PESQ per sample
    """
    # Compute per-channel PESQ
    pesq_left = batch_pesq(est[:, 0, :], ref[:, 0, :], sample_rate, mode)
    pesq_right = batch_pesq(est[:, 1, :], ref[:, 1, :], sample_rate, mode)

    # Compute channel energies for weighting
    energy_left = (ref[:, 0, :] ** 2).sum(dim=1)
    energy_right = (ref[:, 1, :] ** 2).sum(dim=1)
    total_energy = energy_left + energy_right + eps

    # Weight by energy
    weight_left = energy_left / total_energy
    weight_right = energy_right / total_energy

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


def compute_confusion_rate(est: torch.Tensor, mix: torch.Tensor, ref: torch.Tensor, threshold_db: float = 0.0)\
        -> torch.Tensor:
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
        # Extract mono signals for this sample
        est_mono = est[b].mean(0)
        mix_mono = mix[b].mean(0)
        ref_mono = ref[b].mean(0)

        # Compute interferer reference (mix - target)
        interferer_mono = mix_mono - ref_mono

        # Compute SDR with target as reference
        sdr_target = compute_sdr_mono(est_mono, ref_mono)

        # Compute SDR with interferer as reference
        sdr_interferer = compute_sdr_mono(est_mono, interferer_mono)

        # Check if model separated the interferer instead of target
        if sdr_interferer > sdr_target + threshold_db:
            confused[b] = 1.0

    return confused


def compute_sdr_mono(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Compute Signal-to-Distortion Ratio for mono signals.

    Args:
        est: [T] - estimated signal
        ref: [T] - reference signal
        eps: small value for numerical stability

    Returns:
        SDR in dB
    """
    # Find optimal scaling
    alpha = (est * ref).sum() / ((ref ** 2).sum() + eps)

    # Compute SDR
    signal_power = (alpha * ref) ** 2
    distortion_power = (est - alpha * ref) ** 2

    sdr = 10 * torch.log10(signal_power.sum() / (distortion_power.sum() + eps))
    return sdr.item()


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
                                sample_rate: int = 16000) -> Dict[str, torch.Tensor]:
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
        'ew_si_sdr_i': energy_weighted_si_sdr_i(est, mix, ref),
        'ew_estoi': energy_weighted_estoi(est, ref, sample_rate),
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
    'compute_sdr_mono',
    'compute_rtf',
    'compute_evaluation_metrics'
]