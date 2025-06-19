import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from torchmetrics.audio import ShortTimeObjectiveIntelligibility


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
