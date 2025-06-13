import torch
import warnings

# Try to import audio metrics
try:
    from torchmetrics.audio import PerceptualEvaluationSpeechQuality
    from torchmetrics.audio import ShortTimeObjectiveIntelligibility

    AUDIO_METRICS_AVAILABLE = True
except ImportError:
    AUDIO_METRICS_AVAILABLE = False
    warnings.warn(
        "Audio metrics (PESQ, STOI) not available. "
        "Install with: pip install torchmetrics[audio] or pip install pesq pystoi"
    )


def batch_snr(est, ref, mask, eps=1e-8):
    """
    Signal-to-Noise Ratio (SNR) in dB.

    Args:
        est: [B, T] - estimated signal
        ref: [B, T] - reference signal
        mask: [B, T] - valid samples mask
        eps: small value to avoid log(0)

    Returns:
        [B] SNR values in dB
    """
    # Mask the signals
    est_masked = est * mask
    ref_masked = ref * mask

    # Noise is the difference between estimate and reference
    noise = est_masked - ref_masked

    # Compute powers
    signal_power = (ref_masked ** 2).sum(dim=1)
    noise_power = (noise ** 2).sum(dim=1) + eps

    # SNR in dB
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr


def batch_sdr(est, ref, mask, eps=1e-8):
    """
    Signal-to-Distortion Ratio (SDR) in dB.
    Uses projection to find the scaled reference that best matches the estimate.

    Args:
        est: [B, T] - estimated signal
        ref: [B, T] - reference signal
        mask: [B, T] - valid samples mask
        eps: small value for numerical stability

    Returns:
        [B] SDR values in dB
    """
    # Apply mask
    est_masked = est * mask
    ref_masked = ref * mask

    # Find the scaling factor that minimizes ||est - alpha * ref||^2
    # alpha = <est, ref> / <ref, ref>
    num = torch.sum(est_masked * ref_masked, dim=1, keepdim=True)
    den = torch.sum(ref_masked ** 2, dim=1, keepdim=True) + eps
    alpha = num / den

    # Scaled reference (target)
    s_target = alpha * ref_masked

    # Distortion (everything else)
    e_distortion = est_masked - s_target

    # SDR calculation
    target_power = (s_target ** 2).sum(dim=1)
    distortion_power = (e_distortion ** 2).sum(dim=1) + eps

    sdr = 10 * torch.log10(target_power / distortion_power)
    return sdr


def batch_si_snr(est, ref, mask, eps=1e-8):
    """
    Scale-Invariant Signal-to-Noise Ratio (SI-SNR) in dB.

    Args:
        est: [B, T] - estimated signal
        ref: [B, T] - reference signal
        mask: [B, T] - valid samples mask
        eps: small value for numerical stability

    Returns:
        [B] SI-SNR values in dB
    """
    # Zero-mean normalization
    mask_sum = mask.sum(dim=1, keepdim=True) + eps
    est_zm = est * mask - (est * mask).sum(dim=1, keepdim=True) / mask_sum
    ref_zm = ref * mask - (ref * mask).sum(dim=1, keepdim=True) / mask_sum

    # Find the projection scale
    # s_target = <est_zm, ref_zm> / <ref_zm, ref_zm> * ref_zm
    dot_product = torch.sum(est_zm * ref_zm, dim=1, keepdim=True)
    ref_energy = torch.sum(ref_zm ** 2, dim=1, keepdim=True) + eps
    proj_scale = dot_product / ref_energy

    # Projected target
    s_target = proj_scale * ref_zm

    # Noise
    e_noise = est_zm - s_target

    # SI-SNR calculation
    target_power = (s_target ** 2).sum(dim=1)
    noise_power = (e_noise ** 2).sum(dim=1) + eps

    si_snr = 10 * torch.log10(target_power / noise_power)
    return si_snr


def batch_si_sdr(est, ref, mask, eps=1e-8):
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB.
    For speech separation, SI-SDR is typically the same as SI-SNR.

    Args:
        est: [B, T] - estimated signal
        ref: [B, T] - reference signal
        mask: [B, T] - valid samples mask
        eps: small value for numerical stability

    Returns:
        [B] SI-SDR values in dB
    """
    # SI-SDR is the same as SI-SNR for most speech separation tasks
    return batch_si_snr(est, ref, mask, eps)


def batch_estoi(est, ref, sample_rate=16000):
    """
    Extended Short-Time Objective Intelligibility (ESTOI).

    Args:
        est: [B, T] - estimated signal
        ref: [B, T] - reference signal
        sample_rate: sampling rate in Hz

    Returns:
        [B] ESTOI scores (range 0-1)
    """
    if not AUDIO_METRICS_AVAILABLE:
        # Return placeholder values if metrics not available
        B = est.shape[0]
        warnings.warn("ESTOI not available, returning zeros. Install torchmetrics[audio]")
        return torch.zeros(B, device=est.device)

    device = est.device
    B = est.shape[0]

    # Handle MPS by moving to CPU (MPS doesn't support float64)
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
        # Initialize ESTOI metric
        stoi = ShortTimeObjectiveIntelligibility(sample_rate, extended=True).to(device)

        # Process each sample in the batch
        scores = []
        for i in range(B):
            # STOI expects [batch, time] format
            score = stoi(est[i:i + 1], ref[i:i + 1])
            scores.append(score.item())

        return torch.tensor(scores, device=device)


def batch_pesq(est, ref, sample_rate=16000, mode='wb'):
    """
    Perceptual Evaluation of Speech Quality (PESQ).

    Args:
        est: [B, T] - estimated signal
        ref: [B, T] - reference signal
        sample_rate: sampling rate in Hz
        mode: 'wb' for wideband (16kHz) or 'nb' for narrowband (8kHz)

    Returns:
        [B] PESQ scores (range -0.5 to 4.5)
    """
    if not AUDIO_METRICS_AVAILABLE:
        # Return placeholder values if metrics not available
        B = est.shape[0]
        warnings.warn("PESQ not available, returning zeros. Install torchmetrics[audio]")
        return torch.zeros(B, device=est.device)

    device = est.device
    B = est.shape[0]

    # Handle MPS by moving to CPU (MPS doesn't support float64)
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
        # Initialize PESQ metric
        pesq = PerceptualEvaluationSpeechQuality(sample_rate, mode).to(device)

        # Process each sample in the batch
        scores = []
        for i in range(B):
            # PESQ expects [batch, time] format
            score = pesq(est[i:i + 1], ref[i:i + 1])
            scores.append(score.item())

        return torch.tensor(scores, device=device)


# Improvement metrics (suffix 'i' for improvement)
def batch_snr_i(est, mix, ref, mask, eps=1e-8):
    """SNR improvement: SNR(est, ref) - SNR(mix, ref)"""
    snr_est = batch_snr(est, ref, mask, eps)
    snr_mix = batch_snr(mix, ref, mask, eps)
    return snr_est - snr_mix


def batch_sdr_i(est, mix, ref, mask, eps=1e-8):
    """SDR improvement: SDR(est, ref) - SDR(mix, ref)"""
    sdr_est = batch_sdr(est, ref, mask, eps)
    sdr_mix = batch_sdr(mix, ref, mask, eps)
    return sdr_est - sdr_mix


def batch_si_snr_i(est, mix, ref, mask, eps=1e-8):
    """SI-SNR improvement: SI-SNR(est, ref) - SI-SNR(mix, ref)"""
    si_snr_est = batch_si_snr(est, ref, mask, eps)
    si_snr_mix = batch_si_snr(mix, ref, mask, eps)
    return si_snr_est - si_snr_mix


def batch_si_sdr_i(est, mix, ref, mask, eps=1e-8):
    """SI-SDR improvement: SI-SDR(est, ref) - SI-SDR(mix, ref)"""
    si_sdr_est = batch_si_sdr(est, ref, mask, eps)
    si_sdr_mix = batch_si_sdr(mix, ref, mask, eps)
    return si_sdr_est - si_sdr_mix


def compute_all_metrics_baseline(mix, ref, mask, sample_rate=16000):
    """
    Compute all baseline metrics (mixture vs reference).
    For baseline, improvement metrics will be 0.

    Args:
        mix: [B, T] - mixture signal
        ref: [B, T] - reference clean signal
        mask: [B, T] - valid samples mask
        sample_rate: sampling rate in Hz

    Returns:
        Dictionary with all metrics as tensors
    """
    metrics = {
        # Basic metrics
        'SNR': batch_snr(mix, ref, mask),
        'SDR': batch_sdr(mix, ref, mask),
        'SI-SNR': batch_si_snr(mix, ref, mask),
        'SI-SDR': batch_si_sdr(mix, ref, mask),

        # Perceptual metrics
        'ESTOI': batch_estoi(mix, ref, sample_rate),
        'PESQ': batch_pesq(mix, ref, sample_rate),

        # Improvement metrics (0 for baseline)
        'SNRi': torch.zeros(mix.shape[0], device=mix.device),
        'SDRi': torch.zeros(mix.shape[0], device=mix.device),
        'SI-SNRi': torch.zeros(mix.shape[0], device=mix.device),
        'SI-SDRi': torch.zeros(mix.shape[0], device=mix.device),
    }

    return metrics


def compute_ild_error(mix_stereo, ref_stereo, eps=1e-8):
    """
    Compute Interaural Level Difference (ILD) error.

    Args:
        mix_stereo: [2, T] - stereo mixture
        ref_stereo: [2, T] - stereo reference
        eps: small value for numerical stability

    Returns:
        ILD MSE and correlation
    """
    # Compute ILD in dB
    ild_mix = 20 * torch.log10((mix_stereo[0].abs() + eps) / (mix_stereo[1].abs() + eps))
    ild_ref = 20 * torch.log10((ref_stereo[0].abs() + eps) / (ref_stereo[1].abs() + eps))

    # Remove infinite values
    valid_mask = torch.isfinite(ild_mix) & torch.isfinite(ild_ref)
    ild_mix = ild_mix[valid_mask]
    ild_ref = ild_ref[valid_mask]

    if len(ild_mix) == 0:
        return 0.0, 1.0

    # MSE
    ild_mse = ((ild_mix - ild_ref) ** 2).mean().item()

    # Correlation
    if len(ild_mix) > 1:
        ild_mix_centered = ild_mix - ild_mix.mean()
        ild_ref_centered = ild_ref - ild_ref.mean()

        numerator = (ild_mix_centered * ild_ref_centered).sum()
        denominator = torch.sqrt((ild_mix_centered ** 2).sum() * (ild_ref_centered ** 2).sum())

        if denominator > 0:
            ild_corr = (numerator / denominator).item()
        else:
            ild_corr = 0.0
    else:
        ild_corr = 1.0

    return ild_mse, ild_corr


def compute_itd_error(mix_stereo, ref_stereo, sample_rate=16000, max_lag_ms=1.0):
    """
    Compute Interaural Time Difference (ITD) error using cross-correlation.

    Args:
        mix_stereo: [2, T] - stereo mixture
        ref_stereo: [2, T] - stereo reference
        sample_rate: sampling rate in Hz
        max_lag_ms: maximum lag to consider in milliseconds

    Returns:
        ITD MSE in milliseconds
    """
    max_lag_samples = int(max_lag_ms * sample_rate / 1000)

    # Compute ITD for mixture
    xcorr_mix = torch.nn.functional.conv1d(
        mix_stereo[0:1].unsqueeze(0),
        mix_stereo[1:2].unsqueeze(0).flip(-1),
        padding=max_lag_samples
    ).squeeze()

    # Find peak
    peak_idx_mix = xcorr_mix.argmax()
    itd_mix_samples = peak_idx_mix - max_lag_samples
    itd_mix_ms = itd_mix_samples.float() / sample_rate * 1000

    # Compute ITD for reference
    xcorr_ref = torch.nn.functional.conv1d(
        ref_stereo[0:1].unsqueeze(0),
        ref_stereo[1:2].unsqueeze(0).flip(-1),
        padding=max_lag_samples
    ).squeeze()

    peak_idx_ref = xcorr_ref.argmax()
    itd_ref_samples = peak_idx_ref - max_lag_samples
    itd_ref_ms = itd_ref_samples.float() / sample_rate * 1000

    # MSE in milliseconds
    itd_mse = ((itd_mix_ms - itd_ref_ms) ** 2).item()

    return itd_mse


def compute_ipd_error(mix_stereo, ref_stereo, sample_rate=16000, n_fft=512):
    """
    Compute Interaural Phase Difference (IPD) error.

    Args:
        mix_stereo: [2, T] - stereo mixture
        ref_stereo: [2, T] - stereo reference
        sample_rate: sampling rate in Hz
        n_fft: FFT size

    Returns:
        IPD MSE and correlation
    """
    # Compute STFT
    window = torch.hann_window(n_fft, device=mix_stereo.device)

    # Mix STFT
    mix_l_stft = torch.stft(mix_stereo[0], n_fft, hop_length=n_fft // 4,
                            window=window, return_complex=True)
    mix_r_stft = torch.stft(mix_stereo[1], n_fft, hop_length=n_fft // 4,
                            window=window, return_complex=True)

    # Ref STFT
    ref_l_stft = torch.stft(ref_stereo[0], n_fft, hop_length=n_fft // 4,
                            window=window, return_complex=True)
    ref_r_stft = torch.stft(ref_stereo[1], n_fft, hop_length=n_fft // 4,
                            window=window, return_complex=True)

    # Compute IPD
    ipd_mix = torch.angle(mix_l_stft) - torch.angle(mix_r_stft)
    ipd_ref = torch.angle(ref_l_stft) - torch.angle(ref_r_stft)

    # Wrap to [-pi, pi]
    ipd_mix = torch.atan2(torch.sin(ipd_mix), torch.cos(ipd_mix))
    ipd_ref = torch.atan2(torch.sin(ipd_ref), torch.cos(ipd_ref))

    # Compute MSE
    ipd_diff = torch.atan2(torch.sin(ipd_mix - ipd_ref), torch.cos(ipd_mix - ipd_ref))
    ipd_mse = (ipd_diff ** 2).mean().item()

    # Compute correlation (using cosine similarity of complex phasors)
    mix_phasor = torch.exp(1j * ipd_mix)
    ref_phasor = torch.exp(1j * ipd_ref)
    ipd_corr = torch.real(mix_phasor * torch.conj(ref_phasor)).mean().item()

    return ipd_mse, ipd_corr


def compute_binaural_metrics(mix_stereo, ref_stereo, sample_rate=16000):
    """
    Compute all binaural preservation metrics.

    Args:
        mix_stereo: [2, T] - stereo mixture
        ref_stereo: [2, T] - stereo reference
        sample_rate: sampling rate in Hz

    Returns:
        Dictionary with binaural metrics
    """
    ild_mse, ild_corr = compute_ild_error(mix_stereo, ref_stereo)
    itd_mse = compute_itd_error(mix_stereo, ref_stereo, sample_rate)
    ipd_mse, ipd_corr = compute_ipd_error(mix_stereo, ref_stereo, sample_rate)

    return {
        'ILD_MSE': ild_mse,
        'ILD_corr': ild_corr,
        'ITD_MSE': itd_mse,
        'IPD_MSE': ipd_mse,
        'IPD_corr': ipd_corr
    }