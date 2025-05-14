from torchmetrics.functional.audio import (
    signal_noise_ratio,
    scale_invariant_signal_noise_ratio,
    scale_invariant_signal_distortion_ratio,
)


def compute_SNR(estL, estR, mixL, mixR, refL, refR):
    """Compute SNR and SNR improvement (SNRi) for a batch."""
    snr_est = 0.5 * (signal_noise_ratio(estL, refL) + signal_noise_ratio(estR, refR))  # [B]
    snr_mix = 0.5 * (signal_noise_ratio(mixL, refL) + signal_noise_ratio(mixR, refR))  # [B]
    snr_i = snr_est - snr_mix  # [B]
    return snr_est, snr_i


def compute_SI_SNR(estL, estR, mixL, mixR, refL, refR):
    """Compute SI-SNR and SI-SNR improvement (SI-SNRi) for a batch."""
    si_snr_est = 0.5 * (scale_invariant_signal_noise_ratio(estL, refL) +
                        scale_invariant_signal_noise_ratio(estR, refR))  # [B]
    si_snr_mix = 0.5 * (scale_invariant_signal_noise_ratio(mixL, refL) +
                        scale_invariant_signal_noise_ratio(mixR, refR))  # [B]
    si_snr_i = si_snr_est - si_snr_mix  # [B]
    return si_snr_est, si_snr_i


def compute_SI_SDR(estL, estR, mixL, mixR, refL, refR):
    """Compute SI-SDR and SI-SDR improvement (SI-SDRi) for a batch."""
    si_sdr_est = 0.5 * (scale_invariant_signal_distortion_ratio(estL, refL, zero_mean=True) +
                        scale_invariant_signal_distortion_ratio(estR, refR, zero_mean=True))  # [B]
    si_sdr_mix = 0.5 * (scale_invariant_signal_distortion_ratio(mixL, refL, zero_mean=True) +
                        scale_invariant_signal_distortion_ratio(mixR, refR, zero_mean=True))  # [B]
    si_sdr_i = si_sdr_est - si_sdr_mix  # [B]
    return si_sdr_est, si_sdr_i
