from torchmetrics.functional.audio import (
    signal_noise_ratio,
    scale_invariant_signal_noise_ratio,
    signal_distortion_ratio,
    scale_invariant_signal_distortion_ratio,
)


def compute_SNR(estL_c, estR_c, mixL_c, mixR_c, refL_c, refR_c) -> (float, float):
    snr_L = signal_noise_ratio(estL_c, refL_c)
    snr_R = signal_noise_ratio(estR_c, refR_c)
    snr_est = 0.5 * (snr_L + snr_R)
    snr_mix = 0.5 * (signal_noise_ratio(mixL_c, refL_c) + signal_noise_ratio(mixR_c, refR_c))
    snr_i = snr_est - snr_mix

    return snr_est, snr_i


def compute_SI_SNR(estL_c, refL_c, estR_c, refR_c, mixL_c, mixR_c) -> (float, float):
    si_snr_L = scale_invariant_signal_distortion_ratio(estL_c, refL_c, zero_mean=True)
    si_snr_R = scale_invariant_signal_distortion_ratio(estR_c, refR_c, zero_mean=True)
    si_snr_est = 0.5 * (si_snr_L + si_snr_R)
    si_snr_mix = 0.5 * (scale_invariant_signal_distortion_ratio(mixL_c, refL_c, zero_mean=True) +
                        scale_invariant_signal_distortion_ratio(mixR_c, refR_c, zero_mean=True))
    si_snr_i = si_snr_est - si_snr_mix

    return si_snr_est, si_snr_i


def compute_SI_SDR(estL_c, refL_c, estR_c, refR_c, mixL_c, mixR_c) -> (float, float):
    si_sdr_L = signal_distortion_ratio(estL_c, refL_c)
    si_sdr_R = signal_distortion_ratio(estR_c, refR_c)
    si_sdr_est = 0.5 * (si_sdr_L + si_sdr_R)
    si_sdr_mix = 0.5 * (signal_distortion_ratio(mixL_c, refL_c) + signal_distortion_ratio(mixR_c, refR_c))
    si_sdr_i = si_sdr_est - si_sdr_mix

    return si_sdr_est, si_sdr_i
