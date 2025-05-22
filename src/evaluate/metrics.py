import torch


def batch_snr(est, ref, mask, eps=1e-8):
    """
    est, ref, mask: [B, T]
    Returns: [B] SNR (dB) for each item
    """
    signal_power = ((ref * mask) ** 2).sum(dim=1)
    noise_power = (((est - ref) * mask) ** 2).sum(dim=1) + eps
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr


def batch_si_snr(est, ref, mask, eps=1e-8):
    """
    est, ref, mask: [B, T]
    Returns: [B] SI-SNR (dB) for each item
    """
    # Zero-mean
    mask_sum = mask.sum(dim=1, keepdim=True)
    est_zm = est * mask - ((est * mask).sum(dim=1, keepdim=True) / (mask_sum + eps))
    ref_zm = ref * mask - ((ref * mask).sum(dim=1, keepdim=True) / (mask_sum + eps))

    # Projection
    s_target = (torch.sum(est_zm * ref_zm, dim=1, keepdim=True) / (
                torch.sum(ref_zm ** 2, dim=1, keepdim=True) + eps)) * ref_zm
    e_noise = est_zm - s_target
    si_snr = 10 * torch.log10((s_target ** 2).sum(dim=1) / ((e_noise ** 2).sum(dim=1) + eps) + eps)
    return si_snr


# You may want SI-SDR, but SI-SNR is often enough for deep learning evaluation. For SI-SDR:
def batch_si_sdr(est, ref, mask, eps=1e-8):
    # For most speech tasks, SI-SDR = SI-SNR, unless you add scaling factors or reference normalization.
    return batch_si_snr(est, ref, mask, eps=eps)
