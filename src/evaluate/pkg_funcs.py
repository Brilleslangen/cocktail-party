import torch
from joblib import Parallel, delayed


def compute_mask(lengths, T, device):
    """
    lengths: [B] (on any device)
    Returns mask of shape [B, T] (on target device)
    """
    return (torch.arange(T, device=device)[None, :] < lengths.to(device)[:, None]).float()


def compute_energy_weights(
        ref,  # [B, C, T], or list of [C, L_b], or [C, L]
        mask=None,
        eps=1e-8,
):
    """
    Compute energy-based weights for multi-channel signals.

    Args:
        ref:   [B, C, T] batch, or list of [C, L_b] tensors, or [C, L]
        mask:  [B, T], [L], or None
        eps:   float

    Returns:
        weights: [B, C] if batch/list, [C] if single sample
    """
    if isinstance(ref, list):  # Ragged batch: list of [C, L_b]
        weights = []
        for r in ref:
            channel_energy = (r ** 2).sum(dim=1)  # [C]
            total_energy = channel_energy.sum() + eps
            weights.append(channel_energy / total_energy)  # [C]
        return torch.stack(weights, dim=0)  # [B, C]
    elif ref.ndim == 3:  # [B, C, T]
        if mask is not None:
            channel_energy = ((ref * mask.unsqueeze(1)) ** 2).sum(dim=2)  # [B, C]
        else:
            channel_energy = (ref ** 2).sum(dim=2)  # [B, C]
        total_energy = channel_energy.sum(dim=1, keepdim=True) + eps  # [B, 1]
        return channel_energy / total_energy  # [B, C]
    elif ref.ndim == 2:  # [C, L]
        if mask is not None:
            channel_energy = ((ref * mask.unsqueeze(0)) ** 2).sum(dim=1)  # [C]
        else:
            channel_energy = (ref ** 2).sum(dim=1)  # [C]
        total_energy = channel_energy.sum() + eps
        return channel_energy / total_energy  # [C]
    else:
        raise ValueError(
            f"ref must be [C, L], [B, C, T], or list of [C, L_b]. Got {type(ref)}, shape {getattr(ref, 'shape', None)}."
        )


def parallel_batch_metric_with_lengths(
        metric_func,  # Callable: takes (ref_trimmed, est_trimmed)
        est: torch.Tensor,  # [B, C, T]
        ref: torch.Tensor,  # [B, C, T]
        lengths: torch.Tensor,  # [B]
        n_jobs: int = -1,  # 0=serial, -1=all cores, >0=explicit
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
