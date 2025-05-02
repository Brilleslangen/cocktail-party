import torch


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def ms_to_samples(ms, sample_rate=16000) -> int:
    """
    Convert milliseconds to samples.

    Args:
        ms (float): Time in milliseconds.
        sample_rate (int): Sample rate in Hz. Default is 16000.

    Returns:
        int: Time in samples.
    """
    return int(ms * sample_rate / 1000)
