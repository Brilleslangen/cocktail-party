import torch
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("mul", lambda x, y: int(x * y))


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def using_cuda():
    """
    Automatically pin memory if CUDA is available.
    Returns True if pinning is enabled, False otherwise.
    """
    return torch.cuda.is_available() and torch.backends.cudnn.is_available()


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


def prettify_macs(macs: float) -> str:
    """Convert MAC count per second to a human-readable string."""
    if macs < 1e6:
        return f"{macs / 1e3:.2f}K"
    elif macs < 1e9:
        return f"{macs / 1e6:.2f}M"
    elif macs < 1e12:
        return f"{macs / 1e9:.2f}G"
    else:
        return f"{macs / 1e12:.2f}T"


def prettify_param_count(param_count: int) -> str:
    """
    Convert parameter count to a human-readable format.

    Args:
        param_count (int): Number of parameters.

    Returns:
        str: Human-readable parameter count.
    """
    if param_count < 1e3:
        return f"{param_count} parameters"
    elif param_count < 1e6:
        return f"{param_count / 1e3:.2f}K"
    elif param_count < 1e9:
        return f"{param_count / 1e6:.2f}M"
    else:
        return f"{param_count / 1e9:.2f}B"


def format_time(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes, seconds = divmod(seconds, 60)
        return f"{minutes}m {seconds}s"
    else:
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours}h {minutes}m {seconds}s"
