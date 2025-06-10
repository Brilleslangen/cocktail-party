import torch
from torch import nn


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


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_macs(model: nn.Module, seconds: float = 1.0) -> int:
    """Return MACs needed to process ``seconds`` of audio.

    For streaming models we profile a single context window of length
    ``model.input_size`` and multiply by the number of windows required to
    cover ``seconds`` seconds. This allows fair comparison with offline mode
    where the full signal is processed in one pass.
    """
    from thop import profile

    device = next(model.parameters()).device
    sample_rate = getattr(model, "sample_rate", 16000)
    frames = int(sample_rate * seconds)

    streaming = getattr(model, "streaming_mode", False)
    if streaming:
        window = getattr(model, "input_size", frames)
        chunk = getattr(model, "output_size", frames)
        dummy = torch.randn(1, 2, window, device=device)
        macs_per_window, _ = profile(model, inputs=(dummy,), verbose=False)
        windows_per_second = sample_rate / chunk
        macs = macs_per_window * windows_per_second
    else:
        dummy = torch.randn(1, 2, frames, device=device)
        macs, _ = profile(model, inputs=(dummy,), verbose=False)

    return int(macs)


def prettify_macs(macs: float) -> str:
    """Convert MAC count per second to a human readable string."""
    if macs < 1e6:
        return f"{macs / 1e3:.2f}K MAC/s"
    elif macs < 1e9:
        return f"{macs / 1e6:.2f}M MAC/s"
    elif macs < 1e12:
        return f"{macs / 1e9:.2f}G MAC/s"
    else:
        return f"{macs / 1e12:.2f}T MAC/s"


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
