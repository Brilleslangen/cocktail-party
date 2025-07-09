import torch
import platform
from omegaconf import OmegaConf, DictConfig

OmegaConf.register_new_resolver("mul", lambda x, y: int(x * y))


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def setup_device_optimizations(cfg: DictConfig):
    """Configure device-specific optimizations."""
    device = select_device()
    print(device)
    if cfg.training.params.cpu:
        device = torch.device("cpu")
        print("Using CPU for training.")
    device_type = device.type

    # Device-specific settings
    if device_type == "cuda":
        # Enable TF32 on Ampere GPUs for better performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable cudnn benchmarking for consistent input sizes
        torch.backends.cudnn.benchmark = True

        # Get GPU info for logging
        gpu_name = torch.cuda.get_device_name()
        gpu_capability = torch.cuda.get_device_capability()

        print(f"🚀 CUDA device: {gpu_name}")
        print(f"   Compute capability: {gpu_capability[0]}.{gpu_capability[1]}")

        # Check FlashAttention availability
        if gpu_capability[0] >= 8:
            print("   ✓ FlashAttention-2 compatible GPU detected")

        # Mixed precision settings
        use_amp = True
        amp_dtype = torch.float16  # Use bfloat16 if available on Ampere+
        if gpu_capability[0] >= 8:
            amp_dtype = torch.bfloat16
            print("   ✓ Using bfloat16 for mixed precision")
        else:
            print("   ✓ Using float16 for mixed precision")

    elif device_type == "mps":
        # MPS (Apple Silicon) settings
        print(f"🍎 MPS device: {platform.processor()}. Mixed precision disabled.")
        use_amp = False
        amp_dtype = torch.float32

    else:
        # CPU fallback
        print("💻 CPU. Mixed precision disabled")
        use_amp = False
        amp_dtype = torch.float32

    return device, use_amp, amp_dtype


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
