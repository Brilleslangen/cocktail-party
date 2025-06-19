from src.models.separator.stateful.liquid_separator import LiquidSeparator
from src.models.separator.stateless.transformer_separator import TransformerSeparator
from src.models.separator.stateless.tcn_separator import TCNSeparator
from src.models.separator.stateless.no_separator import NoSeparator

try:
    from src.models.separator.stateful.mamba_separator import MambaSeparator
except ImportError:
    # If mamba_ssm is not installed, we can still use TCNSeparator
    YELLOW = "\033[93m"  # bright yellow
    RESET = "\033[0m"
    print(
        f"{YELLOW}Warning: MambaSeparator is not available. "
        "Please install mamba_ssm if you wish to use it. "
        f"Only available with NVIDIA GPUs.{RESET}\n"
    )

    pass

__all__ = ["TCNSeparator", "MambaSeparator", "LiquidSeparator", "TransformerSeparator", "NoSeparator"]
