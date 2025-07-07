import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from src.models.submodules import SubModule


class MambaSeparator(SubModule):
    """
    Mamba-based separator for binaural feature streams.

    Args:
        input_dim (int): Number of input channels.
        output_dim (int): Number of output channels (e.g., 2 x D masks).
        d_model (int): Internal model dimension of Mamba2.
        d_state (int): State dimension of Mamba2.
        d_conv (int): Convolutional kernel size for Mamba2.
        expand (int): Expansion factor inside Mamba2.
        n_layers (int): Number of stacked Mamba2 blocks.
        frames_per_output (int): Target number of output frames (used in streaming mode).
        streaming_mode (bool): Whether to apply pooling for online processing.
        context_size_ms (float): Context used for documentation or receptive field estimation.
        name (str): Module identifier.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        n_layers: int,
        frames_per_output: int,
        streaming_mode: bool,
        context_size_ms: float,
        name: str,
    ):
        super().__init__()
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_size_ms = context_size_ms
        self.streaming_mode = streaming_mode

        self.input_proj = nn.Conv1d(input_dim, d_model, kernel_size=1)

        self.mamba_layers = nn.Sequential(
            *[
                Mamba2(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand
                )
                for _ in range(n_layers)
            ]
        )

        self.stream_pool = nn.AdaptiveAvgPool1d(frames_per_output)

        self.output_proj = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(d_model, output_dim, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input of shape [B, input_dim, T]
        Returns:
            Tensor: Output of shape [B, output_dim, T'] (may be reduced if streaming)
        """
        x = self.input_proj(x)      # [B, d_model, T]
        x = x.transpose(1, 2)       # [B, T, d_model]
        x = self.mamba_layers(x)    # [B, T, d_model]
        x = x.transpose(1, 2)       # [B, d_model, T]

        if self.streaming_mode:
            x = self.stream_pool(x)  # [B, d_model, frames_per_output]

        return self.output_proj(x)   # [B, output_dim, T']

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return self.output_dim