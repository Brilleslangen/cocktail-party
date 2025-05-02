import torch
from torch import nn

from src.helpers import ms_to_samples
from src.models.submodules import SubModule


class TasNetEncoder(SubModule):
    """
    Time-domain analysis filterbank (Conv-TasNet encoder).

    Performs:
      1) 1-D convolution to project raw waveform into N-dimensional feature space
      2) Non-negative activation via ReLU to enforce positivity
    """
    def __init__(self, num_filters: int, filter_length_ms: int, stride_ms: int, sample_rate: int, causal: bool):
        """
        Initialize the Conv1d encoder.

        Args:
            num_filters (int): Number of learned basis filters N.
            filter_length (int): Length of each filter (in samples).
            stride (int): Hop size between filters (in samples).
            causal (bool): If True, use causal padding (no future context).
        """
        # Initialize base classes
        super().__init__()

        # Input is always single-channel waveform
        self.in_channels = 1
        self.num_filters = num_filters
        self.filter_length = ms_to_samples(filter_length_ms, sample_rate)  # filter length in samples
        self.stride = ms_to_samples(stride_ms, sample_rate)  # stride in samples

        # Determine padding: causal vs non-causal
        pad = self.filter_length - self.stride if causal else (self.filter_length // 2)

        # 1-D convolution: [B, 1, T] → [B, N, T_frames]
        self.conv1d = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.num_filters,
            kernel_size=self.filter_length,
            stride=self.stride,
            padding=pad,
            bias=False
        )

        # Activation to enforce non-negativity
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.

        Args:
            x (torch.Tensor): Input waveform of shape [B, T] or [B, 1, T].

        Returns:
            torch.Tensor: Encoded features [B, N, T_frames], where
                T_frames = floor((T + 2*pad - (filter_length-1) -1)/stride) + 1
        """
        # Ensure shape is [B, 1, T]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Convolution + ReLU
        out = self.conv1d(x)  # [B, N, T_frames]
        return self.relu(out)

    def get_input_dim(self) -> int:
        """Return the number of input channels (always 1)."""
        return self.in_channels

    def get_output_dim(self) -> int:
        """Return the number of output channels (num_filters)."""
        return self.num_filters
