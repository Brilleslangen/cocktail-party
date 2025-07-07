import torch
from torch import nn
from src.helpers import ms_to_samples
from src.models.submodules import SubModule


class TasNetEncoder(SubModule):
    """
    Time-domain analysis filterbank (Conv-TasNet encoder).
    """

    def __init__(self, num_filters: int, filter_length_ms: int, stride_ms: int, sample_rate: int, causal: bool,
                 relu: bool = False):
        super().__init__()
        self.in_channels = 1
        self.num_filters = num_filters
        self.filter_length = ms_to_samples(filter_length_ms, sample_rate)
        self.stride = ms_to_samples(stride_ms, sample_rate)

        self.pad = self.filter_length - self.stride if causal else (self.filter_length // 2)

        self.conv1d = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.num_filters,
            kernel_size=self.filter_length,
            stride=self.stride,
            padding=self.pad,
            bias=False
        )

        self.relu = nn.ReLU() if relu else None  # Paper "Ultra-Low Latency Speech Enhancement - A Comprehensive Study"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder with diagnostics.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, T]

        # Input checks
        # print("[Encoder] Input NaNs:", torch.isnan(x).any().item(), "| Max:", x.max().item(), "| Min:", x.min().item())

        # Conv1d
        out = self.conv1d(x)
        # print("[Encoder] Conv1d Output NaNs:", torch.isnan(out).any().item(), "| Max:", out.max().item(), "| Min:", out.min().item())

        # ReLU
        if self.relu is not None:
            out = self.relu(out)

        return out

    def get_input_dim(self) -> int:
        return self.in_channels

    def get_output_dim(self) -> int:
        return self.num_filters
