import torch
from torch import nn

from src.helpers import ms_to_samples
from src.models.submodules import SubModule


class TasNetDecoder(SubModule):
    """
    Conv-TasNet decoder: inverse filterbank via transposed convolution.

    Takes masked encoder outputs and reconstructs time-domain waveform.

    Steps:
      1) 1-D ConvTranspose1d from N channels back to 1 channel
      2) Overlap-add automatically handled by stride and padding
    """
    def __init__(self, num_filters: int, filter_length_ms: int, stride_ms: int, sample_rate: int, causal: bool):
        """
        Initialize the ConvTranspose1d layer for waveform reconstruction.

        Args:
            num_filters (int): Number of basis filters N (input channels).
            filter_length_ms (int): Length of each learned filter (kernel size).
            stride (int): Hop size (must match encoder stride).
            causal (bool): If True, use causal padding (no future overlap).
        """
        # record dimensions for config introspection
        super().__init__()
        self.num_filters = num_filters
        self.out_channels = 1
        self.stride = ms_to_samples(stride_ms, sample_rate)  # stride in samples
        self.filter_length = ms_to_samples(filter_length_ms, sample_rate)  # filter length in samples

        # padding ensures perfect overlap-add
        # causal: pad = L-stride; non-causal: symmetric padding = L//2
        self.pad = self.filter_length - self.stride if causal else (self.filter_length // 2)
        self.deconv1d = nn.ConvTranspose1d(
            in_channels=self.num_filters,
            out_channels=self.out_channels,
            kernel_size=self.filter_length,
            stride=self.stride,
            padding=self.pad,
            bias=False
        )

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """
        Decode masked features back to time-domain waveform.

        Args:
            w (torch.Tensor): Encoded & masked features of shape [B, N, T_frames].

        Returns:
            torch.Tensor: Reconstructed waveform [B, 1, T_samples].
        """
        return self.deconv1d(w)

    def get_input_dim(self) -> int:
        """Return the number of input channels (num_filters)."""
        return self.num_filters

    def get_output_dim(self) -> int:
        """Return the number of output channels (should be 1)."""
        return self.out_channels
