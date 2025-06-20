import torch
from torch import nn

from src.helpers import ms_to_samples, select_device


class Oracle(nn.Module):
    """Baseline model that simply returns its input."""

    def __init__(self, sample_rate: int, use_targets_as_input: bool, streaming_mode: bool,
                 stream_context_size_ms: float, stream_chunk_size_ms, **kwargs):
        super().__init__()
        self.identity = True
        self.sample_rate = sample_rate
        self.use_targets_as_input = use_targets_as_input
        self.streaming_mode = streaming_mode
        self.input_size = ms_to_samples(stream_context_size_ms, sample_rate)
        self.output_size = ms_to_samples(stream_chunk_size_ms, sample_rate)  # Size of the output chunk in samples
        self.device = select_device()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
