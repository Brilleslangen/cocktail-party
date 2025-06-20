import torch
from torch import nn


class Oracle(nn.Module):
    """Baseline model that simply returns its input."""

    def __init__(self, sample_rate: int, use_targets_as_input: bool, streaming_mode: bool, **kwargs):
        super().__init__()
        self.identity = True
        self.sample_rate = sample_rate
        self.use_targets_as_input = use_targets_as_input
        self.streaming_mode = streaming_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
