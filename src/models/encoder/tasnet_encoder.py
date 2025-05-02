import torch
from torch import nn
from src.models.submodules import SubModule


class TasNetEncoder(SubModule):
    """
    Time-domain analysis filterbank as in Conv-TasNet:
      1) 1-D Conv (1→N filters)
      2) Non-negative activation (ReLU)
    See Luo & Mesgarani
    """

    def __init__(self, num_filters: int, filter_length: int, stride: int, causal: bool):
        super().__init__()
        self.in_channels = 1
        self.num_filters = num_filters
        padding = filter_length - stride if causal else (filter_length // 2)
        self.conv1d = nn.Conv1d(1, num_filters, kernel_size=filter_length,
                                stride=stride, padding=padding, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,  T] or [B, 1, T]
        returns: [B, N, T_frames]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B,1,T]
        w = self.conv1d(x)  # [B,N,T']
        w = self.relu(w)  # enforce non-negativity
        return w

    def get_input_dim(self) -> int:
        return self.in_channels

    def get_output_dim(self) -> int:
        return self.num_filters
