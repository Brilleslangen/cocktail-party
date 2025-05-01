import torch
from torch import nn
from src.models.submodules import SubModule


class TasNetDecoder(SubModule):
    """
    Inverse filterbank via transposed convolution:
      1) 1-D ConvTranspose (N→1 filters)
      2) Overlap-add reconstruction
    """

    def __init__(self, num_filters: int, filter_length: int, stride: int, causal: bool):
        super().__init__()
        self.num_filters = num_filters
        self.out_channels = 1
        padding = filter_length - stride if causal else (filter_length // 2)
        self.deconv1d = nn.ConvTranspose1d(self.num_filters, self.out_channels, kernel_size=filter_length,
                                           stride=stride, padding=padding, bias=False)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """
        w: [B, N, T_frames]
        returns: [B, 1, T]  or squeeze to [B, T]
        """
        x_hat = self.deconv1d(w)  # [B,1,T]
        return x_hat.squeeze(1)  # [B, T]

    def get_input_dim(self) -> int:
        return self.num_filters

    def get_output_dim(self) -> int:
        return self.out_channels
