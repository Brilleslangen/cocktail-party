import torch
from torch import nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP

from src.models.submodules import SubModule


class LiquidSeparator(SubModule):
    """
    Liquid‐Time‐Constant (CfC) separator producing two normalized masks.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            num_neurons: int,
            name):
        super().__init__()
        assert output_dim % 2 == 0, "output_dim must be 2*D"
        self.D = output_dim // 2

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name

        num_neurons = max(num_neurons, output_dim + 3)
        print('Num neurons:', num_neurons)
        wiring = AutoNCP(num_neurons, output_dim)
        self.cfc = CfC(input_dim, wiring, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] where C = input_dim
        B, C, T = x.shape
        seq = x.permute(0, 2, 1)  # → [B, T, C]
        h, _ = self.cfc(seq)
        h = h.permute(0, 2, 1)     # → [B, C, T]

        return h

    def get_output_dim(self) -> int:
        return self.output_dim

    def get_input_dim(self) -> int:
        return self.input_dim
