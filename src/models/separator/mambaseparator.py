import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from src.models.submodules import SubModule


class MambaSeparator(SubModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        context_size_ms: float,
        name: str
    ):
        super().__init__()
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_size_ms = context_size_ms

        self.input_proj = nn.Conv1d(input_dim, d_model, kernel_size=1)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.output_proj = nn.Conv1d(d_model, output_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)  # [B, input_dim, T] → [B, d_model, T]
        x = x.transpose(1, 2)   # [B, T, d_model]
        x = self.mamba(x)       # [B, T, d_model]
        x = x.transpose(1, 2)   # [B, d_model, T]
        return self.output_proj(x)  # [B, output_dim, T]

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return self.output_dim
