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
            context_size_ms: float,
            name,
            **kwargs):
        super().__init__()
        assert output_dim % 2 == 0, "output_dim must be 2*D"
        self.D = output_dim // 2

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.context_size_ms = context_size_ms
        self.name = name

        num_neurons = max(num_neurons, output_dim + 3)
        print('Num neurons:', num_neurons)
        wiring = AutoNCP(num_neurons, output_dim)
        self.cfc = CfC(input_dim, wiring, batch_first=True)
        self.hidden_state = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] where C = input_dim
        seq = x.permute(0, 2, 1)  # → [B, T, C]
        h, self.hidden_state = self.cfc(seq, hx=self.hidden_state)
        h = h.permute(0, 2, 1)     # → [B, C, T]

        return h

    def reset_state(self):
        """
        Reset the separator state if it has one.
        """
        self.hidden_state = None

    def detach_state(self):
        """
        Detach the separator state if it has one.
        """
        print('\ndetaching state l\n')
        if self.hidden_state is not None:
            # Detach hidden state to truncate BPTT
            if isinstance(self.hidden_state, torch.Tensor):
                self.hidden_state = self.hidden_state.detach()
                print('Single')
            else:
                self.hidden_state = tuple(h.detach() for h in self.hidden_state)

    def get_output_dim(self) -> int:
        return self.output_dim

    def get_input_dim(self) -> int:
        return self.input_dim
