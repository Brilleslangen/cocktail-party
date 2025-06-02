import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP


class LiquidSeparator(nn.Module):
    """
    Stacked Liquid-Time-Constant (CfC) separator for TasNet-style models.
    Suitable for fair comparison with Transformer and Mamba separators.

    Args:
        input_dim (int): Feature dimension input (from encoder).
        output_dim (int): Feature dimension output (to decoder).
        num_neurons (int): Neurons per CfC block.
        num_layers (int): Number of stacked CfC blocks.
        context_size_ms (float): For documentation/receptive field info.
        name (str): Module name/identifier.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            num_neurons: int,
            num_layers: int,
            context_size_ms: float = None,
            name: str = "",
            **kwargs,
    ):
        super().__init__()
        assert output_dim % 2 == 0, "output_dim must be 2*D"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_size_ms = context_size_ms
        self.name = name
        self.n_layers = num_layers

        num_neurons = max(num_neurons, output_dim + 3)
        self.cfc_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else output_dim
            self.cfc_layers.append(CfC(in_dim, AutoNCP(num_neurons, output_dim), batch_first=True))

        self.norms = nn.ModuleList([
            nn.LayerNorm(output_dim)
            for _ in range(num_layers)
        ])
        # Store hidden state for each layer (reset per utterance/sequence)
        self.hidden_states = [None for _ in range(num_layers)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): [B, C, T], channel-first (encoder output)
        Returns:
            Tensor: [B, C, T], channel-first (for mask/decoder)
        """
        x = x.permute(0, 2, 1)  # [B, T, C] for CfC
        out = x
        for i, (cfc, norm) in enumerate(zip(self.cfc_layers, self.norms)):
            prev = out
            out, self.hidden_states[i] = cfc(out, hx=self.hidden_states[i])
            out = norm(out)
            if prev.shape == out.shape:
                out = out + prev  # Residual connection
        out = out.permute(0, 2, 1)  # [B, C, T]
        return out

    def reset_state(self):
        """
        Reset hidden state (call at start of each new utterance/batch).
        """
        self.hidden_states = [None for _ in range(self.n_layers)]

    def get_output_dim(self) -> int:
        return self.output_dim

    def get_input_dim(self) -> int:
        return self.input_dim
