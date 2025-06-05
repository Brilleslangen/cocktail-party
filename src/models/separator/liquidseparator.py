import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP
from src.models.separator.base_separator import BaseSeparator


class LiquidSeparator(BaseSeparator):
    """
    Liquid Neural Network (CfC) based separator with stateful processing.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            d_model: int,
            n_blocks: int,
            num_neurons: int,
            frames_per_output: int,
            streaming_mode: bool,
            context_size_ms: float,
            causal_proj: bool,
            name: str,
            **kwargs
    ):
        self.num_neurons = num_neurons

        # Ensure we have enough neurons
        self.num_neurons = max(num_neurons, d_model + 3)

        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=d_model,
            n_blocks=n_blocks,
            frames_per_output=frames_per_output,
            streaming_mode=streaming_mode,
            context_size_ms=context_size_ms,
            name=name,
            causal=causal_proj,
            stateful=True,  # Liquid is stateful
            **kwargs
        )

        # Hidden states for each block
        self.hidden_states = [None] * n_blocks

    def _build_block(self, block_idx: int) -> nn.Module:
        return LiquidBlock(
            d_model=self.d_model,
            num_neurons=self.num_neurons
        )

    def reset_state(self):
        """Reset all hidden states."""
        self.hidden_states = [None] * self.n_blocks

    def detach_state(self):
        """Detach hidden states to truncate BPTT."""
        for i in range(self.n_blocks):
            if self.hidden_states[i] is not None:
                if isinstance(self.hidden_states[i], torch.Tensor):
                    self.hidden_states[i] = self.hidden_states[i].detach()
                else:
                    self.hidden_states[i] = tuple(h.detach() for h in self.hidden_states[i])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Override to handle stateful processing."""
        # Input normalization and projection
        x = self.input_norm(x)
        x = self.input_proj(x)  # [B, d_model, T]

        # Pass through stacked blocks with residual connections
        for i, block in enumerate(self.blocks):
            residual = x
            x, self.hidden_states[i] = block(x, self.hidden_states[i])
            x = x + residual  # Residual connection

        # Apply streaming pool if needed
        if self.streaming_mode and self.stream_pool is not None:
            x = self.stream_pool(x)

        # Output projection
        return self.output_proj(x)


class LiquidBlock(nn.Module):
    """
    A single Liquid Neural Network block with CfC neurons.
    """

    def __init__(self, d_model: int, num_neurons: int):
        super().__init__()

        self.d_model = d_model

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

        # Create wiring for liquid network
        wiring = AutoNCP(num_neurons, d_model)
        self.cfc = CfC(d_model, wiring, batch_first=True)

    def forward(self, x: torch.Tensor, hidden_state=None):
        """
        Args:
            x: [B, d_model, T]
            hidden_state: Hidden state from previous chunk
        Returns:
            Output with same shape and updated hidden state
        """
        # Transpose for CfC: [B, C, T] -> [B, T, C]
        x_t = x.transpose(1, 2)

        # Apply layer norm
        x_norm = self.norm(x_t)

        # Apply CfC
        x_out, new_hidden = self.cfc(x_norm, hx=hidden_state)

        # Transpose back: [B, T, C] -> [B, C, T]
        return x_out.transpose(1, 2), new_hidden
