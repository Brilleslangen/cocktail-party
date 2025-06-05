import torch
import torch.nn as nn
from mamba_ssm import Mamba2
from src.models.separator.base_separator import BaseSeparator


class MambaSeparator(BaseSeparator):
    """
    Mamba-2 based separator using stateful state-space models.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            d_model: int,
            n_blocks: int,
            d_state: int,
            d_conv: int,
            expand: int,
            frames_per_output: int,
            streaming_mode: bool,
            context_size_ms: float,
            name: str,
            **kwargs
    ):
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=d_model,
            n_blocks=n_blocks,
            frames_per_output=frames_per_output,
            streaming_mode=streaming_mode,
            context_size_ms=context_size_ms,
            name=name,
            causal=True,
            stateful=True,  # Mamba is stateful
            **kwargs
        )

        # Hidden states for each block
        self.hidden_states = [None] * n_blocks

    def _build_block(self, block_idx: int) -> nn.Module:
        return MambaBlock(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand
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
                    # Mamba2 might return tuple of states
                    self.hidden_states[i] = tuple(h.detach() if h is not None else None
                                                  for h in self.hidden_states[i])

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


class MambaBlock(nn.Module):
    """
    A single stateful Mamba-2 block with LayerNorm and residual connection.
    """

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

    def forward(self, x: torch.Tensor, hidden_state=None):
        """
        Args:
            x: [B, d_model, T]
            hidden_state: Hidden state from previous chunk
        Returns:
            Output with same shape and updated hidden state
        """
        # Transpose for Mamba: [B, C, T] -> [B, T, C]
        x_t = x.transpose(1, 2)

        # Apply norm (PreNorm)
        x_norm = self.norm(x_t)

        # Apply Mamba with state handling
        if hidden_state is not None:
            # Mamba2 supports passing initial states
            x_out, new_hidden = self.mamba(x_norm, initial_states=hidden_state)
        else:
            # First chunk - no initial state
            x_out = self.mamba(x_norm)
            # Extract final states for next chunk
            # Note: This assumes Mamba2 exposes a way to get final states
            # You may need to adapt based on actual Mamba2 API
            new_hidden = getattr(self.mamba, 'final_states', None)

        # Transpose back: [B, T, C] -> [B, C, T]
        return x_out.transpose(1, 2), new_hidden