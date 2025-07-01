import torch
import torch.nn as nn
from abc import abstractmethod
from src.models.submodules import SubModule


class BaseSeparator(SubModule):
    """
    Abstract base class for separator models following the common architecture:
    1. Input normalization (LayerNorm/GroupNorm)
    2. Input projection to d_model
    3. N stacked separator blocks with residual connections
    4. Output projection to 2D masks

    5. For streaming mode:
    - Stateless models: Process context window, output only the last chunk
    - Stateful models: Process chunk, output same-sized chunk
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            d_model: int,
            n_blocks: int,
            d_ff: int,
            dropout_val: float,
            frames_per_output: int,
            frames_per_context: int,
            pos_enc: bool,
            streaming_mode: bool,
            context_size_ms: float,
            name: str,
            causal: bool = True,
            **kwargs
    ):
        super().__init__()
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout_val
        self.n_blocks = n_blocks
        self.streaming_mode = streaming_mode
        self.context_size_ms = context_size_ms
        self.causal = causal
        self.frames_per_output = frames_per_output
        self.frames_per_context = frames_per_context
        self.hidden_states = [None] * n_blocks

        # Input normalization
        self.input_norm = CausalLayerNorm(input_dim, channel_last=True) if causal else nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, d_model)

        self.positional_encoding = PositionalEncoding(d_model, max_len=self.frames_per_context) if pos_enc else None
        print('positional encoding:', pos_enc)

        print('input dim:', input_dim, 'd_model:', d_model,)

        # Stacked separator blocks
        self.blocks = nn.ModuleList([
            self._build_block(block_idx=i) for i in range(n_blocks)
        ])
        self.stateful = self.blocks[0].stateful if n_blocks > 0 else False

        # Output projection
        self.output_proj = nn.Sequential(nn.PReLU(), nn.Linear(d_model, output_dim))

    @abstractmethod
    def _build_block(self, block_idx: int) -> nn.Module:
        """Build a single separator block. Must be implemented by subclasses."""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input of shape [B, input_dim, T]
                       For stateless models in streaming: T = context_frames
                       For stateful models in streaming: T = chunk_frames
        Returns:
            Tensor: Output masks of shape [B, output_dim, T_out]
                   For stateless models in streaming: T_out = chunk_frames (last portion)
                   For stateful models in streaming: T_out = chunk_frames (same as input)
                   For offline mode: T_out = T
        """
        # Input normalization and projection
        x = x.transpose(1, 2)  # [B, T, input_dim]

        x = self.input_norm(x)
        x = self.input_proj(x)

        if self.positional_encoding:
            x = self.positional_encoding(x)

        # Pass through stacked blocks with residual connections
        # Each block handles its own format and residual mechanic internally
        if self.stateful:
            for i, block in enumerate(self.blocks):
                x, self.hidden_states[i] = block(x, self.hidden_states[i])
        else:
            for block in self.blocks:
                x, _ = block(x)

        # Output projection
        x = self.output_proj(x)  # [B, T, output_dim]

        # Handle streaming output sizing
        if self.streaming_mode and self.stateful:
            assert x.size(1) == self.frames_per_output, \
                f"Stateful separator must output same-sized chunk. Got {x.size(-1)}, expected {self.frames_per_output}."

        if self.streaming_mode and self.frames_per_output < x.size(-1):
            x = x[:, -self.frames_per_output:, :]

        return x.transpose(1, 2)  # [B, output_dim, T_out]

    def reset_state(self, batch_size: int, chunk_len: int):
        """Reset all hidden states."""
        if hasattr(self.blocks[0], 'build_fresh_state'):
            self.hidden_states = [self.blocks[i].build_fresh_state(batch_size, chunk_len, i) for i in range(self.n_blocks)]
        self.hidden_states = [None] * self.n_blocks

    def detach_state(self):
        """Detach hidden states to truncate BPTT."""
        for i in range(self.n_blocks):
            state = self.hidden_states[i]
            if state is not None:
                if isinstance(state, torch.Tensor):
                    self.hidden_states[i] = state.detach()
                else:
                    old_tuple = state.key_value_memory_dict[i]
                    state.key_value_memory_dict[i] = tuple(h.detach() if h is not None else None for h in old_tuple)

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return self.output_dim


def build_FFN(d_model: int, d_ff: int, dropout: float) -> nn.Sequential:
    """
    Build a feedforward block with GELU activation and dropout.
    Typically used in Transformer-like architectures.
    """
    return nn.Sequential(
        nn.Linear(d_model, d_ff),  # Typically d_ff = 4 * d_model
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_ff, d_model),
        nn.Dropout(dropout)
    )


class ResidualBlock(nn.Module):
    """
    Abstract base class for residual blocks used in the separator model.
    """
    def __init__(self, d_model: int, d_ff: int, dropout_val: float, causal: bool, post_core_gelu: bool, stateful: bool):
        super().__init__()
        self.d_model = d_model
        self.causal = causal
        self.stateful = stateful
        self.dropout_val = dropout_val

        # Choose LayerNorm based on causality
        Norm = CausalLayerNorm if self.causal else nn.LayerNorm

        self.norm1 = Norm(d_model)
        self.core = self._build_core_layer()  # Multi-Head Attention, S4, Mamba or Liquid
        self.post_core_gelu = nn.GELU() if post_core_gelu else None
        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout1 = nn.Dropout(dropout_val)

        self.norm2 = Norm(d_model)
        self.ffn = build_FFN(d_model, d_ff, dropout_val)

    @abstractmethod
    def _build_core_layer(self) -> nn.Module:
        """
        Abstract method to build the core layer of the block.
        Should be implemented by subclasses to return the specific core layer.
        """
        raise NotImplementedError("Subclasses muswt implement _build_core_layer method.")

    def forward(self, x, hidden_state=None):
        # Core layer (+ gelu) + Linear + Dropout + Residual
        residual = x
        x = self.norm1(x)

        if self.stateful:
            x, hidden_state = self.core(x, hidden_state)
        else:
            x = self.core(x)

        x = self.post_core_gelu(x) if self.post_core_gelu else x
        x = self.linear1(x)
        x = self.dropout1(x)
        x = x + residual

        # FFN + Residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual

        return x, hidden_state


class PositionalEncoding(nn.Module):
    """
    Positional Encoding module to add positional information to the input. (only for Transformer architectures)
    """
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [B, T, d_model]
        x = x + self.pe[:, :x.size(1)]
        return x


class CausalLayerNorm(nn.Module):
    """
    Causal Layer Normalization for streaming models.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-5, channel_last: bool = True):
        super().__init__()
        self.channel_last = channel_last
        self.eps = eps

        if channel_last:
            self.weight = nn.Parameter(torch.ones(1, 1, normalized_shape))
            self.bias = nn.Parameter(torch.zeros(1, 1, normalized_shape))
        else:
            self.weight = nn.Parameter(torch.ones(1, normalized_shape, 1))
            self.bias = nn.Parameter(torch.zeros(1, normalized_shape, 1))

    def forward(self, x):
        if self.channel_last:
            B, T, C = x.shape
            cumsum = torch.cumsum(x, dim=1)
            cumsum_sq = torch.cumsum(x.pow(2), dim=1)
            count = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, T, 1).expand(B, T, C)
        else:
            B, C, T = x.shape
            cumsum = torch.cumsum(x, dim=2)
            cumsum_sq = torch.cumsum(x.pow(2), dim=2)
            count = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, 1, T).expand(B, C, T)

        mean = cumsum / count
        var = cumsum_sq / count - mean.pow(2)
        var = torch.clamp(var, min=0.0)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        return x_norm * self.weight + self.bias

