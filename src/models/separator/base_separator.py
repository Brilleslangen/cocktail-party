import torch
import torch.nn as nn
from abc import abstractmethod
from src.models.submodules import SubModule


class BaseSeparator(SubModule):
    """
    Abstract base class for all separator models following the common architecture:
    1. Input normalization (LayerNorm/GroupNorm)
    2. Input projection to d_model
    3. N stacked separator blocks with residual connections
    4. Output projection to 2D masks

    For streaming mode:
    - Stateless models: Process context window, output only the last chunk
    - Stateful models: Process chunk, output same-sized chunk
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            d_model: int,
            n_blocks: int,
            frames_per_output: int,
            streaming_mode: bool,
            context_size_ms: float,
            name: str,
            causal: bool = True,
            stateful: bool = False,
            **kwargs
    ):
        super().__init__()
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.streaming_mode = streaming_mode
        self.context_size_ms = context_size_ms
        self.causal = causal
        self.stateful = stateful
        self.frames_per_output = frames_per_output

        # Input normalization
        if causal:
            self.input_norm = CausalLayerNorm(input_dim)
        else:
            self.input_norm = nn.GroupNorm(1, input_dim, eps=1e-8)

        # Input projection
        self.input_proj = nn.Conv1d(input_dim, d_model, kernel_size=1)

        # Stacked separator blocks
        self.blocks = nn.ModuleList([
            self._build_block(block_idx=i) for i in range(n_blocks)
        ])

        # Output projection
        self.output_proj = nn.Conv1d(d_model, output_dim, kernel_size=1)

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

        Note: Base separator maintains [B, C, T] format throughout.
              Individual models handle transposition internally if needed.
        """
        # Input normalization and projection
        x = self.input_norm(x)
        x = self.input_proj(x)  # [B, d_model, T]

        # Pass through stacked blocks with residual connections
        # Each block handles its own format internally
        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual  # Residual connection

        # Output projection
        x = self.output_proj(x)  # [B, output_dim, T]

        # Handle streaming output sizing
        if self.streaming_mode and not self.stateful:
            # Stateless models: slice the last chunk from the context window
            # This preserves exact temporal alignment without averaging
            if self.frames_per_output < x.size(-1):
                x = x[..., -self.frames_per_output:]
        # Note: Stateful models output the full chunk as-is (no slicing needed)

        return x

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return self.output_dim


class CausalLayerNorm(nn.Module):
    """
    Causal LayerNorm that only uses past and present statistics.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, normalized_shape, 1))
        self.bias = nn.Parameter(torch.zeros(1, normalized_shape, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T]
        Returns:
            Normalized x with same shape
        """
        B, C, T = x.shape

        # Compute cumulative statistics
        cumsum = torch.cumsum(x, dim=-1)
        cumsum_sq = torch.cumsum(x.pow(2), dim=-1)

        # Count of elements up to each position
        count = torch.arange(1, T + 1, device=x.device, dtype=x.dtype)
        count = count.view(1, 1, T).expand(B, 1, T)

        # Compute running mean and variance
        mean = cumsum / count
        var = (cumsum_sq / count) - mean.pow(2)

        # Normalize
        x_norm = (x - mean) / (var + self.eps).sqrt()

        return x_norm * self.weight + self.bias