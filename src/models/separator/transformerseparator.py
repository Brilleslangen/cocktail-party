import torch
import torch.nn as nn
from src.models.separator.base_separator import BaseSeparator
from src.helpers import ms_to_samples


class TransformerSeparator(BaseSeparator):
    """
    Transformer-based separator using causal multi-head self-attention.
    Supports both local (windowed) attention and full attention modes.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            d_model: int,
            n_blocks: int,
            n_heads: int,
            d_ff: int,
            dropout: float = 0.1,
            causal: bool = True,
            local_attention: bool = True,
            attention_window_ms: float = None,
            stride_ms: float = 2.0,
            sample_rate: int = 16000,
            frames_per_output: int = 1,
            streaming_mode: bool = False,
            context_size_ms: float = 32.0,
            name: str = "transformer",
            **kwargs
    ):
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.local_attention = local_attention

        # Calculate attention window in frames if using local attention
        if local_attention and attention_window_ms is not None:
            # Convert ms to samples, then to frames
            window_samples = ms_to_samples(attention_window_ms, sample_rate)
            stride_samples = ms_to_samples(stride_ms, sample_rate)
            self.attention_window_frames = max(1, window_samples // stride_samples)
        else:
            self.attention_window_frames = None

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
            stateful=False,  # Transformer is stateless
            **kwargs
        )

    def _build_block(self, block_idx: int) -> nn.Module:
        return TransformerBlock(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout=self.dropout,
            causal=False,
            local_attention=self.local_attention,
            attention_window=self.attention_window_frames
        )


class TransformerBlock(nn.Module):
    """
    A single Transformer block with causal self-attention and FFN.
    Supports both local (windowed) attention and full attention.
    """

    def __init__(
            self,
            d_model: int,
            n_heads: int,
            d_ff: int,
            dropout: float,
            causal: bool,
            local_attention: bool = True,
            attention_window: int = None
    ):
        super().__init__()
        self.causal = causal
        self.local_attention = local_attention
        self.attention_window = attention_window

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def _create_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create attention mask based on causality and local attention settings.

        Args:
            seq_len: Sequence length
            device: Device to create mask on

        Returns:
            Attention mask of shape [seq_len, seq_len]
        """
        # Start with causal mask if needed
        if self.causal:
            # Upper triangular matrix (future positions masked)
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device) * float('-inf'),
                diagonal=1
            )
        else:
            mask = torch.zeros(seq_len, seq_len, device=device)

        # Apply local attention window if specified
        if self.local_attention and self.attention_window is not None:
            # Create local attention mask
            for i in range(seq_len):
                # Each position can only attend to positions within the window
                start = max(0, i - self.attention_window + 1)
                # Mask positions outside the window
                if start > 0:
                    mask[i, :start] = float('-inf')

        return mask

    def _get_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # Cache mask per (seq_len, device)
        if not hasattr(self, "_mask_cache"):
            self._mask_cache = {}
        key = (seq_len, device)
        if key not in self._mask_cache:
            self._mask_cache[key] = self._create_attention_mask(seq_len, device)
        return self._mask_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            Output with same shape
        """
        # Transpose for attention: [B, T, d_model] -> [B, T]
        B, T, C = x.shape

        # Create attention mask
        attn_mask = self._get_attention_mask(T, x.device)

        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # FFN with residual
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.norm2(x)

        return x  # [B, T, d_model]
