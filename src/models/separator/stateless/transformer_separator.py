import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.separator.base_separator import BaseSeparator, build_FFN, ResidualBlock
from src.helpers import ms_to_samples


class TransformerSeparator(BaseSeparator):
    """
    Transformer-based separator using causal multi-head self-attention.
    Supports both local (windowed) attention and full attention modes.
    Uses FlashAttention-2 when available via PyTorch's scaled_dot_product_attention.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            d_model: int,
            n_blocks: int,
            n_heads: int,
            d_ff: int,
            dropout_val: float,
            local_attention: bool,
            stride_ms: float,
            sample_rate: int,
            frames_per_output: int,
            streaming_mode: bool,
            context_size_ms: float,
            causal_proj: bool,
            name: str = "transformer",
            **kwargs
    ):
        self.n_heads = n_heads
        self.local_attention = local_attention

        # Calculate attention window in frames if using local attention
        if local_attention and context_size_ms is not None:
            window_samples = ms_to_samples(context_size_ms, sample_rate)
            stride_samples = ms_to_samples(stride_ms, sample_rate)
            self.attention_window_frames = max(1, window_samples // stride_samples)
        else:
            self.attention_window_frames = None

        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=d_model,
            n_blocks=n_blocks,
            d_ff=d_ff,
            dropout_val=dropout_val,
            frames_per_output=frames_per_output,
            streaming_mode=streaming_mode,
            context_size_ms=context_size_ms,
            name=name,
            causal=causal_proj,
            **kwargs
        )

        # Check if FlashAttention is available
        _check_flash_attention()

    def _build_block(self, block_idx: int) -> nn.Module:
        return TransformerBlock(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout_val=self.dropout,
            causal=True,
            local_attention=self.local_attention,
            attention_window=self.attention_window_frames
        )


class TransformerBlock(ResidualBlock):
    """
    Transformer block with FlashAttention-2 support.
    Uses PreNorm architecture for better training stability.
    """

    def __init__(
            self,
            d_model: int,
            d_ff: int,
            dropout_val: float,
            causal: bool,
            n_heads: int,
            local_attention: bool = False,
            attention_window: int = None
    ):
        self.n_heads = n_heads
        self.local_attention = local_attention
        self.attention_window = attention_window
        self.head_dim = d_model // n_heads
        super().__init__(d_model, d_ff, dropout_val, causal, post_core_gelu=False, stateful=False)

    def _build_core_layer(self) -> nn.Module:
        return MultiHeadAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout=self.dropout_val,
            causal=self.causal,
            local_attention=self.local_attention,
        )


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer with optional casual (+local) attention.
    Uses FlashAttention-2 when available.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float, causal: bool, local_attention: bool = False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        self.local_attention = local_attention
        self.causal = causal

        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

    def _create_attention_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Create attention mask for local attention window."""
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)

        if self.local_attention and self.attention_window is not None:
            for i in range(seq_len):
                start = max(0, i - self.attention_window + 1)
                if start > 0:
                    mask[i, :start] = float('-inf')

        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv_proj(x)  # [B, T, 3*d_model]

        # Reshape QKV for multi-head attention
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, n_heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [B, n_heads, T, head_dim]

        # Prepare attention mask
        attn_mask = None
        if self.local_attention and self.attention_window is not None:
            # Local attention needs explicit mask
            attn_mask = self._create_attention_mask(T, x.device, x.dtype)
            is_causal = False  # Use explicit mask
        else:
            # For full causal attention, let SDPA handle it efficiently
            is_causal = self.causal

        # Apply scaled dot-product attention (uses FlashAttention when available)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal
        )  # [B, n_heads, T, head_dim]

        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).contiguous()  # [B, T, n_heads, head_dim]
        attn_out = attn_out.reshape(B, T, C)  # [B, T, d_model]

        return self.out_proj(attn_out)


def _check_flash_attention():
    """Check if FlashAttention-2 is available and will be used."""
    if torch.cuda.is_available():
        # PyTorch 2.0+ automatically uses FlashAttention when available
        # Check if the current CUDA device supports it
        device_capability = torch.cuda.get_device_capability()
        if device_capability[0] >= 8:  # Ampere or newer (compute capability 8.0+)
            print(f"✓ FlashAttention-2 available on {torch.cuda.get_device_name()}")
        else:
            print(f"⚠ FlashAttention-2 not available on {torch.cuda.get_device_name()} "
                  f"(requires compute capability 8.0+, got {device_capability[0]}.{device_capability[1]})")

    # Verify scaled_dot_product_attention is available
    if hasattr(F, 'scaled_dot_product_attention'):
        # Check which backends are available
        from torch.backends.cuda import sdp_kernel
        if torch.cuda.is_available():
            with sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False) as context:
                try:
                    # Test if flash attention can be used
                    test_q = torch.randn(1, 8, 16, 64, device='cuda', dtype=torch.float16)
                    test_k = test_v = test_q
                    _ = F.scaled_dot_product_attention(test_q, test_k, test_v)
                    print("✓ FlashAttention-2 kernel confirmed available")
                except:
                    print("⚠ FlashAttention-2 kernel not available, will use memory-efficient attention")
