import torch.nn as nn
from mamba_ssm import Mamba2
from src.models.separator.base_separator import BaseSeparator, ResidualBlock


class MambaSeparator(BaseSeparator):
    """Mamba-2 based separator following the BaseSeparator design."""

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            d_model: int,
            n_blocks: int,
            d_state: int,
            d_conv: int,
            expand: int,
            d_ff: int,
            dropout_val: float,
            frames_per_output: int,
            streaming_mode: bool,
            context_size_ms: float,
            causal_proj: bool,
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
            d_ff=d_ff,
            dropout_val=dropout_val,
            frames_per_output=frames_per_output,
            streaming_mode=streaming_mode,
            context_size_ms=context_size_ms,
            name=name,
            causal=causal_proj,
            **kwargs
        )

    def _build_block(self, block_idx: int) -> nn.Module:
        return MambaBlock(
            d_model=self.d_model,
            d_state=self.d_state,
            d_conv=self.d_conv,
            expand=self.expand,
            d_ff=self.d_ff,
            dropout_val=self.dropout,
            causal=self.causal,
        )


class MambaBlock(ResidualBlock):
    """Single Mamba-2 block used inside the separator."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, headdim: int,
                 d_ff: int, dropout_val: float, causal: bool, **kwargs):
        self.d_state = d_state
        self.headdim = headdim
        self.d_conv = d_conv
        self.expand = expand
        super().__init__(d_model, d_ff, dropout_val, causal,
                         post_core_gelu=False, stateful=True)

    def _build_core_layer(self) -> nn.Module:
        class MambaWrapper(nn.Module):
            def __init__(self, d_model, d_state, d_conv, headdim, expand):
                super().__init__()
                self.mamba = Mamba2(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    headdim=headdim,
                    expand=expand,
                )

            def forward(self, x, state=None):
                if state is not None:
                    return self.mamba(x, initial_states=state)
                out = self.mamba(x)
                return out, getattr(self.mamba, "final_states", None)

        return MambaWrapper(self.d_model, self.d_state, self.d_conv, self.expand)
