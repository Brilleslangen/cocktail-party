from types import SimpleNamespace

import torch
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
        headdim: int,
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
        self.dropout_val = dropout_val
        self.headdim = headdim

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
            headdim=self.headdim,
            dropout_val=self.dropout_val,
            causal=self.causal,
            chunk_len=self.frames_per_output,
            layer_idx=block_idx,
        )


class MambaBlock(ResidualBlock):
    """Single Mamba-2 block used inside the separator."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, headdim: int,
                 d_ff: int, dropout_val: float, causal: bool, chunk_len: int, layer_idx: int,
                 **kwargs):
        self.d_state = d_state
        self.headdim = headdim
        self.d_conv = d_conv
        self.expand = expand
        self.chunk_len = chunk_len
        self.layer_idx = layer_idx
        super().__init__(d_model, d_ff, dropout_val, causal,
                         post_core_gelu=False, stateful=True)

    def _build_core_layer(self) -> nn.Module:
        class MambaWrapper(nn.Module):
            def __init__(self, d_model, d_state, d_conv, headdim, expand, layer_idx, chunk_len):
                super().__init__()
                self.mamba = Mamba2(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    headdim=headdim,
                    expand=expand,
                    layer_idx=layer_idx,
                )
                self.layer_idx = layer_idx
                self.chunk_len = chunk_len if chunk_len is not None else 0
                self.curr_position = 0  # Track current position in the sequence

            def build_fresh_state(self, batch_size: int, max_seqlen: int):  # Check layer_idx
                conv_state, ssm_state = self.mamba.allocate_inference_cache(batch_size, max_seqlen)
                inference_params = SimpleNamespace(
                    key_value_memory_dict={self.layer_idx: (conv_state, ssm_state)},
                    seqlen_offset=0  # Start from 0 for chunk processing
                )
                return inference_params

            def forward(self, x, state=None):
                if state is not None:
                    B, T, D = x.shape  # [batch, time, features]
                    outputs = []

                    state.seqlen_offset = self.curr_position

                    if self.curr_position < 100:
                        print(state.key_value_memory_dict[self.layer_idx][0].shape)

                    for t in range(T):
                        frame = x[:, t:t+1, :]
                        out_frame = self.mamba(frame, inference_params=state)
                        outputs.append(out_frame)

                        # Update offset for next frame
                        state.seqlen_offset += 1

                    # Concatenate outputs: [B, T, D]
                    out = torch.cat(outputs, dim=1)

                    self.curr_position += self.chunk_len  # Increment position by chunk length due to buf roll
                    
                    return out, state
                else:
                    # No state - offline processing
                    out = self.mamba(x)
                    return out, None

        return MambaWrapper(self.d_model, self.d_state, self.d_conv, self.headdim, self.expand, self.layer_idx, 
                            self.chunk_len)
