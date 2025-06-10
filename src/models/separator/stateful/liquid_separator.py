import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP
from src.models.separator.base_separator import BaseSeparator, ResidualBlock


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
            d_ff: int,
            dropout_val: float,
            num_neurons: int,
            frames_per_output: int,
            streaming_mode: bool,
            context_size_ms: float,
            causal_proj: bool,
            name: str,
            **kwargs
    ):
        # Ensure we have enough neurons
        self.num_neurons = max(num_neurons, d_model + 3)

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
        return LiquidBlock(
            d_model=self.d_model,
            num_neurons=self.num_neurons,
            d_ff=self.d_ff,
            dropout_val=self.dropout,
            causal=self.causal,
        )


class LiquidBlock(ResidualBlock):
    """
    A single Liquid Neural Network block with CfC neurons.
    """

    def __init__(self, d_model: int, num_neurons: int, d_ff: int, dropout_val: float, causal: bool):
        self.num_neurons = num_neurons
        super().__init__(d_model, d_ff, dropout_val, causal, post_core_gelu=False, stateful=True)

    def _build_core_layer(self) -> nn.Module:
        wiring = AutoNCP(self.num_neurons, self.d_model)
        return CfC(self.d_model, wiring, batch_first=True)
