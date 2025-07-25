import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.separator.base_separator import BaseSeparator, build_FFN, ResidualBlock
from src.helpers import ms_to_samples


class GhostSeparator:
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            streaming_mode: bool,
            context_size_ms: float,
            pos_enc: bool,
            name,
            **kwargs
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.streaming_mode = streaming_mode
        self.context_size_ms = context_size_ms
        self.pos_enc = pos_enc
        self.stateful = True
        self.name = name

    def reset_state(self, batch_size, chunk_len, dtype):
        pass
