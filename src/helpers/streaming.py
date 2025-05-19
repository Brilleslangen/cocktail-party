from collections import deque
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def chunkify_batch(
        batch: torch.Tensor,
        original_lengths: torch.LongTensor,
        chunk_size: int) -> Tuple[Tensor, Tensor]:
    """
    Splits each audio sample in the batch into non-overlapping hop-sized chunks.
    Pads the last chunk of each sample with zeros if it's shorter than hop_size.

    Args:
        batch: Tensor of shape [B, C, T_max]
        original_lengths: 1D LongTensor of length B, containing the true length of each sample
        chunk_size: Number of samples per chunk (hop)

    Returns:
        chunks: Tensor of shape [B, max_num_chunks, C, chunk_size]
        num_chunks_per_sample: LongTensor of shape [B], number of valid chunks per sample
    """
    batch_size, num_channels, max_time = batch.shape
    num_chunks_per_sample = ((original_lengths + chunk_size - 1) // chunk_size)
    max_chunks = int(num_chunks_per_sample.max().item())  # maximum number of chunks across the batch

    # Allocate output tensor: [B, max_chunks, C, chunk_size], filled with zeros
    chunk_batch = batch.new_zeros((batch_size, max_chunks, num_channels, chunk_size))

    # Populate the chunks tensor with actual data
    for batch_idx in range(batch_size):
        true_length = original_lengths[batch_idx].item()
        n_chunks = int(num_chunks_per_sample[batch_idx].item())
        for chunk_idx in range(n_chunks):
            start_chunk = chunk_idx * chunk_size
            end_chunk = min(start_chunk + chunk_size, true_length)
            segment = batch[batch_idx, :, start_chunk:end_chunk]  # [C, <=chunk_size]
            chunk_batch[batch_idx, chunk_idx, :, : end_chunk - start_chunk] = segment

    return chunk_batch, num_chunks_per_sample


class Streamer:
    def __init__(self, model, device):
        self.model = model
        self.buffer_len = getattr(model, "buffer_len", None)
        self.chunk_size = getattr(model, "analysis_hop", None)

        if self.buffer_len is None or self.chunk_size is None:
            raise ValueError("Model must define buffer_len and analysis_hop")

        self.buffer = None  # We initialize it for each batch to accommodate different batch sizes = parallelization
        if hasattr(model.separator, "reset_state"):
            model.separator.reset_state()

    def init_buffer(self, B: int, C: int):
        self.buffer = torch.zeros(B, C, self.buffer_len, device=self.buffer.device)

    def push(self, new_samples: Tensor) -> Optional[Tensor]:
        # new_samples: [B, C, stride_len]
        self.buffer.extend(new_samples.unbind(-1))
        if len(self.buffer) < self.window_len:
            return None  # still warming up
        # once full, assemble window and call model
        window = torch.stack(list(self.buffer))  # shape [window_len]
        out = self.model(window.unsqueeze(0))  # returns [B, C, stride_len]
        return out

    def forward(self, batch: Tensor) -> Optional[Tensor]:
        """
        Forward pass through the model with streaming.
        Args:
            batch (Tensor): [B, C, T]
        Returns:
            Tensor: [B, C, T] or None if not enough samples
        """
        # split into chunks
        chunks = self.split_streams_into_chunks(batch)
        out_chunks = []
        for chunk in chunks:
            out_chunk = []
            for window in chunk:
                out = self.push(window)
                if out is not None:
                    out_chunk.append(out)
            if out_chunk:
                out_chunks.append(torch.cat(out_chunk, dim=-1))
        return torch.cat(out_chunks, dim=-1) if out_chunks else None
