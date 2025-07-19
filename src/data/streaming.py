from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


class Streamer:
    def __init__(self, model: torch.nn.Module):
        """
        model: your TasNet (or other) separator.
          Must define model.analysis_window (buffer_len) and model.analysis_hop (chunk_size).
          Must accept inputs of shape [B, C, buffer_len] and output [B, C, chunk_size].
        """
        self.model = model
        self.buffer_size = getattr(model, "input_size", None)
        self.chunk_size = getattr(model, "output_size", None)
        if not self.buffer_size or not self.chunk_size:
            raise ValueError("Model must define context size and chunk size")
        self.buffer: Optional[Tensor] = None
        self.pad_warmup = self.buffer_size - self.chunk_size
        self.device = model.device

    def reset(self, batch_size: int, channels: int):
        """Zero the ring‐buffer and reset any model state."""
        self.buffer = torch.zeros(batch_size, channels, self.buffer_size, device=self.device)

    def push(self, new_chunk: Tensor) -> Optional[Tensor]:
        """
        new_chunk: [B, C, chunk_size]
        Returns:
          [B, C, chunk_size] regardless if buffer is filled or not.
        """
        self.buffer = torch.roll(self.buffer, shifts=-self.chunk_size, dims=-1)  # roll buffer left by chunk_size
        self.buffer[:, :, -self.chunk_size:] = new_chunk  # store new audio at the right
        out = self.model(self.buffer)  # [B, C, chunk_size]
        return out.cpu()

    def stream_batch(self, mix_batch: Tensor, refs: Tensor, lengths: torch.Tensor,
                     trim_warmup=True) -> tuple[Tensor, Tensor, Tensor]:
        B, C, T = mix_batch.shape

        # Reinitialize buffer
        self.reset(batch_size=B, channels=C)

        out_full = torch.zeros(B, C, T)
        for i, chunk in enumerate(iter_chunks(mix_batch, self.chunk_size)):
            est = self.push(chunk)
            start = i * self.chunk_size
            end = min(start + self.chunk_size, T)
            out_full[..., start:end] = est[..., :end - start]

        ref_trimmed = refs[..., self.pad_warmup:] if trim_warmup else refs
        est_trimmed = (out_full[..., self.pad_warmup:T] if trim_warmup else out_full[..., :T])

        lengths_trimmed = lengths - self.pad_warmup  # Check this later when we mask and sum

        return est_trimmed, ref_trimmed, lengths_trimmed


def iter_chunks(batch: Tensor, chunk_size: int):
    T = batch.size(-1)
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk = batch[..., start:end]
        if chunk.size(-1) < chunk_size:  # Pad the last chunk if it's smaller than chunk_size
            chunk = F.pad(chunk, (0, chunk_size - chunk.size(-1)))
        yield chunk