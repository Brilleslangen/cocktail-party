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

    def push(self, new_chunk: Tensor) -> Tensor:
        """
        Process a chunk that's currently on CPU.

        Args:
            new_chunk: [B, C, chunk_size] on CPU

        Returns:
            [B, C, chunk_size] on CPU
        """
        # Move only this chunk to GPU
        chunk_gpu = new_chunk.to(self.device)

        # Update buffer (on GPU)
        self.buffer = torch.roll(self.buffer, shifts=-self.chunk_size, dims=-1)
        self.buffer[:, :, -self.chunk_size:] = chunk_gpu

        # Process through model (on GPU)
        out = self.model(self.buffer)  # [B, C, chunk_size]

        # Immediately move output back to CPU to free GPU memory
        return out.cpu()

    def stream_batch(self, mix_batch: Tensor, refs: Tensor, lengths: torch.LongTensor,
                     trim_warmup=True) -> tuple[Tensor, Tensor, Tensor]:
        """
        Stream process a batch, keeping only chunks on GPU.

        Args:
            mix_batch: [B, C, T] - should be on CPU
            refs: [B, C, T] - should be on CPU
            lengths: [B] - should be on CPU

        Returns:
            All outputs on GPU (moved back at the end)
        """
        B, C, T = mix_batch.shape

        # Ensure input is on CPU
        if mix_batch.is_cuda:
            mix_batch = mix_batch.cpu()
        if refs.is_cuda:
            refs = refs.cpu()

        # Initialize buffer on GPU
        self.reset(batch_size=B, channels=C)

        # Process chunks, accumulating on CPU
        out_chunks = []
        for chunk in iter_chunks(mix_batch, self.chunk_size):
            # chunk is on CPU, push handles GPU transfer
            est = self.push(chunk)  # Returns CPU tensor
            out_chunks.append(est)

        # Concatenate on CPU
        out_full = torch.cat(out_chunks, dim=-1)  # Still on CPU

        # Trim if needed (still on CPU)
        if trim_warmup:
            ref_trimmed = refs[..., self.pad_warmup:]
            est_trimmed = out_full[..., self.pad_warmup:T]
            lengths_trimmed = lengths - self.pad_warmup
        else:
            ref_trimmed = refs
            est_trimmed = out_full[..., :T]
            lengths_trimmed = lengths

        # Move final results to GPU only at the end
        return (est_trimmed.to(self.device),
                ref_trimmed.to(self.device),
                lengths_trimmed.to(self.device))


def iter_chunks(batch: Tensor, chunk_size: int):
    T = batch.size(-1)
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk = batch[..., start:end]
        if chunk.size(-1) < chunk_size:  # Pad the last chunk if it's smaller than chunk_size
            chunk = F.pad(chunk, (0, chunk_size - chunk.size(-1)))
        yield chunk
