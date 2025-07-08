from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


class Streamer:
    def __init__(self, model: torch.nn.Module, use_pinned_memory: bool = True):
        self.model = model
        self.buffer_size = getattr(model, "input_size", None)
        self.chunk_size = getattr(model, "output_size", None)
        if not self.buffer_size or not self.chunk_size:
            raise ValueError("Model must define context size and chunk size")
        self.pad_warmup = self.buffer_size - self.chunk_size
        self.device = next(model.parameters()).device
        self.use_pinned = use_pinned_memory and torch.cuda.is_available()

        # Pre-allocate pinned memory buffers for faster transfers
        if self.use_pinned:
            self.chunk_buffer = None  # Will be allocated on first use
            self.output_buffer = None

    def reset(self, batch_size: int, channels: int):
        """Initialize GPU buffer and optionally pinned memory buffers."""
        self.buffer = torch.zeros(batch_size, channels, self.buffer_size, device=self.device)

        if self.use_pinned and self.chunk_buffer is None:
            # Allocate pinned memory for faster CPU-GPU transfers
            self.chunk_buffer = torch.zeros(batch_size, channels, self.chunk_size,
                                            pin_memory=True)
            self.output_buffer = torch.zeros(batch_size, channels, self.chunk_size,
                                             pin_memory=True)

    def push(self, new_chunk: Tensor) -> Tensor:
        """Process chunk with optimized memory transfers."""
        if self.use_pinned:
            # Copy to pinned memory first (non-blocking)
            self.chunk_buffer.copy_(new_chunk, non_blocking=True)
            chunk_gpu = self.chunk_buffer.to(self.device, non_blocking=True)
        else:
            chunk_gpu = new_chunk.to(self.device)

        # Update buffer
        self.buffer = torch.roll(self.buffer, shifts=-self.chunk_size, dims=-1)
        self.buffer[:, :, -self.chunk_size:] = chunk_gpu

        # Process
        with torch.cuda.amp.autocast(enabled=False):  # Disable if not needed
            out = self.model(self.buffer)

        if self.use_pinned:
            # Use pinned memory for output transfer
            self.output_buffer.copy_(out, non_blocking=True)
            torch.cuda.synchronize()  # Ensure transfer completes
            return self.output_buffer.clone()
        else:
            return out.cpu()

    def stream_batch(self, mix_batch: Tensor, refs: Tensor, lengths: torch.LongTensor,
                     trim_warmup=True) -> tuple[Tensor, Tensor, Tensor]:
        """Same as Streamer but with pinned memory optimization."""
        B, C, T = mix_batch.shape

        # Move to CPU if needed
        mix_cpu = mix_batch.cpu() if mix_batch.is_cuda else mix_batch
        refs_cpu = refs.cpu() if refs.is_cuda else refs

        self.reset(batch_size=B, channels=C)

        # Pre-allocate output tensor on CPU
        n_chunks = (T + self.chunk_size - 1) // self.chunk_size
        out_full = torch.zeros(B, C, n_chunks * self.chunk_size,
                               pin_memory=self.use_pinned)

        # Process chunks
        chunk_idx = 0
        for chunk in iter_chunks(mix_cpu, self.chunk_size):
            est = self.push(chunk)
            out_full[..., chunk_idx * self.chunk_size:(chunk_idx + 1) * self.chunk_size] = est
            chunk_idx += 1

        # Trim
        if trim_warmup:
            ref_trimmed = refs_cpu[..., self.pad_warmup:]
            est_trimmed = out_full[..., self.pad_warmup:T]
            lengths_trimmed = lengths - self.pad_warmup
        else:
            ref_trimmed = refs_cpu
            est_trimmed = out_full[..., :T]
            lengths_trimmed = lengths

        # Final GPU transfer
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
