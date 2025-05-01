from typing import Tuple, Any
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from src.models.submodules import SubModule


class TasNet(nn.Module):
    def __init__(self, encoder: DictConfig, decoder: DictConfig, separator: DictConfig, feature_dim: int,
                 stft_stride: int, **kwargs: Any):
        super().__init__()
        # instantiate enc/dec from their Hydra blocks
        self.encoder: SubModule = instantiate(encoder)
        self.decoder: SubModule = instantiate(decoder)

        # STFT params
        self.stft_window = feature_dim
        self.stft_stride = stft_stride

        # Compute separator input/output dimensions
        D = self.encoder.get_output_dim()  # encoder channel dim
        F = self.stft_window // 2 + 1  # freq‐bins from STFT
        separator.input_dim = 2 * D + 3 * F  # left+right (2D) + spatial (3F)
        separator.output_dim = 2 * D  # two masks of size D

        print(f"Separator config: {separator}")
        self.separator: SubModule = instantiate(separator)

    def pad_signal(self, input):
        # Ensure input is shape (B, 1, T) or (B, 2, T)
        if input.dim() == 2:
            input = input.unsqueeze(1)  # [B, 1, T]
        batch, ch, T = input.shape

        # 2) Compute how much “rest” padding you need so (T + 2*pad) is divisible by win/stride
        win, stride = self.win, self.stride
        rest = win - (stride + T % win) % win
        if rest > 0:
            pad = input.new_zeros(batch, ch, rest)
            input = torch.cat([input, pad], dim=2)

        # 3) Pad an extra `stride` on both ends
        pad_aux = input.new_zeros(batch, ch, stride)
        input = torch.cat([pad_aux, input, pad_aux], dim=2)

        return input, rest

    def compute_spatial_features(self, left_waveform: torch.Tensor, right_waveform: torch.Tensor) -> torch.Tensor:
        """
        Extracts ILD, cosIPD, and sinIPD from binaural inputs.
        Args:
            left_waveform, right_waveform: [batch, time]
        Returns:
            spatial_feats: [batch, frames, 3 * freq_bins]
        """
        # 1) STFT transform
        stft_left = torch.stft(left_waveform,
                               n_fft=self.stft_window,
                               hop_length=self.stft_stride,
                               win_length=self.stft_window,
                               return_complex=True)  # [B, F, T]
        stft_right = torch.stft(right_waveform,
                                n_fft=self.stft_window,
                                hop_length=self.stft_stride,
                                win_length=self.stft_window,
                                return_complex=True)

        # 2) Magnitude & phase
        mag_left, phase_left = stft_left.abs(), torch.angle(stft_left)
        mag_right, phase_right = stft_right.abs(), torch.angle(stft_right)

        # 3) Interaural Level Difference (ILD)
        ild = 10 * (mag_left.log10() - mag_right.log10())  # [B, F, T]

        # 4) Interaural Phase Difference (IPD) -> cos & sin
        ipd = phase_left - phase_right
        cos_ipd = torch.cos(ipd)  # [B, F, T]
        sin_ipd = torch.sin(ipd)

        # 5) Stack and reshape: [B, T, 3*F]
        #    Permute to time-major then flatten freq features
        batch, freq_bins, frames = ild.shape
        ild_t = ild.permute(0, 2, 1)  # [B, T, F]
        cos_t = cos_ipd.permute(0, 2, 1)  # [B, T, F]
        sin_t = sin_ipd.permute(0, 2, 1)  # [B, T, F]

        # stack into time-major then swap to channels-first:
        spatial_feats = torch.cat([ild_t, cos_t, sin_t], dim=-1)  # [B, T, 3F]
        return spatial_feats.permute(0, 2, 1)  # [B, 3F, T]

    def forward(self, mixture: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mixture: [batch, 2, time_samples]
        Returns:
            (left_out, right_out) for binaural
            or mono_out if mono_fallback=True
        """
        left_input, right_input = mixture[:, 0], mixture[:, 1]

        # Preprocess
        encoded_left = self.encoder(left_input)
        encoded_right = self.encoder(right_input)
        spatial_feats = self.compute_spatial_features(left_input, right_input)

        # Fusion: [left; right; spatial]
        fused_input = torch.cat([encoded_left,
                                 encoded_right,
                                 spatial_feats],
                                dim=1)  # [B, frames, 2D + 3F]

        # Produce isolation masks for left/right channels
        mask_estimates = self.separator(fused_input)   # [Batch, 2*Channel_dim, Time]
        mask_left, mask_right = mask_estimates.chunk(2, dim=1)  # each [B, frames, D]

        # Apply masks to encoded features
        masked_left = mask_left * encoded_left
        masked_right = mask_right * encoded_right

        # Decode to time-domain waveforms
        left_output = self.decoder(masked_left)
        right_output = self.decoder(masked_right)

        return left_output, right_output
