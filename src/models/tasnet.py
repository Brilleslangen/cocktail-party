from typing import Tuple
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.config import TasNetConfig


class TasNet(nn.Module):
    def __init__(self, config: TasNetConfig | DictConfig):
        super().__init__()
        self.encoder = instantiate(config.encoder)
        self.decoder = instantiate(config.decoder)

        # --- Spatial feature params ---
        self.stft_window = config.feature_dim
        self.stft_hop = config.stft_stride

        # Separator input dim = 2*D + 3*F (ILD + cos(IPD) + sin(IPD))
        freq_bins = self.stft_window // 2 + 1
        sep_input_dim = 2 * config.feature_dim + 3 * freq_bins
        # self.separator = instantiate(config.separator, input_dim=sep_input_dim)

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
                               hop_length=self.stft_hop,
                               win_length=self.stft_window,
                               return_complex=True)  # [B, F, T]
        stft_right = torch.stft(right_waveform,
                                n_fft=self.stft_window,
                                hop_length=self.stft_hop,
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

        spatial_feats = torch.cat([ild_t, cos_t, sin_t], dim=-1)  # [B, T, 3F]
        return spatial_feats

    def forward(self, mixture: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mixture: [batch, 2, time_samples]
        Returns:
            (left_out, right_out) for binaural
            or mono_out if mono_fallback=True
        """
        left_input, right_input = mixture[:, 0], mixture[:, 1]  # each [B, T]

        # Preprocess
        encoded_left = self.encoder(left_input)  # [B, frames, D]
        encoded_right = self.encoder(right_input)  # [B, frames, D]
        spatial_feats = self.compute_spatial_features(left_input, right_input) # [B, frames, 3*F]

        # Fusion: [left; right; spatial]
        fused_input = torch.cat([encoded_left,
                                 encoded_right,
                                 spatial_feats],
                                dim=-1)  # [B, frames, 2D + 3F]

        # Produce isolation masks for left/right channels
        mask_estimates = self.separator(fused_input)  # [B, frames, 2D]
        mask_left, mask_right = mask_estimates.chunk(2, dim=-1)  # each [B, frames, D]

        # Apply masks to encoded features
        masked_left = mask_left * encoded_left
        masked_right = mask_right * encoded_right

        # Decode to time-domain waveforms
        left_output = self.decoder(masked_left)
        right_output = self.decoder(masked_right)

        return left_output, right_output


class VanillaEncoder(nn.Module):
    """
    Time-domain analysis filterbank as in Conv-TasNet:
      1) 1-D Conv (1→N filters)
      2) Non-negative activation (ReLU)
    See Luo & Mesgarani
    """
    def __init__(self, N: int, L: int, stride: int, causal: bool):
        super().__init__()
        padding = L - stride if causal else (L // 2)
        self.conv1d = nn.Conv1d(1, N, kernel_size=L,
                                stride=stride, padding=padding, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,  T] or [B, 1, T]
        returns: [B, N, T_frames]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B,1,T]
        w = self.conv1d(x)  # [B,N,T']
        w = self.relu(w)  # enforce non-negativity
        return w


class VanillaDecoder(nn.Module):
    """
    Inverse filterbank via transposed convolution:
      1) 1-D ConvTranspose (N→1 filters)
      2) Overlap-add reconstruction
    """

    def __init__(self, N: int, L: int, stride: int, causal: bool):
        super().__init__()
        padding = L - stride if causal else (L // 2)
        self.deconv1d = nn.ConvTranspose1d(N, 1, kernel_size=L, stride=stride, padding=padding, bias=False)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """
        w: [B, N, T_frames]
        returns: [B, 1, T]  or squeeze to [B, T]
        """
        x_hat = self.deconv1d(w)  # [B,1,T]
        return x_hat.squeeze(1)  # [B, T]

