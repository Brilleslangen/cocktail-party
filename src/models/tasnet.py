from typing import Tuple, Any
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.helpers import ms_to_samples
from src.models.submodules import SubModule


class TasNet(nn.Module):
    """
    Binaural Conv-TasNet model with spatial feature fusion.

    Separates a stereo mixture into left/right outputs by:
      1) encoding each channel,
      2) extracting interaural spatial features (ILD, cosIPD, sinIPD),
      3) concatenating channels + spatial features,
      4) running a separator to predict masks,
      5) applying masks and decoding back to waveforms.
    """

    def __init__(self, encoder: DictConfig, decoder: DictConfig, separator: DictConfig, sample_rate: int,
                 window_length_ms: float, stride_ms: float, **kwargs: Any):
        super().__init__()

        # -------------------------------------------------------------------
        # Instantiate encoder and decoder from their Hydra configs
        # -------------------------------------------------------------------
        self.encoder: SubModule = instantiate(encoder)
        self.decoder: SubModule = instantiate(decoder)

        # -------------------------------------------------------------------
        # STFT parameters for spatial features
        # -------------------------------------------------------------------
        self.stft_window = ms_to_samples(window_length_ms, sample_rate)  # STFT window length (samples)
        self.stft_hop = ms_to_samples(stride_ms, sample_rate)  # STFT hop length (samples)
        self.register_buffer("stft_window_fn", torch.hann_window(self.stft_window))

        # -------------------------------------------------------------------
        # Compute separator's channel dimensions
        # -------------------------------------------------------------------
        D = self.encoder.get_output_dim()  # number of encoder filters
        F = self.stft_window // 2 + 1  # number of STFT freq bins
        sep_input_dim = 2 * D + 3 * F  # [left; right; spatial]
        sep_output_dim = 2 * D  # two masks of size D

        # Instantiate separator with computed dims
        self.separator: SubModule = instantiate(separator, input_dim=sep_input_dim, output_dim=sep_output_dim)

    def pad_signal(self, mixture: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Pads a mono or stereo waveform so its length aligns with encoder/STFT hops,
        and adds context padding on both ends.

        Args:
            mixture (torch.Tensor): [batch, channels, time] where channels=1 or 2.

        Returns:
            padded (torch.Tensor): [batch, channels, time + rest + 2*hop]
            rest (int): number of zeros appended at the end before context padding.
        """
        B, C, T = mixture.shape
        win, hop = self.stft_window, self.stft_hop

        # Make (T + rest) divisible by window to align frames
        rest = win - (hop + T % win) % win
        if rest > 0:
            pad_end = mixture.new_zeros(B, C, rest)
            mixture = torch.cat([mixture, pad_end], dim=2)

        # Add extra hop-length padding on both start and end
        pad_ctx = mixture.new_zeros(B, C, hop)
        mixture = torch.cat([pad_ctx, mixture, pad_ctx], dim=2)
        return mixture, rest

    def compute_spatial_features(self, left_waveform: torch.Tensor, right_waveform: torch.Tensor) -> torch.Tensor:
        """
        Compute interaural spatial cues via STFT.

        Args:
            left_waveform (torch.Tensor): [batch, time]
            right_waveform (torch.Tensor): [batch, time]

        Returns:
            spatial_feats (torch.Tensor): [batch, 3*F, T_frames]
                concatenated ILD, cos(IPD), and sin(IPD) features.
        """
        eps = 1e-8  # small value to prevent log(0)

        # 1) STFT
        stft_left = torch.stft(
            left_waveform,
            n_fft=self.stft_window,
            hop_length=self.stft_hop,
            win_length=self.stft_window,
            window=self.stft_window_fn,
            return_complex=True
        )  # [B, F, T]
        stft_right = torch.stft(
            right_waveform,
            n_fft=self.stft_window,
            hop_length=self.stft_hop,
            win_length=self.stft_window,
            window=self.stft_window_fn,
            return_complex=True
        )  # [B, F, T]

        # 2) Magnitude & Phase
        mag_left = stft_left.abs() + eps
        mag_right = stft_right.abs() + eps
        phase_left = torch.angle(stft_left)
        phase_right = torch.angle(stft_right)

        # 3) ILD (Interaural Level Difference)
        ild = 10.0 * (mag_left.log10() - mag_right.log10())  # [B, F, T]

        # 4) IPD (Interaural Phase Difference)
        ipd = phase_left - phase_right
        cos_ipd = torch.cos(ipd)
        sin_ipd = torch.sin(ipd)

        # 5) Stack [ILD; cos(IPD); sin(IPD)] → [B, 3F, T]
        ild_t = ild.permute(0, 2, 1)
        cos_t = cos_ipd.permute(0, 2, 1)
        sin_t = sin_ipd.permute(0, 2, 1)
        stacked = torch.cat([ild_t, cos_t, sin_t], dim=2)
        return stacked.permute(0, 2, 1)  # [B, 3F, T]

    def forward(self, mixture: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the binaural TasNet.

        Args:
            mixture (torch.Tensor): [batch, 2, time_samples]

        Returns:
            left_out (torch.Tensor): [batch, time_samples]
            right_out (torch.Tensor): [batch, time_samples]
        """
        # Pad for alignment
        mixture, rest = self.pad_signal(mixture)
        left, right = mixture[:, 0], mixture[:, 1]

        # Encode each channel
        enc_left = self.encoder(left)  # [B, D, T_frames]
        enc_right = self.encoder(right)  # [B, D, T_frames]

        # Compute spatial features on padded signals
        sp_feats = self.compute_spatial_features(left, right)  # [B, 3F, T_frames]

        # Fuse features channel-wise
        fused = torch.cat([enc_left, enc_right, sp_feats], dim=1)  # [B, 2D+3F, T_frames]

        # Estimate masks
        masks = self.separator(fused)  # [B, 2D, T_frames]
        mL, mR = masks.chunk(2, dim=1)  # each [B, D, T_frames]

        # Apply masks and decode
        outL = self.decoder(mL * enc_left)  # [B, 1, T_padded]
        outR = self.decoder(mR * enc_right)  # [B, 1, T_padded]

        # Remove padding to recover original length
        hop = self.stft_hop
        start, end = hop, -(rest + hop)
        left_out = outL[:, :, start:end].squeeze(1)  # [B, T]
        right_out = outR[:, :, start:end].squeeze(1)  # [B, T]

        return left_out, right_out
