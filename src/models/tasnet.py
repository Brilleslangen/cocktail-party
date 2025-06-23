import math
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
                 window_length_ms: float, stride_ms: float, streaming_mode: bool, stream_chunk_size_ms: float,
                 filter_length_ms, use_targets_as_input: bool, use_spatial_features: bool, no_separator: bool,
                 device: torch.device, **kwargs: Any):
        super().__init__()

        self.no_separator = no_separator  # Encoder decoder identity evaluation
        self.use_targets_as_input = use_targets_as_input  # Use target as input for training
        self.sample_rate = sample_rate  # Sample rate in Hz
        self.device = device

        # -------------------------------------------------------------------
        # Instantiate encoder and decoder from their Hydra configs
        # -------------------------------------------------------------------
        self.encoder: SubModule = instantiate(encoder)
        self.decoder: SubModule = instantiate(decoder)
        self.use_spatial_features = use_spatial_features

        # -------------------------------------------------------------------
        # STFT parameters for spatial features
        # -------------------------------------------------------------------
        self.analysis_window = ms_to_samples(window_length_ms, sample_rate)  # STFT window length (samples)
        self.analysis_hop = ms_to_samples(stride_ms, sample_rate)  # STFT hop length (samples)
        self.register_buffer("stft_window_fn", torch.hann_window(self.analysis_window).float())

        if self.use_spatial_features:
            self.spatial_feature_extractor = RobustSpatialFeatureExtractor(
                n_fft=self.analysis_window,
                eps=1e-8
            )

        # -------------------------------------------------------------------
        # Compute separator's channel dimensions
        # -------------------------------------------------------------------
        D = self.encoder.get_output_dim()  # number of encoder filters
        F = self.analysis_window // 2 + 1  # number of STFT freq bins
        sep_input_dim = 2 * D  # left and right channels encoded
        if self.use_spatial_features:
            sep_input_dim += 3 * F  # [left; right; spatial]
        sep_output_dim = 2 * D  # two masks of size D

        # Separator streaming params
        self.output_size = ms_to_samples(stream_chunk_size_ms, sample_rate)  # samples of raw audio
        self.frames_per_output = math.ceil((self.output_size + 2 * self.decoder.pad - self.decoder.filter_length)
                                           / self.analysis_hop) + 1  # Only for streaming mode

        # Instantiate separator with computed dims
        self.separator: SubModule = instantiate(separator, input_dim=sep_input_dim, output_dim=sep_output_dim,
                                                frames_per_output=self.frames_per_output)

        # Streaming params
        self.streaming_mode = streaming_mode  # Size input timespan
        self.input_size = ms_to_samples(max(window_length_ms, self.separator.context_size_ms), sample_rate)
        self.sep_context_size = ms_to_samples(self.separator.context_size_ms, sample_rate)  # samples of raw audio
        self.frames_per_context = int(self.separator.context_size_ms // stride_ms) + 1  # number of input frames
        print(f'frames_per_context: {self.frames_per_context}, frames_per_output: {self.frames_per_output}')

    def reset_state(self, batch_size: int, chunk_len: int):
        """
        Reset the separator state if it has one.
        """
        if hasattr(self.separator, "reset_state"):
            self.separator.reset_state(batch_size, chunk_len)

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
        win, hop = self.analysis_window, self.analysis_hop

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
        eps = 1e-2  # Small value to avoid log(0) issues

        # Compute STFT
        stft_left = torch.stft(
            left_waveform,
            n_fft=self.analysis_window,
            hop_length=self.analysis_hop,
            win_length=self.analysis_window,
            window=self.stft_window_fn,
            return_complex=True
        )  # [B, F, T]

        stft_right = torch.stft(
            right_waveform,
            n_fft=self.analysis_window,
            hop_length=self.analysis_hop,
            win_length=self.analysis_window,
            window=self.stft_window_fn,
            return_complex=True
        )  # [B, F, T]

        # Magnitude & Phase
        mag_left = stft_left.abs().clamp(min=eps).float()  # Clamp to avoid zero
        mag_right = stft_right.abs().clamp(min=eps).float()
        phase_left = torch.angle(stft_left).float()
        phase_right = torch.angle(stft_right).float()

        # ILD (Interaural Level Difference) - more stable computation
        if torch.any(mag_left <= 0):
            print("Non-positive mag_left before log10!", mag_left.min().item())
            mag_left = mag_left.clamp(min=eps).float()
        if torch.any(mag_right <= 0):
            print("Non-positive mag_right before log10!", mag_right.min().item())
            mag_right = mag_right.clamp(min=eps).float()
        ild = 10.0 * (torch.log10(mag_left) - torch.log10(mag_right))
        ild = torch.nan_to_num(ild, nan=0.0, posinf=60.0, neginf=-60.0)
        ild = torch.clamp(ild, min=-60.0, max=60.0)

        # IPD (Interaural Phase Difference)
        ipd = phase_left - phase_right
        if torch.any(mag_right <= 0):
            print("Non-positive mag_right before log10!", phase_left.min().item(), phase_right.min().item())
        ipd = torch.remainder(ipd + torch.pi, 2 * torch.pi) - torch.pi
        cos_ipd = torch.cos(ipd)
        sin_ipd = torch.sin(ipd)

        # Stack [ILD; cos(IPD); sin(IPD)] → [B, 3F, T]
        ild_t = ild.permute(0, 2, 1)
        cos_t = cos_ipd.permute(0, 2, 1)
        sin_t = sin_ipd.permute(0, 2, 1)
        stacked = torch.cat([ild_t, cos_t, sin_t], dim=2)
        spatial_features = stacked.permute(0, 2, 1)  # [B, 3F, T]

        # Compute statistics for normalization
        mean = spatial_features.mean(dim=(1, 2), keepdim=True)
        std = spatial_features.std(dim=(1, 2), keepdim=True).clamp(min=eps)
        spatial_features = (spatial_features - mean) / std + 1e-8  # Normalize to zero mean, unit variance

        spatial_features = torch.clamp(spatial_features, min=-8.0, max=8.0)  # Training stability

        return spatial_features

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the binaural TasNet.

        Args:
            mixture (torch.Tensor): [batch, 2, time_samples]

        Returns:
            left_out (torch.Tensor): [batch, time_samples]
            right_out (torch.Tensor): [batch, time_samples]
        """
        # Pad for alignment in offline mode
        mixture, rest = (mixture, None) if self.streaming_mode else self.pad_signal(mixture)
        # print('rest', rest)
        left, right = mixture[:, 0], mixture[:, 1]

        sp_feats = None
        if self.use_spatial_features:
            sp_feats = self.compute_spatial_features(left, right)

        if self.streaming_mode:
            enc_left = self.encoder(left[..., -self.sep_context_size:])
            enc_right = self.encoder(right[..., -self.sep_context_size:])
            if sp_feats is not None:
                sp_feats = sp_feats[..., -self.frames_per_context:]
        else:
            enc_left = self.encoder(left)
            enc_right = self.encoder(right)

        # Fuse features channel-wise
        if sp_feats is not None:
            fused = torch.cat([enc_left, enc_right, sp_feats], dim=1)  # [B, 2D+3F, T_frames]
        else:
            fused = torch.cat([enc_left, enc_right], dim=1)

        if self.streaming_mode:  # Only use last frames for output
            enc_left, enc_right = enc_left[..., -self.frames_per_output:], enc_right[..., -self.frames_per_output:]

        if self.no_separator:
            # Identity evaluation - skip separator
            outL = self.decoder(enc_left)
            outR = self.decoder(enc_right)
        else:  # Estimate masks
            masks = self.separator(fused)  # [B, 2D, T_frames]
            mL, mR = masks.chunk(2, dim=1)  # each [B, D, T_frames]

            # Apply masks and decode
            outL = self.decoder(mL * enc_left)  # [B, 1, T_padded]
            outR = self.decoder(mR * enc_right)  # [B, 1, T_padded]

        # Remove padding to recover original length
        if not self.streaming_mode:
            hop = self.analysis_hop
            start, end = hop, -(rest + hop)
            outL = outL[:, :, start:end]  # [B, T]
            outR = outR[:, :, start:end]  # [B, T]

        out = torch.stack([outL.squeeze(1), outR.squeeze(1)], dim=1)

        return out


def calculate_frames_per_output(chunk_size_ms: float, stride_ms: float, kernel_size_ms: float, sample_rate: int,
                                causal: bool = False) -> int:
    chunk_samples = ms_to_samples(chunk_size_ms, sample_rate)
    stride_samples = ms_to_samples(stride_ms, sample_rate)
    kernel_samples = ms_to_samples(kernel_size_ms, sample_rate)

    # Determine decoder padding in samples
    pad = (kernel_samples - stride_samples) if causal else (kernel_samples // 2)
    return math.ceil((chunk_samples + 2 * pad - kernel_samples) / stride_samples) + 1


class RobustSpatialFeatureExtractor(nn.Module):
    """
    Robust spatial feature extraction with learnable gating and smooth value bounding.
    Designed for stable mixed-precision training on CUDA.
    """

    def __init__(self, n_fft: int, eps: float = 1e-8):
        super().__init__()
        self.n_fft = n_fft
        self.n_freqs = n_fft // 2 + 1
        self.eps = eps

        # Learnable parameters for adaptive scaling and gating
        self.ild_scale = nn.Parameter(torch.ones(1))
        self.ipd_scale = nn.Parameter(torch.ones(1))

        # Learnable frequency-dependent gates (sigmoid will bound to [0,1])
        self.ild_gate = nn.Parameter(torch.ones(self.n_freqs))
        self.ipd_gate = nn.Parameter(torch.ones(self.n_freqs))

        # Learnable offset for stability
        self.ild_offset = nn.Parameter(torch.zeros(1))

        # Temporal smoothing conv (causal)
        self.smooth_kernel_size = 3
        self.ild_smooth = nn.Conv1d(self.n_freqs, self.n_freqs,
                                    kernel_size=self.smooth_kernel_size,
                                    padding=self.smooth_kernel_size - 1,
                                    groups=self.n_freqs, bias=False)
        self.ipd_smooth = nn.Conv1d(2 * self.n_freqs, 2 * self.n_freqs,
                                    kernel_size=self.smooth_kernel_size,
                                    padding=self.smooth_kernel_size - 1,
                                    groups=2 * self.n_freqs, bias=False)

        # Initialize smoothing kernels with exponential decay
        with torch.no_grad():
            # Causal exponential smoothing kernel
            kernel = torch.tensor([0.5, 0.3, 0.2]).float()
            self.ild_smooth.weight.data = kernel.view(1, 1, -1).repeat(self.n_freqs, 1, 1)
            self.ipd_smooth.weight.data = kernel.view(1, 1, -1).repeat(2 * self.n_freqs, 1, 1)

    def stable_log_ratio(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute log(x/y) in a numerically stable way.
        Uses log1p for better precision with small values.
        """
        # Add small epsilon for stability
        x_safe = x + self.eps
        y_safe = y + self.eps

        # Use log difference instead of log division
        # log(x/y) = log(x) - log(y)
        # But we'll use a more stable formulation
        ratio = x_safe / y_safe

        # For ratios close to 1, log1p is more accurate
        # log(ratio) = log(1 + (ratio - 1)) = log1p(ratio - 1)
        return torch.where(
            torch.abs(ratio - 1) < 0.5,
            torch.log1p(ratio - 1),
            torch.log(ratio)
        )

    def adaptive_layer_norm(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Adaptive layer normalization that's more stable than standard normalization.
        Uses running statistics for stability.
        """
        # Compute statistics with higher precision
        mean = x.mean(dim=dim, keepdim=True)
        var = x.var(dim=dim, keepdim=True, unbiased=False)

        # Stable normalization with adaptive epsilon based on variance magnitude
        adaptive_eps = self.eps + 0.1 * torch.sqrt(var.detach())
        x_norm = (x - mean) / torch.sqrt(var + adaptive_eps)

        return x_norm

    def smooth_bound(self, x: torch.Tensor, bound: float = 60.0) -> torch.Tensor:
        """
        Smooth bounding using tanh-based soft clipping.
        Maps values smoothly to approximately [-bound, bound].
        """
        # Scale factor to control how quickly we approach the bounds
        scale = bound / 2.0
        return bound * torch.tanh(x / scale)

    def forward(self, stft_left: torch.Tensor, stft_right: torch.Tensor) -> torch.Tensor:
        """
        Extract robust spatial features from STFT.

        Args:
            stft_left: Complex STFT tensor [B, F, T]
            stft_right: Complex STFT tensor [B, F, T]

        Returns:
            Spatial features [B, 3F, T] containing ILD, cos(IPD), sin(IPD)
        """
        B, F, T = stft_left.shape

        # 1. Compute magnitudes with stability
        mag_left = stft_left.abs().float()
        mag_right = stft_right.abs().float()

        # Add frequency-dependent regularization to prevent extreme values
        freq_reg = torch.linspace(0.1, 0.01, F, device=mag_left.device).view(1, F, 1)
        mag_left = mag_left + freq_reg
        mag_right = mag_right + freq_reg

        # 2. Compute ILD with stable log ratio
        log_ratio = self.stable_log_ratio(mag_left, mag_right)
        ild_raw = 20.0 * log_ratio  # Use 20 instead of 10 for dB SPL

        # Apply learnable gating and scaling
        ild_gated = ild_raw * torch.sigmoid(self.ild_gate).view(1, F, 1)
        ild_scaled = self.ild_scale * ild_gated + self.ild_offset

        # Temporal smoothing (causal)
        ild_smooth = self.ild_smooth(ild_scaled)[:, :, :T]  # Remove padding

        # Smooth bounding instead of hard clamping
        ild = self.smooth_bound(ild_smooth, bound=60.0)

        # 3. Compute IPD with stability
        # Use normalized complex division for phase difference
        eps_complex = self.eps + 0j
        left_norm = stft_left / (stft_left.abs() + self.eps)
        right_norm = stft_right / (stft_right.abs() + self.eps)

        # Phase difference via complex multiplication with conjugate
        phase_diff = left_norm * right_norm.conj()

        # Extract cos and sin directly from complex representation
        cos_ipd_raw = phase_diff.real
        sin_ipd_raw = phase_diff.imag

        # Apply learnable gating
        ipd_gate_expanded = torch.sigmoid(self.ipd_gate).view(1, F, 1)
        cos_ipd_gated = cos_ipd_raw * ipd_gate_expanded
        sin_ipd_gated = sin_ipd_raw * ipd_gate_expanded

        # Stack for joint smoothing
        ipd_stacked = torch.cat([cos_ipd_gated, sin_ipd_gated], dim=1)
        ipd_smooth = self.ipd_smooth(ipd_stacked * self.ipd_scale)[:, :, :T]

        # Re-normalize to unit circle after smoothing
        cos_ipd, sin_ipd = ipd_smooth.chunk(2, dim=1)
        norm = torch.sqrt(cos_ipd ** 2 + sin_ipd ** 2 + self.eps)
        cos_ipd = cos_ipd / norm
        sin_ipd = sin_ipd / norm

        # 4. Stack features
        spatial_features = torch.cat([ild, cos_ipd, sin_ipd], dim=1)  # [B, 3F, T]

        # 5. Adaptive normalization per feature type
        ild_norm = self.adaptive_layer_norm(spatial_features[:, :F], dim=2)
        cos_norm = self.adaptive_layer_norm(spatial_features[:, F:2 * F], dim=2)
        sin_norm = self.adaptive_layer_norm(spatial_features[:, 2 * F:], dim=2)

        spatial_features_norm = torch.cat([ild_norm, cos_norm, sin_norm], dim=1)

        return spatial_features_norm


# Updated compute_spatial_features method for TasNet class
def compute_spatial_features_robust(self, left_waveform: torch.Tensor,
                                    right_waveform: torch.Tensor) -> torch.Tensor:
    """
    Compute spatial features using the robust extractor.

    Args:
        left_waveform: [batch, time]
        right_waveform: [batch, time]

    Returns:
        Spatial features [batch, 3*F, T_frames]
    """
    # Compute STFT
    stft_left = torch.stft(
        left_waveform,
        n_fft=self.analysis_window,
        hop_length=self.analysis_hop,
        win_length=self.analysis_window,
        window=self.stft_window_fn,
        return_complex=True
    )

    stft_right = torch.stft(
        right_waveform,
        n_fft=self.analysis_window,
        hop_length=self.analysis_hop,
        win_length=self.analysis_window,
        window=self.stft_window_fn,
        return_complex=True
    )

    # Use the robust feature extractor
    return self.spatial_feature_extractor(stft_left, stft_right)

