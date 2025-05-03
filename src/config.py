from dataclasses import dataclass


@dataclass
class VanillaCoderConfig:
    _target_: str       # Hydra target path for the encoder/decoder class
    num_filters: int    # number of learned basis filters
    filter_length: int  # filter length in samples
    stride: int         # convolution stride / hop size in samples
    causal: bool        # whether to apply causal convolution


@dataclass
class SeparatorConfig:
    _target_: str           # Hydra target path for the separator class
    num_stacks: int         # number of repeated TCN stacks
    blocks_per_stack: int   # number of convolutional blocks per stack


@dataclass
class TasNetConfig:
    encoder: VanillaCoderConfig
    separator: SeparatorConfig
    decoder: VanillaCoderConfig

    feature_dim: int         # STFT window length in samples
    stft_stride: int         # STFT hop length in samples
    mono_fallback: bool      # fallback to mono if spatial inputs absent
