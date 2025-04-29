# src/config/tasnet_config.py
from dataclasses import dataclass


@dataclass
class VanillaCoderConfig:
    _target_: str
    N: int
    L: int
    stride: int
    causal: bool


@dataclass
class SeparatorConfig:
    _target_: str
    num_stacks: int
    blocks_per_stack: int


@dataclass
class TasNetConfig:
    encoder: VanillaCoderConfig
    # separator: SeparatorConfig
    decoder: VanillaCoderConfig

    feature_dim: int
    stft_stride: int
    mono_fallback: bool
