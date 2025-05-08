import torch
import torchaudio
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import hydra

from torch.serialization import add_safe_globals
from omegaconf.listconfig import ListConfig
from omegaconf.base import ContainerMetadata
from typing import Any  # For PyTorch >=2.6 compatibility
from src.helpers import select_device

# Allow OmegaConf globals for loading
add_safe_globals([ListConfig, ContainerMetadata, Any])


@hydra.main(version_base="1.3", config_path="../configs", config_name="tasnet_baseline")
def run_inference(cfg: DictConfig):
    # === Paths ===
    CHECKPOINT_PATH = "outputs/model.pt"
    INPUT_AUDIO = "../datasets/static/val/clean/mixture_0000400.wav"
    OUTPUT_AUDIO = "../inference/separated_output_overlap_noisy_randaz.wav"

    # === Setup ===
    device = torch.device(select_device())
    model_cfg = cfg.model_arch

    print("[Config Loaded]")
    print(OmegaConf.to_yaml(cfg))

    # === Build model ===
    model = instantiate(model_cfg).to(device)

    # === Load checkpoint ===
    state = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.eval()

    # === Load and validate input audio ===
    waveform, sample_rate = torchaudio.load(INPUT_AUDIO)  # [channels, time]
    if sample_rate != cfg.model_arch.sample_rate:
        raise ValueError(f"Expected sample rate {cfg.model_arch.sample_rate}, got {sample_rate}")
    if waveform.shape[0] != 2:
        raise ValueError(f"Expected stereo input, got {waveform.shape[0]} channel(s)")

    # === Run model ===
    waveform = waveform.unsqueeze(0).to(device)  # [1, 2, T]
    with torch.no_grad():
        left, right = model(waveform)

    # === Save output ===
    separated = torch.stack([left.squeeze(0), right.squeeze(0)], dim=0).cpu()  # [2, time]
    torchaudio.save(OUTPUT_AUDIO, separated, sample_rate)
    print(f"[✓] Inference complete. Output saved to: {OUTPUT_AUDIO}")


if __name__ == "__main__":
    run_inference()
