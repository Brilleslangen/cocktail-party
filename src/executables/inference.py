import os
import glob
import torch
import torchaudio
import hydra
import wandb
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch import nn
from src.helpers import select_device

# Allow OmegaConf to handle PyTorch types if needed
from torch.serialization import add_safe_globals
from omegaconf.listconfig import ListConfig
from omegaconf.base import ContainerMetadata
from typing import Any
# add_safe_globals([ListConfig, ContainerMetadata, Any]) What is this?


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def run_inference(cfg: DictConfig):
    device = torch.device(select_device())

    # Determine checkpoint path via W&B Artifact or local file
    if cfg.wandb.enabled:
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            job_type="inference",
            name=cfg.name,
            reinit='finish_previous'
        )
        artifact = run.use_artifact(cfg.inference.artifact, type="model")
        os.makedirs(cfg.training.model_save_dir, exist_ok=True)
        artifact_dir = artifact.download(root=cfg.training.model_save_dir)
        pt_files = glob.glob(os.path.join(artifact_dir, "*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in artifact at {artifact_dir}")
        checkpoint_path = pt_files[0]
        print(f"[W&B] Downloaded checkpoint from artifact to {checkpoint_path}")
    else:
        checkpoint_path = cfg.inference.local_checkpoint  # e.g. "outputs/{run_name}.pt"
        print(f"[LOCAL] Using checkpoint at {checkpoint_path}")

    # Build model and load weights
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = OmegaConf.to_object(cfg['model_arch'])
    print(model_cfg)
    model = instantiate(model_cfg).to(device)
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.eval()
    print(f"[✓] Loaded model weights from {checkpoint_path}")

    # Load and validate input audio
    waveform, sample_rate = torchaudio.load(cfg.inference.input_audio)
    if sample_rate != cfg.model_arch.sample_rate:
        raise ValueError(f"Expected sample rate {cfg.model_arch.sample_rate}, got {sample_rate}")
    if waveform.shape[0] != 2:
        raise ValueError(f"Expected stereo input, got {waveform.shape[0]} channel(s)")

    # Run inference
    waveform = waveform.unsqueeze(0).to(device)  # [1, 2, T]
    with torch.no_grad():
        left, right = model(waveform)

    # Save output
    separated = torch.stack([left.squeeze(0), right.squeeze(0)], dim=0).cpu()  # [2, time]
    os.makedirs(os.path.dirname(cfg.inference.output_audio), exist_ok=True)
    torchaudio.save(cfg.inference.output_audio, separated, sample_rate)
    print(f"[✓] Inference complete. Output saved to: {cfg.inference.output_audio}")

    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    run_inference()
