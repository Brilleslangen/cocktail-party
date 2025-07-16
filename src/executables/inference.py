import os
import torch
import torchaudio
import hydra
import wandb
from omegaconf import DictConfig
from hydra.utils import instantiate
from src.helpers import select_device


@hydra.main(version_base="1.3", config_path="../../configs", config_name="inference/default")
def run_inference(cfg: DictConfig):
    device = select_device()

    model_name = cfg.model_artifact.split(':')[0] if ':' in cfg.model_artifact else cfg.model_artifact

    # Initialize W&B if enabled
    if cfg.wandb.enabled:
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            job_type="inference",
            name=f"{model_name}_inference",
            tags=cfg.wandb.tags,
            reinit='finish_previous'
        )

        artifact = run.use_artifact(cfg.model_artifact, type="model")
        model_dir = artifact.download(root=cfg.training.model_save_dir)
        artifact_path = os.path.join(model_dir, f"{model_name}.pt")

        print(f"[W&B] Checkpoint downloaded: {artifact_path}")
    else:
        artifact_path = cfg.inference.local_checkpoint
        print(f"[LOCAL] Using checkpoint: {artifact_path}")

    state = torch.load(artifact_path, map_location=device, weights_only=False)

    # Extract model config from checkpoint
    if 'cfg' not in state:
        raise ValueError("Checkpoint does not contain model configuration.")

    artifact_cfg = state['cfg']
    model = instantiate(artifact_cfg['model_arch'], device=device).to(device)

    if "model_state" not in state:
        raise ValueError("Checkpoint does not contain model_state.")

    model.load_state_dict(state["model_state"])
    model.eval()
    print("✅ Model loaded successfully")

    # Load input audio
    dataset_dir = os.path.join("artifacts", "static-2-spk-noise12")
    input_audio_path = os.path.join(dataset_dir, cfg.inference.input_audio_file)
    waveform, sample_rate = torchaudio.load(input_audio_path)
    if sample_rate != artifact_cfg['model_arch']['sample_rate']:
        raise ValueError(f"Expected sample rate {artifact_cfg['model_arch']['sample_rate']}, got {sample_rate}")
    if waveform.shape[0] != 2:
        raise ValueError(f"Expected stereo input, got {waveform.shape[0]} channel(s)")

    # Run inference
    waveform = waveform.unsqueeze(0).to(device)
    with torch.no_grad():
        ests = model(waveform)

    # Save output
    separated = ests.squeeze(0).cpu()
    output_path = os.path.join(cfg.inference.output_dir, f"{model_name}_{cfg.inference.output_audio_file}")
    os.makedirs(cfg.inference.output_dir, exist_ok=True)
    torchaudio.save(output_path, separated, sample_rate)

    print(f"[✓] Output saved: {output_path}")

    # Push output to W&B as artifact
    if cfg.wandb.enabled and cfg.inference.save_to_wandb:
        output_artifact = wandb.Artifact(name=f"{model_name}_output", type="audio")
        output_artifact.add_file(output_path)
        run.log_artifact(output_artifact)

    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    run_inference()
