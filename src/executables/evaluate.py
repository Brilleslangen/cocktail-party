import os
import time
import glob
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np
import hydra
import wandb
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from tabulate import tabulate

from src.data.collate import setup_train_dataloaders
from src.evaluate import compute_mask, count_parameters, count_macs
from src.evaluate.train_metrics import energy_weighted_si_sdr_i
from src.evaluate.eval_metrics import batch_estoi, batch_pesq
from src.helpers import select_device, prettify_macs, prettify_param_count
from src.data.streaming import Streamer
from binaqual import calculate_binaqual


def compute_binaqual(
    est:  torch.Tensor,        # [B, 2, T]  – estimated / degraded
    ref:  torch.Tensor,        # [B, 2, T]  – clean reference
    mask: torch.Tensor | None = None,  # [B, T]  – 1 = valid sample
    sample_rate: int = 16_000
) -> torch.Tensor:
    """
    Return BINAQUAL localisation-similarity (0–1) for each item in a batch.

    Parameters
    ----------
    est, ref :  stereo waveforms shaped [B, 2, T]  (any dtype / device)
    mask     :  optional time-domain validity mask [B, T]  (1=keep, 0=skip)
    sample_rate :  sampling rate in Hz (needs to match the tensors) *

    * The patched loader inside `binaqual` defaults to 16 kHz.
      If you work at 48 kHz (paper default), adjust that default once
      at import time or resample tensors before calling this function.
    """
    if est.shape != ref.shape:
        raise ValueError("est and ref must have identical shapes [B, 2, T]")

    B, C, T = est.shape
    if C != 2:
        raise ValueError("BINAQUAL is defined for stereo signals (C == 2).")

    if mask is not None and mask.shape != (B, T):
        raise ValueError("mask must be [B, T] matching the time dimension.")

    scores = []

    # loop over the batch – calculate_binaqual is light-weight
    for b in range(B):
        # apply mask if given
        idx = slice(None) if mask is None else mask[b].bool().cpu().numpy()

        # (2, T) → (T, 2) → NumPy float32
        ref_np = ref[b, :, idx].permute(1, 0).contiguous().cpu().float().numpy()
        est_np = est[b, :, idx].permute(1, 0).contiguous().cpu().float().numpy()

        # direct tensor input – no temp-files
        nsim_vals, ls = calculate_binaqual(ref_np, est_np)
        # ls is already the localisation-similarity  (product of NSIM_L & NSIM_R)
        scores.append(torch.tensor(ls, dtype=torch.float32))

    return torch.stack(scores)      # [B]


def compute_confusion_rate(est: torch.Tensor, mix: torch.Tensor, ref: torch.Tensor,
                           mask: torch.Tensor, threshold_db: float = 0.0) -> torch.Tensor:
    """
    Compute confusion rate based on SDR difference threshold.
    A sample is confused if separating the interferer gives better SDR than separating the target.

    Args:
        est: [B, 2, T] - estimated stereo signal
        mix: [B, 2, T] - mixture stereo signal
        ref: [B, 2, T] - reference stereo signal (target)
        mask: [B, T] - valid samples mask
        threshold_db: SDR difference threshold for confusion detection

    Returns:
        [B] - binary confusion indicator per sample (1 if confused, 0 if correct)
    """
    B = est.shape[0]
    confused = torch.zeros(B, device=est.device)

    for b in range(B):
        # Extract mono signals for this sample
        est_mono = est[b].mean(0)[mask[b] == 1]
        mix_mono = mix[b].mean(0)[mask[b] == 1]
        ref_mono = ref[b].mean(0)[mask[b] == 1]

        # Compute interferer reference (mix - target)
        interferer_mono = mix_mono - ref_mono

        # Compute SDR with target as reference
        sdr_target = compute_sdr_mono(est_mono, ref_mono)

        # Compute SDR with interferer as reference
        sdr_interferer = compute_sdr_mono(est_mono, interferer_mono)

        # Check if model separated the interferer instead of target
        if sdr_interferer > sdr_target + threshold_db:
            confused[b] = 1.0

    return confused


def compute_sdr_mono(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Compute Signal-to-Distortion Ratio for mono signals.

    Args:
        est: [T] - estimated signal
        ref: [T] - reference signal
        eps: small value for numerical stability

    Returns:
        SDR in dB
    """
    # Find optimal scaling
    alpha = (est * ref).sum() / ((ref ** 2).sum() + eps)

    # Compute SDR
    signal_power = (alpha * ref) ** 2
    distortion_power = (est - alpha * ref) ** 2

    sdr = 10 * torch.log10(signal_power.sum() / (distortion_power.sum() + eps))
    return sdr.item()


def compute_rtf(model: nn.Module, audio_duration: float, batch_size: int = 1,
                num_runs: int = 10, device: torch.device = None) -> float:
    """
    Compute Real-Time Factor (RTF) - processing_time / audio_duration.
    RTF < 1.0 means faster than real-time.

    Args:
        model: the model to evaluate
        audio_duration: duration of audio to process in seconds
        batch_size: batch size for inference
        num_runs: number of runs for averaging
        device: torch device

    Returns:
        RTF value
    """
    if device is None:
        device = next(model.parameters()).device

    sample_rate = model.sample_rate
    num_samples = int(audio_duration * sample_rate)

    # Create dummy input
    dummy_input = torch.randn(batch_size, 2, num_samples, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_input)

    # Time the inference
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    avg_time = np.mean(times)
    rtf = avg_time / audio_duration

    return rtf


def evaluate_model(model: nn.Module, test_loader, device: torch.device,
                   streaming_mode: bool = False) -> Dict[str, Tuple[float, float]]:
    """
    Evaluate model on all metrics and return mean and std for each.

    Returns:
        Dictionary mapping metric_name -> (mean, std)
    """
    model.eval()
    streamer = Streamer(model) if streaming_mode else None

    # Metric accumulators
    all_si_sdr_i = []
    all_estoi = []
    all_pesq = []
    all_binaqual = []
    all_confusion = []

    use_targets_as_input = getattr(model, "use_targets_as_input", False)
    print(f"🎯 Using targets as input: {use_targets_as_input}")

    with torch.no_grad():
        for mix, refs, lengths in tqdm(test_loader, desc="Evaluating", leave=False):
            mix, refs, lengths = mix.to(device), refs.to(device), lengths.to(device)
            model_input = refs if use_targets_as_input else mix


            # Reset state for stateful models
            if hasattr(model, "reset_state"):
                model.reset_state()
            elif hasattr(model, "separator") and hasattr(model.separator, "reset_state"):
                model.separator.reset_state()

            # Forward pass
            if streaming_mode:
                ests, refs_trimmed, lengths_trimmed = streamer.stream_batch(model_input, refs, lengths, trim_warmup=True)
                mix_trimmed = mix[..., streamer.pad_warmup:]
                # Use trimmed versions
                ests, refs, mix = ests, refs_trimmed, mix_trimmed
                lengths = lengths_trimmed
            else:
                ests = model(model_input)

            # Compute mask
            mask = compute_mask(lengths, ests.size(-1), ests.device)

            # Energy-weighted SI-SDRi
            si_sdr_i = energy_weighted_si_sdr_i(ests, mix, refs, mask)
            all_si_sdr_i.extend(si_sdr_i.cpu().numpy())

            # ESTOI (per channel then average)
            estoi_left = batch_estoi(ests[:, 0, :], refs[:, 0, :], model.sample_rate)
            estoi_right = batch_estoi(ests[:, 1, :], refs[:, 1, :], model.sample_rate)
            estoi = (estoi_left + estoi_right) / 2
            all_estoi.extend(estoi.cpu().numpy())

            # PESQ (per channel then average)
            pesq_left = batch_pesq(ests[:, 0, :], refs[:, 0, :], model.sample_rate, mode='wb')
            pesq_right = batch_pesq(ests[:, 1, :], refs[:, 1, :], model.sample_rate, mode='wb')
            pesq = (pesq_left + pesq_right) / 2
            all_pesq.extend(pesq.cpu().numpy())

            # BINAQUAL
            binaqual = compute_binaqual(ests, refs, mask, sample_rate=model.sample_rate)
            all_binaqual.extend(binaqual.cpu().numpy())

            # Confusion Rate
            confusion = compute_confusion_rate(ests, mix, refs, mask)
            all_confusion.extend(confusion.cpu().numpy())

    # Compute RTF
    rtf = compute_rtf(model, audio_duration=1.0, device=device)

    # Compute mean and std for each metric
    results = {
        'EW-SI-SDRi (dB)': (np.mean(all_si_sdr_i), np.std(all_si_sdr_i)),
        'ESTOI': (np.mean(all_estoi), np.std(all_estoi)),
        'PESQ': (np.mean(all_pesq), np.std(all_pesq)),
        'BINAQUAL ': (np.mean(all_binaqual), np.std(all_binaqual)),
        'Confusion Rate (%)': (np.mean(all_confusion) * 100, np.std(all_confusion) * 100),
        'RTF': (rtf, 0.0)  # RTF doesn't have std
    }

    return results


def format_results_table(results: Dict[str, Tuple[float, float]],
                         param_count: str, macs_str: str) -> Tuple[str, str]:
    """
    Format results into two tables: one for values, one for standard deviations.
    """
    # Add model capacity metrics (no std)
    results['Parameters'] = (param_count, 'N/A')
    results['MACs/s'] = (macs_str, 'N/A')

    # Prepare data for tables
    metrics = []
    values = []
    stds = []

    for metric, (value, std) in results.items():
        metrics.append(metric)
        if isinstance(value, str):
            values.append(value)
            stds.append(std)
        else:
            if metric == 'Confusion Rate (%)':
                values.append(f"{value:.1f}")
                stds.append(f"{std:.1f}")
            elif metric == 'RTF':
                values.append(f"{value:.3f}")
                stds.append("N/A")
            elif metric in ['EW-SI-SDRi (dB)', 'ILD-RMSE (dB)']:
                values.append(f"{value:.2f}")
                stds.append(f"{std:.2f}")
            else:  # ESTOI, PESQ
                values.append(f"{value:.3f}")
                stds.append(f"{std:.3f}")

    # Create tables
    value_table = tabulate([values], headers=metrics, tablefmt='grid')
    std_table = tabulate([stds], headers=metrics, tablefmt='grid')

    return value_table, std_table


def save_evaluation_results(experiment_name: str, model_name: str,
                            value_table: str, std_table: str):
    """
    Save or update evaluation results in the experiment file.
    """
    output_dir = Path("evaluation_outputs")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"{experiment_name}.txt"

    # Read existing content if file exists
    existing_content = ""
    if output_file.exists():
        with open(output_file, 'r') as f:
            existing_content = f.read()

    # Check if this model already exists in the file
    model_header = f"=== Model: {model_name} ==="

    if model_header in existing_content:
        # Find and replace the existing section
        lines = existing_content.split('\n')
        start_idx = None
        end_idx = None

        for i, line in enumerate(lines):
            if line.strip() == model_header:
                start_idx = i
            elif start_idx is not None and line.strip().startswith("=== Model:"):
                end_idx = i
                break

        if end_idx is None:
            end_idx = len(lines)

        # Create new section
        new_section = [
            model_header,
            "",
            "Values:",
            value_table,
            "",
            "Standard Deviations:",
            std_table,
            ""
        ]

        # Replace the section
        new_lines = lines[:start_idx] + new_section + lines[end_idx:]
        new_content = '\n'.join(new_lines)
    else:
        # Append new model results
        new_section = f"""
{model_header}

Values:
{value_table}

Standard Deviations:
{std_table}

"""
        new_content = existing_content + new_section

    # Write updated content
    with open(output_file, 'w') as f:
        f.write(new_content.strip() + '\n')

    print(f"Results saved to {output_file}")


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Main evaluation script that downloads model from W&B and evaluates it.
    """
    device = select_device()

    # Parse arguments
    if not cfg.model_artifact:
        raise ValueError("Please provide model_artifact name via --model_artifact=<artifact_name>")

    # Initialize W&B
    if cfg.wandb.enabled:
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            tags=cfg.wandb.tags,
            job_type='evaluation',
            name=cfg.model_artifact,
            reinit='finish_previous'
        )

        artifact = run.use_artifact(cfg.model_artifact, type="model")
        os.makedirs(cfg.training.model_save_dir, exist_ok=True)
        artifact_dir = artifact.download(root=cfg.training.model_save_dir)
        pt_files = glob.glob(os.path.join(artifact_dir, "*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in artifact at {artifact_dir}")
        checkpoint_path = pt_files[0]
        print(f"📥 Found checkpoint from: {checkpoint_path}")
    else:
        # Use local checkpoint
        checkpoint_path = f"{cfg.training.model_save_dir}/{cfg.model_artifact}.pt"
        print(f"[LOCAL] Using checkpoint at {checkpoint_path}")

    print(f"🔍 Evaluating model: {cfg.model_artifact}")
    print(f"📁 Experiment: {cfg.group}")

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config from checkpoint
    if 'cfg' in state:
        artifact_cfg = state['cfg']
    else:
        raise ValueError("❌ Checkpoint does not contain model configuration.")

    # Build model
    model = instantiate(artifact_cfg['model_arch']).to(device)

    # Load weights
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        raise ValueError("❌ Checkpoint does not contain model_state.")

    model.eval()
    print("✅ Model loaded successfully")

    # Get model statistics
    param_count = count_parameters(model)
    macs = count_macs(model)
    pretty_params = prettify_param_count(param_count)
    pretty_macs = prettify_macs(macs)

    print(f"📊 Model Statistics:")
    print(f"   Parameters: {pretty_params}")
    print(f"   MACs/s: {pretty_macs}")

    # Setup test dataloader
    _, test_loader = setup_train_dataloaders(cfg)

    # Check streaming mode
    streaming_mode = getattr(model, "streaming_mode", False)
    print(f"🔄 Streaming mode: {streaming_mode}")

    # Run evaluation
    print("\n🧪 Running evaluation...")
    results = evaluate_model(model, test_loader, device, streaming_mode)

    # Format results
    value_table, std_table = format_results_table(results, pretty_params, pretty_macs)

    # Print results
    print("\n📊 Evaluation Results:")
    print("\nValues:")
    print(value_table)
    print("\nStandard Deviations:")
    print(std_table)

    # Save results
    save_evaluation_results(cfg.group, cfg.model_artifact, value_table, std_table)

    # Log to W&B
    if cfg.wandb.enabled:
        wandb_results = {}
        for metric, (value, std) in results.items():
            if isinstance(value, (int, float)):
                wandb_results[f"eval/{metric.replace(' ', '_').replace('(', '').replace(')', '')}"] = value
                if isinstance(std, (int, float)) and std != 0.0:
                    wandb_results[f"eval/{metric.replace(' ', '_').replace('(', '').replace(')', '')}_std"] = std

        wandb.log(wandb_results)
        wandb.finish()

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()