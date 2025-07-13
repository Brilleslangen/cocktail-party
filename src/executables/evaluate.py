import os
from pathlib import Path
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import hydra
import wandb
from tqdm import tqdm
from omegaconf import DictConfig
from hydra.utils import instantiate
from tabulate import tabulate

from src.data.collate import setup_test_dataloader
from src.evaluate import count_parameters, count_macs
from src.evaluate.eval_metrics import compute_evaluation_metrics, compute_rtf
from src.helpers import prettify_macs, prettify_param_count, setup_device_optimizations
from src.data.streaming import Streamer


def evaluate_model(model: nn.Module, test_loader, streaming_mode: bool, device: torch.device,
                   use_amp: bool, amp_dtype: torch.dtype, ignore_confused: bool = False
                   ) -> Tuple[Dict[str, Tuple[float, float, float]], Dict[str, List[float]]]:
    """
    Evaluate model on all metrics and return mean, std, median for each metric,
    as well as individual values for violin plots.

    Returns:
        (summary_results, metric_values)
        - summary_results: Dict[metric_name -> (mean, std, median)]
        - metric_values: Dict[metric_name -> List of values]
    """
    model.eval()
    streamer = Streamer(model) if streaming_mode else None

    metric_values = {
        "mc_si_sdr": [],
        "mc_si_sdr_i": [],
        "ew_si_sdr": [],
        "ew_si_sdr_i": [],
        "ew_estoi": [],
        "ew_pesq": [],
        "binaqual": [],
        "confusion_rate": []
    }

    use_targets_as_input = getattr(model, "use_targets_as_input", False)
    print(f"🎯 Using targets as input: {use_targets_as_input}")

    with torch.no_grad():
        for mix, refs, lengths in tqdm(test_loader, desc="Evaluating", leave=False):
            if not streaming_mode:
                mix, refs, lengths = mix.to(device), refs.to(device), lengths.to(device)

            model_input = refs if use_targets_as_input else mix
            B, C, T = mix.shape

            if hasattr(model, "reset_state"):
                model.reset_state(batch_size=B, chunk_len=T, dtype=amp_dtype if use_amp else torch.float32)

            if use_amp and device.type == "cuda":
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    if streaming_mode:
                        ests, refs, lengths = streamer.stream_batch(model_input, refs, lengths, trim_warmup=True)
                        ests, refs, lengths = ests.to(device), refs.to(device), lengths.to(device)
                        mix = mix[..., streamer.pad_warmup:]
                    else:
                        ests = model(model_input)
            else:
                if streaming_mode:
                    ests, refs, lengths = streamer.stream_batch(model_input, refs, lengths, trim_warmup=True)
                    mix = mix[..., streamer.pad_warmup:]
                else:
                    ests = model(model_input)

            metrics = compute_evaluation_metrics(ests, mix, refs, model.sample_rate, lengths)

            if ignore_confused:
                confusion_rate = metrics.get('confusion_rate', torch.zeros_like(next(iter(metrics.values()))))
                mask = confusion_rate != 1.0  # True for rows to keep
                for key in metrics.keys():
                    if isinstance(metrics[key], torch.Tensor) and metrics[key].shape == confusion_rate.shape:
                        metrics[key] = metrics[key][mask]

            for metric_name, values in metrics.items():
                metric_values[metric_name].extend(values.cpu().numpy().tolist())

    # Compute summary statistics
    summary_results = {}
    for metric_name, values in metric_values.items():
        mean = np.mean(values)
        std = np.std(values)
        median = np.median(values)

        if metric_name == 'mc_si_sdr':
            display_name = 'MC-SI-SDR (dB)'
        elif metric_name == 'mc_si_sdr_i':
            display_name = 'MC-SI-SDRi (dB)'
        elif metric_name == 'ew_si_sdr':
            display_name = 'EW-SI-SDR (dB)'
        elif metric_name == 'ew_si_sdr_i':
            display_name = 'EW-SI-SDRi (dB)'
        elif metric_name == 'ew_estoi':
            display_name = 'ESTOI'
        elif metric_name == 'ew_pesq':
            display_name = 'PESQ'
        elif metric_name == 'binaqual':
            display_name = 'BINAQUAL'
        elif metric_name == 'confusion_rate':
            display_name = 'Confusion Rate (%)'
            mean *= 100
            std *= 100
            median *= 100

        summary_results[display_name] = (mean, std, median)

    rtf = compute_rtf(model, audio_duration=1.0, device=device)
    summary_results['RTF'] = (rtf, 0.0, 0.0)

    return summary_results, metric_values



def format_results_table(results: Dict[str, Tuple[float, float, float]],
                         param_count: str, macs_str: str) -> Tuple[str, str, str]:
    """
    Format results into three tables: one for means, one for standard deviations, and one for medians.
    """
    # Add model capacity metrics (no std or median)
    results['Parameters'] = (param_count, 'N/A', 'N/A')
    results['MACs/s'] = (macs_str, 'N/A', 'N/A')

    # Prepare data for tables
    metrics = []
    means = []
    stds = []
    medians = []

    for metric, (mean, std, median) in results.items():
        metrics.append(metric)
        if isinstance(mean, str):
            means.append(mean)
            stds.append(std)
            medians.append(median)
        else:
            if metric == 'Confusion Rate (%)':
                means.append(f"{mean:.1f}")
                stds.append(f"{std:.1f}")
                medians.append(f"{median:.1f}")
            elif metric == 'RTF':
                means.append(f"{mean:.3f}")
                stds.append("N/A")
                medians.append(f"{median:.3f}")
            elif metric in ['EW-SI-SDRi (dB)']:
                means.append(f"{mean:.2f}")
                stds.append(f"{std:.2f}")
                medians.append(f"{median:.2f}")
            else:  # ESTOI, PESQ, BINAQUAL
                means.append(f"{mean:.3f}")
                stds.append(f"{std:.3f}")
                medians.append(f"{median:.3f}")

    # Create tables
    mean_table = tabulate([means], headers=metrics, tablefmt='grid')
    std_table = tabulate([stds], headers=metrics, tablefmt='grid')
    median_table = tabulate([medians], headers=metrics, tablefmt='grid')

    return mean_table, std_table, median_table



def save_evaluation_results(experiment_name: str, model_name: str,
                            value_table: str, median_table: str, std_table: str):
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
            "",
            "Medians:",
            median_table,
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

Medians:
{median_table}

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
    device, use_amp, amp_dtype = setup_device_optimizations(cfg)
    amp_dtype = torch.float32
    model_name = cfg.model_artifact.split(':')[0] if ':' in cfg.model_artifact else cfg.model_artifact
    print(type(model_name))

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
            name=model_name,
            reinit='finish_previous'
        )

        artifact = run.use_artifact(cfg.model_artifact, type="model")
        os.makedirs(cfg.training.model_save_dir, exist_ok=True)
        artifact_dir = artifact.download(root=cfg.training.model_save_dir)
        artifact_path = artifact_dir + f"/{model_name}.pt"
        print(f"📥 Found checkpoint from: {artifact_path}")
    else:
        # Use local checkpoint
        artifact_path = f"{cfg.training.model_save_dir}/{model_name}.pt"
        print(f"[LOCAL] Using checkpoint at {artifact_path}")

    print(f"🔍 Evaluating model: {model_name}")
    print(f"📁 Experiment: {cfg.group}")

    state = torch.load(artifact_path, map_location=device, weights_only=False)

    # Extract config from checkpoint
    if 'cfg' in state:
        artifact_cfg = state['cfg']
    else:
        raise ValueError("❌ Checkpoint does not contain model configuration.")

    # Build model
    model = instantiate(artifact_cfg['model_arch'], device=device).to(device)

    # Load weights
    if "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        raise ValueError("❌ Checkpoint does not contain model_state.")

    model.eval()
    print("✅ Model loaded successfully")

    # Get model statistics
    try:
        param_count = count_parameters(model)
        macs = count_macs(model)
        pretty_params = prettify_param_count(param_count)
        pretty_macs = prettify_macs(macs)
    except Exception as e:
        print(f"⚠️  Error counting parameters or MACs: {e}")
        pretty_params = "-"
        pretty_macs = "-"

    print(f"📊 Model Statistics:")
    print(f"   Parameters: {pretty_params}")
    print(f"   MACs/s: {pretty_macs}")

    # Setup test dataloader
    test_loader = setup_test_dataloader(cfg)

    # Check streaming mode
    streaming_mode = getattr(model, "streaming_mode", False)
    print(f"🔄 Streaming mode: {streaming_mode}")

    # Run evaluation
    print("\n🧪 Running evaluation...")
    if hasattr(cfg, 'num_runs') and cfg.num_runs > 1:
        print(f"🔁 Running {cfg.num_runs} runs for stability...")
        if hasattr(cfg, 'ignore_confused'):
            run_results, metric_values = [evaluate_model(model, test_loader, streaming_mode, device, use_amp, amp_dtype,
                                          ignore_confused=cfg.ignore_confused) for _ in range(cfg.num_runs)]
        else:
            run_results, metric_values = [evaluate_model(model, test_loader, streaming_mode, device, use_amp, amp_dtype,
                                          ignore_confused=False) for _ in range(cfg.num_runs)]
        # Average results across runs
        results = {}
        for key in run_results[0].keys():
            means = [torch.as_tensor(result[key][0]).float() for result in run_results]
            medians = [torch.as_tensor(result[key][1]).float() for result in run_results]
            stds = [torch.as_tensor(result[key][2]).float() for result in run_results]
            mean = torch.stack(means).mean()
            median = torch.stack(medians).mean()
            std = torch.stack(stds).mean()
            results[key] = (mean.item(), median.item(), std.item())
    elif hasattr(cfg, 'ignore_confused'):
        results, metric_values = evaluate_model(model, test_loader, streaming_mode, device, use_amp, amp_dtype, ignore_confused=cfg.ignore_confused)
    else:
        results, metric_values = evaluate_model(model, test_loader, streaming_mode, device, use_amp, amp_dtype, ignore_confused=False)
    # Format results
    value_table, std_table, median_table = format_results_table(results, pretty_params, pretty_macs)

    # Print results
    print("\n📊 Evaluation Results:")
    print("Ignoring confused: ", cfg.ignore_confused) if hasattr(cfg, 'ignore_confused') else None
    print("\nValues:")
    print(value_table)
    print("\nMedians:")
    print(median_table)
    print("\nStandard Deviations:")
    print(std_table)

    # Save results
    save_evaluation_results(cfg.group, model_name, value_table, median_table, std_table)

    # Save full metrics df for plot
    df_metrics = pd.DataFrame(metric_values)
    output_path = f"outputs/{model_name}_metrics_confused_{cfg.ignore_confused}.csv" if hasattr(cfg, 'ignore_confused') else f"outputs/{model_name}_metrics_confused_false.csv"
    df_metrics.to_csv(output_path, index=False)

    if cfg.wandb.enabled:
        output_artifact = wandb.Artifact(name=f"{model_name}_metric_log", type="csv")
        output_artifact.add_file(output_path)
        run.log_artifact(output_artifact)

    # Log to W&B
    if cfg.wandb.enabled:
        wandb_results = {}
        for metric, (value, std) in results.items():
            if isinstance(value, (int, float)):
                metric_key = metric.replace(' ', '_').replace('(', '').replace(')', '').replace('%', '')
                wandb_results[f"eval/{metric_key}"] = value
                if isinstance(std, (int, float)) and std != 0.0:
                    wandb_results[f"eval/{metric_key}_std"] = std

        wandb.log(wandb_results)
        wandb.finish()

    print("\n✅ Evaluation complete!\n\n")


if __name__ == "__main__":
    main()
