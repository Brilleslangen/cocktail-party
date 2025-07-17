import os
import csv
import torch
import hydra
import numpy as np
from pathlib import Path

import wandb
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from src.evaluate import count_macs
from src.helpers import prettify_macs, setup_device_optimizations


@hydra.main(version_base="1.3", config_path="../../configs", config_name="runs/efficiency/compute_macs_context")
def main(cfg: DictConfig):
    """
    Compute MACs for different context sizes for large streaming models.
    """
    # Models to evaluate
    models_to_eval = ['l-conv-sym', 'l-mamba-sym', 'l-transformer-sym', 'l-liquid-sym',
                      's-conv-sym', 's-mamba-sym', 's-transformer-sym', 's-liquid-sym',
                      's-conv-asym', 's-mamba-asym', 's-transformer-asym', 's-liquid-asym']

    # Context sizes to test (in ms)
    context_sizes_ms = np.arange(1000, 50, -50).tolist()  # 1000ms down to 50ms in 50ms steps

    # Setup output directory and CSV file
    output_dir = Path("evaluation_outputs")
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "macs_vs_context_size.csv"

    # Setup device
    device, use_amp, amp_dtype = setup_device_optimizations(cfg)

    # Open CSV file for writing
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['model', 'context_size_ms', 'macs', 'gmacs', 'pretty_macs']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        writer.writerow({
            'model': 'test',
            'context_size_ms': 0,
            'macs': 0,
            'gmacs': 0,
            'pretty_macs': '0'
        })

        # Process each model
        for model_name in models_to_eval:
            print(f"\n{'=' * 60}")
            print(f"Processing model: {model_name}")
            print(f"{'=' * 60}")

            # Load the specific streaming config
            model_cfg = hydra.compose(config_name=f"runs/streaming/{model_name}")

            # Merge with base config
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
            model_cfg_dict = OmegaConf.to_container(model_cfg, resolve=True)

            # Update the base config with model-specific settings
            for key, value in model_cfg_dict.items():
                if key in cfg_dict:
                    if isinstance(cfg_dict[key], dict) and isinstance(value, dict):
                        cfg_dict[key].update(value)
                    else:
                        cfg_dict[key] = value
                else:
                    cfg_dict[key] = value

            # Convert back to DictConfig
            merged_cfg = OmegaConf.create(cfg_dict)

            # Process each context size
            for context_ms in context_sizes_ms:
                print(f"\n  Context size: {context_ms}ms")

                # Create a copy of the config to modify
                test_cfg = OmegaConf.create(OmegaConf.to_container(merged_cfg, resolve=True))

                # Override context size
                test_cfg.model_arch.separator.context_size_ms = float(context_ms)
                test_cfg.model_arch.stream_chunk_size_ms = min(4.0, float(context_ms))  # Ensure chunk <= context

                # Also update the input size based on context
                test_cfg.model_arch.streaming_mode = True

                try:
                    # Build model
                    model = instantiate(test_cfg.model_arch, device=device).to(device)
                    model.eval()

                    # Count MACs
                    macs = count_macs(model, seconds=1.0)
                    gmacs = macs / 1e9
                    pretty_macs_str = prettify_macs(macs)

                    print(f"    MACs: {macs:,}")
                    print(f"    GMACs: {gmacs:.3f}")
                    print(f"    Pretty: {pretty_macs_str}")

                    # Write to CSV
                    print(f"    Writing results to CSV...", pretty_macs_str)
                    writer.writerow({
                        'model': model_name,
                        'context_size_ms': context_ms,
                        'macs': macs,
                        'gmacs': gmacs,
                        'pretty_macs': pretty_macs_str
                    })

                    # Clean up model to free memory
                    del model
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"    ERROR: Failed to process context size {context_ms}ms")
                    print(f"    Error: {str(e)}")
                    continue

    with wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, job_type="publish-csv") as run:
        artifact = wandb.Artifact(
            name="macs_vs_context",
            type="csv",
            description="macs_vs_context_size.csv - MACs vs Context Size for Large Streaming Models"
        )

        csv_path_abs = os.path.abspath(csv_path)
        assert os.path.isfile(csv_path_abs), f"File does not exist: {csv_path_abs}"

        artifact.add_file(csv_path_abs)
        run.log_artifact(artifact)
        run.finish()

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {csv_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
