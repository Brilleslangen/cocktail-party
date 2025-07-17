import torch
import hydra
from pathlib import Path
import numpy as np
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from tabulate import tabulate

from src.evaluate.eval_metrics import compute_rtf, compute_rtf_streaming
from src.helpers import setup_device_optimizations


@hydra.main(version_base="1.3", config_path="../../configs", config_name="runs/efficiency/rtfs")
def main(cfg: DictConfig):
    models_to_eval = [
        'l-conv-sym', 'l-mamba-sym', 'l-transformer-sym', 'l-liquid-sym',
        's-conv-sym', 's-mamba-sym', 's-transformer-sym', 's-liquid-sym',
        's-conv-asym', 's-mamba-asym', 's-transformer-asym', 's-liquid-asym', 's-mamba-short-asym'
    ]

    device, use_amp, amp_dtype = setup_device_optimizations(cfg)

    results = []

    for model_name in models_to_eval:
        print(f"\n{'=' * 60}\nProcessing model: {model_name}\n{'=' * 60}")

        try:
            model_cfg = hydra.compose(config_name=f"runs/streaming/{model_name}")
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
            model_cfg_dict = OmegaConf.to_container(model_cfg, resolve=True)

            for key, value in model_cfg_dict.items():
                if key in cfg_dict:
                    if isinstance(cfg_dict[key], dict) and isinstance(value, dict):
                        cfg_dict[key].update(value)
                    else:
                        cfg_dict[key] = value
                else:
                    cfg_dict[key] = value

            merged_cfg = OmegaConf.create(cfg_dict)

            model = instantiate(merged_cfg.model_arch, device=device).to(device)
            model.eval()

            rtf_ratio = 156
            rtf_gpu = compute_rtf_streaming(model)
            rtf = rtf_gpu * rtf_ratio
            print(f"    RTF: {rtf:.5f}")
            results.append([model_name, rtf, rtf_gpu])

            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"    ERROR: Failed to process model {model_name}\n    Error: {str(e)}")
            results.append([model_name, "ERROR"])
            continue

    print("\n" + "=" * 60)
    print(tabulate(results, headers=["Model", "RTF", 'RTF_GPU'], floatfmt=".5f", tablefmt="github"))
    print("=" * 60)


if __name__ == "__main__":
    main()
