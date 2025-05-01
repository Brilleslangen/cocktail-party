import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
import torch


@hydra.main(config_path="../configs", config_name="tasnet_baseline")
def main(cfg: DictConfig):
    model = instantiate(cfg.model_arch)

    print("\n=== TasNet Model Architecture ===")
    print(model)
    print("\n=== Effective Configuration ===")
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
