# src/train.py
import hydra
from omegaconf import OmegaConf, DictConfig
from src.config import TasNetConfig
from src.models.tasnet import TasNet


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig):
    """
    Entry point for constructing the TasNet model.

    Steps:
      1. Merge Python dataclass defaults with Hydra-loaded YAML/CLI overrides.
      2. Convert the merged DictConfig into a TasNetConfig dataclass.
      3. Instantiate the TasNet model.
      4. Print the model architecture and effective configuration.
    """
    merged_cfg = OmegaConf.merge(OmegaConf.structured(TasNetConfig), config.model)
    model_config: TasNetConfig = OmegaConf.to_object(merged_cfg)
    model = TasNet(model_config)

    # 4) Display model summary and the final config
    print("\n=== TasNet Model Architecture ===")
    print(model)
    print("\n=== Effective Configuration ===")
    # Print the merged config (including defaults & overrides)
    print(OmegaConf.to_yaml(merged_cfg))


if __name__ == "__main__":
    main()
