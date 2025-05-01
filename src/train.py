import hydra
import torch
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate


@hydra.main(version_base="1.3", config_path="../configs", config_name="tasnet_baseline")
def main(cfg: DictConfig):
    model = instantiate(cfg.model_arch)

    print("\n=== TasNet Model Architecture ===")
    print(model)
    print("\n=== Effective Configuration ===")
    print(OmegaConf.to_yaml(cfg))

    test_model(model)


def test_model(model):
    x = torch.rand(2, 32000)
    x = model(x)
    s1 = x[0]
    print('Test output shape:', s1.shape)


if __name__ == "__main__":
    main()
