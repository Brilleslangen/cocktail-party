import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig


def test_model_pass(cfg):
    model = instantiate(cfg.model_arch)

    print("\n=== TasNet Model Architecture ===")
    print(model)
    # print("\n=== Effective Configuration ===")
    # print(OmegaConf.to_yaml(cfg))

    # batch size 2, stereo channels, 32000 samples
    batch, channels, time = 2, 2, 32000
    mixture = torch.rand(batch, channels, time)

    # forward returns a tuple (left, right)
    left_out, right_out = model(mixture)

    assert left_out.shape == (batch, time), f"Expected (B, T), got {left_out.shape}"
    assert right_out.shape == (batch, time), f"Expected (B, T), got {right_out.shape}"

    print(f"Test passed! Output shapes: left={left_out.shape}, right={right_out.shape}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="tasnet_baseline")
def main(cfg: DictConfig):
    test_model_pass(cfg)


if __name__ == "__main__":
    main()
