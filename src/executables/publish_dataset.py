import hydra
import wandb
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    with wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, job_type="publish-dataset") as run:
        artifact = wandb.Artifact(
            name="static-2-spk-noise",
            type="dataset",
            description="Static 2-speaker binaural mixtures with clean references (train/val/test splits)"
        )

        # Add dataset directory with splits clearly marked
        artifact.add_dir("../datasets/static")

        run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    main()
