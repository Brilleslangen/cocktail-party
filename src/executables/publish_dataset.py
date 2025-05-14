import hydra
import wandb
from omegaconf import DictConfig
import os



@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    with wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, job_type="publish-dataset") as run:
        artifact = wandb.Artifact(
            name="static-2-spk-noise",
            type="dataset",
            description="Static 2-speaker binaural mixtures with clean references (train/val/test splits)"
        )

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        dataset_path = os.path.join(project_root, "datasets/static")

        assert os.path.isdir(dataset_path), f"Path does not exist: {dataset_path}"

        # Add dataset directory with splits clearly marked
        artifact.add_dir(dataset_path)

        run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    main()
