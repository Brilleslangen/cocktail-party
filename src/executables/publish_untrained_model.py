# publish_untrained_models.py
import os, glob, time, platform
import hydra, wandb, torch
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.utils import instantiate

from src.helpers import select_device, prettify_macs, prettify_param_count
from src.evaluate import count_parameters, count_macs

######################################################################
# Utility
######################################################################


def build_and_publish(cfg: DictConfig, model_cfg: DictConfig, tag_suffix: str = ""):
    """
    Instantiates `model_cfg`, computes stats, writes a tiny checkpoint,
    and publishes it as a W&B model artifact.
    """
    device = select_device()
    model = instantiate(model_cfg).to(device)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    try:
        param_count = count_parameters(model)
        macs = count_macs(model)
        pretty_params = prettify_param_count(param_count)
        pretty_macs = prettify_macs(macs)
    except RuntimeError as e:
        print(f"⚠️  Error counting parameters or MACs: {e}")
        param_count = "-"
        macs = "-"
        pretty_params = "-"
        pretty_macs = "-"

    run_name = f"{cfg.name}_{pretty_params}"

    # ------------------------------------------------------------------
    # W&B run
    # ------------------------------------------------------------------
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        tags=cfg.wandb.tags,
        job_type="publish_untrained",
        name=f"publish_{run_name}",
        reinit="finish_previous",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    run.summary["model/param_count"] = pretty_params
    run.summary["model/macs_per_second"] = pretty_macs
    run.summary["model/macs_pretty"] = pretty_macs
    run.summary["trained"] = False

    # ------------------------------------------------------------------
    # Save checkpoint (tiny – may even be empty)
    # ------------------------------------------------------------------
    ckpt_dir = cfg.training.model_save_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{run_name}.pt")

    torch.save(
        {
            "cfg": {"model_arch": OmegaConf.to_container(model_cfg, resolve=True)},
            "model_state": model.state_dict(),  # {} for oracle models
        },
        ckpt_path,
    )

    # ------------------------------------------------------------------
    # Publish artifact
    # ------------------------------------------------------------------
    art = wandb.Artifact(run_name, type="model")
    art.add_file(ckpt_path)
    art.metadata.update(
        {
            "param_count": param_count,
            "macs": macs,
            "trained": False,
        }
    )
    run.log_artifact(art)
    run.finish()

    print(f"🚀 Published '{run_name}'  |  params: {pretty_params}, macs: {pretty_macs}")


######################################################################
# Hydra entry-point
######################################################################


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """Publish **exactly one** untrained model: whatever `cfg.model_arch` is."""
    build_and_publish(cfg)


if __name__ == "__main__":
    main()
