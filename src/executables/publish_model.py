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


def build_and_publish(cfg: DictConfig, artifact_name: str = None):
    """
    Publishes a (possibly pre-trained) model as a W&B model artifact.
    If `artifact_name` is provided, loads model and config from artifact path.
    Otherwise, builds a new model as specified in cfg.
    """
    device = select_device()
    model, model_cfg = None, None

    input_cfg = cfg

    if artifact_name is not None:
        # ----------------------------------------
        # Load model checkpoint from artifact
        # ----------------------------------------
        artifact_dir = './artifacts'
        artifact_path = os.path.join(artifact_dir, artifact_name)
        state = torch.load(artifact_path, map_location=device, weights_only=False)
        if 'cfg' not in state or 'model_state' not in state:
            raise ValueError("❌ Checkpoint missing 'cfg' or 'model_state'.")
        cfg = OmegaConf.create(state['cfg'])

        print(cfg)
        model_cfg = state['cfg']['model_arch']
        model = instantiate(model_cfg, device=device).to(device)
        model.load_state_dict(state['model_state'])
    else:
        # ----------------------------------------
        # Build new model from config
        # ----------------------------------------
        model_cfg = cfg.model_arch
        model = instantiate(model_cfg).to(device)

    # ----------------------------------------
    # Stats
    # ----------------------------------------
    try:
        param_count = count_parameters(model)
        macs = count_macs(model)
        pretty_params = prettify_param_count(param_count)
        pretty_macs = prettify_macs(macs)
    except Exception as e:
        print(f"⚠️  Error counting parameters or MACs: {e}")
        param_count = "-"
        macs = "-"
        pretty_params = "-"
        pretty_macs = "-"

    try:
        run_name = f"{cfg.name}_{pretty_params}" if pretty_params != "-" else cfg.name
    except Exception as e:
        run_name = artifact_name.split('.')[0]

    # ----------------------------------------
    # W&B run
    # ----------------------------------------
    run = wandb.init(
        project=input_cfg.wandb.project,
        entity=input_cfg.wandb.entity,
        group=input_cfg.wandb.group,
        tags=input_cfg.wandb.tags,
        job_type="publish_untrained" if artifact_name is None else "publish_trained",
        name=f"publish_{run_name}",
        reinit="finish_previous",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    run.summary["model/param_count"] = pretty_params
    run.summary["model/macs_per_second"] = pretty_macs
    run.summary["model/macs_pretty"] = pretty_macs
    run.summary["trained"] = artifact_name is not None

    # ----------------------------------------
    # Save checkpoint (to temp dir)
    # ----------------------------------------
    ckpt_dir = cfg.training.model_save_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{run_name}.pt")

    # Always use artifact_cfg if loaded, else cfg
    model_cfg_for_ckpt = model_cfg if model_cfg is not None else OmegaConf.to_container(cfg.model_arch, resolve=True)

    torch.save(
        {
            "cfg": {"model_arch": model_cfg_for_ckpt},
            "model_state": model.state_dict(),
        },
        ckpt_path,
    )

    # ----------------------------------------
    # Publish artifact
    # ----------------------------------------
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
    if hasattr(cfg, 'publish_model'):
        artifact_name = cfg.publish_model
        build_and_publish(cfg, artifact_name)
    else:
        build_and_publish(cfg)


if __name__ == "__main__":
    main()
