export WANDB_ARTIFACT_CACHE_DIR="${WANDB_ARTIFACT_CACHE_DIR:-$HOME/cocktail-party/artifacts}"
python -m src.executables.evaluate --config-name=jobs/evaluate_offline_all hydra/launcher=submitit_idun hydra.launcher.timeout_min=30
