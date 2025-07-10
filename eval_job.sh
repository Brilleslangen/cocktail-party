export WANDB_ARTIFACT_CACHE_DIR=/cluster/home/nicolts/cocktail-party/artifacts
python -m src.executables.train --config-name=jobs/evaluate_offline_all hydra/launcher=submitit_idun
