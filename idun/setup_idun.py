#!/usr/bin/env python3
"""Setup script for NTNU IDUN cluster.

This script sets up a virtual environment and installs all
requirements for the Cocktail Party project, including
Mamba-2 (``mamba-ssm`` package). It expects that the
necessary modules for Python and CUDA are available on IDUN.

The script mirrors ``idun/setup_idun.sh`` but allows running
from Python. Run it after loading the following modules::

  module purge
  module load Python/3.10.8-GCCcore-12.2.0
  module load CUDA/12.1.1
  module load cuDNN/8.9.2.26-CUDA-12.1.1
  module load git/2.45.1-GCCcore-13.3.0

"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

MODULES = [
    "Python/3.10.8-GCCcore-12.2.0",
    "CUDA/12.1.1",
    "cuDNN/8.9.2.26-CUDA-12.1.1",
    "git/2.45.1-GCCcore-13.3.0",
]


def _load_module_env() -> dict[str, str]:
    """Load modules in a login shell and return the resulting environment."""
    command = "module purge && " + " && ".join(f"module load {m}" for m in MODULES) + " && env"
    result = subprocess.run(
        ["bash", "-l", "-c", command], capture_output=True, text=True, check=True
    )
    env = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    return env


def _run(cmd: str, env: dict[str, str]) -> None:
    print(f"[RUN] {cmd}")
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash", env=env)


def main() -> None:
    env = _load_module_env()

    project_dir = Path(os.environ.get("HOME", "")) / "projects" / "cocktail-party"
    project_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(project_dir)

    if not Path(".git").exists():
        print("Cloning repository...")
        _run(
            "git clone git@github.com:YOUR_USERNAME/YOUR_REPO.git . || "
            "git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git .",
            env,
        )
    else:
        _run("git pull", env)

    venv_dir = project_dir / "venv"
    if not venv_dir.exists():
        _run(f"python -m venv {venv_dir}", env)
    pip = venv_dir / "bin" / "pip"

    _run(f"{pip} install --upgrade pip wheel setuptools", env)
    _run(
        f"{pip} install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121",
        env,
    )
    _run(f"{pip} install causal-conv1d==1.4.0", env)
    _run(f"{pip} install mamba-ssm==2.2.4", env)

    other_pkgs = [
        "hydra-core==1.3.2",
        "numpy==2.1.3",
        "omegaconf==2.3.0",
        "scipy==1.14.1",
        "soundfile==0.12.1",
        "tqdm==4.67.1",
        "wandb==0.19.1",
        "torchmetrics[audio]==1.6.0",
        "ncps==1.0.1",
        "thop==0.1.1.post2209072238",
    ]
    _run(f"{pip} install {' '.join(other_pkgs)}", env)

    for d in ["datasets", "artifacts", "outputs", "wandb", "logs"]:
        (project_dir / d).mkdir(exist_ok=True)

    activate_sh = project_dir / "activate.sh"
    with activate_sh.open("w") as fh:
        fh.write("#!/bin/bash\n")
        fh.write("module purge\n")
        for m in MODULES:
            fh.write(f"module load {m}\n")
        fh.write(f"source {venv_dir}/bin/activate\n")
        fh.write("echo '✅ Environment activated!'\n")
        fh.write(f"cd {project_dir}\n")
    activate_sh.chmod(0o755)

    print("✅ Setup complete!\n")
    print(f"To activate in future sessions, run: source {activate_sh}")


if __name__ == "__main__":
    main()