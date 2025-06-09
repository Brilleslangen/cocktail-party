#!/bin/bash
set -e

# Clone S4 and install
git clone https://github.com/HazyResearch/state-spaces.git
cd state-spaces
pip install .

echo "S4 installed in venv '$VENV_NAME' with PyTorch ($CUDA_PYTORCH_TAG)."
