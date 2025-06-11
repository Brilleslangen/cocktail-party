#!/bin/bash
# Quick activation script
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

source /cluster/home/nicolts/projects/cocktail-party/venv/bin/activate

echo "✅ Environment activated!"
echo "📍 Working directory: /cluster/home/nicolts/projects/cocktail-party"
cd "/cluster/home/nicolts/projects/cocktail-party"
