#!/bin/bash
# Quick activation script
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load libsndfile/1.0.31-GCCcore-12.3.0

source /cluster/home/nicolts/cocktail-party/venv/bin/activate

echo "✅ Environment activated!"
echo "📍 Working directory: /cluster/home/nicolts/cocktail-party"
cd "/cluster/home/nicolts/cocktail-party"
