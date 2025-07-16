#!/bin/bash
# Quick activation script
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"

echo "✅ Environment activated!"
echo "📍 Working directory: $SCRIPT_DIR"
cd "$SCRIPT_DIR"