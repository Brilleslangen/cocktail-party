#!/bin/bash
# interactive_gpu.sh - Start an interactive GPU session for debugging

# Default values
TIME=${1:-"2:00:00"}  # Default 2 hours
MEM=${2:-"16G"}       # Default 16GB RAM

echo "🚀 Requesting interactive GPU session..."
echo "   Time: $TIME"
echo "   Memory: $MEM"
echo ""

# Request interactive session
srun --partition=gpu \
     --account=YOUR_ACCOUNT \
     --time=$TIME \
     --mem=$MEM \
     --cpus-per-task=8 \
     --gres=gpu:1 \
     --pty bash -i

# The above command will give you an interactive shell on a GPU node
# Once you're in, you can:
# 1. module load Python/3.11.3-GCCcore-12.3.0 CUDA/12.1.1 cuDNN/8.9.2.26-CUDA-12.1.1
# 2. source ~/projects/cocktail-party/activate.sh
# 3. Run your training/debugging commands interactively