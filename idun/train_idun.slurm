#!/bin/bash
#SBATCH --job-name=cocktail-party-train
#SBATCH --output=logs/train_%j.out        # Standard output log
#SBATCH --error=logs/train_%j.err         # Standard error log
#SBATCH --ntasks=1                        # Number of tasks (processes)
#SBATCH --cpus-per-task=12                # Number of CPU cores per task
#SBATCH --mem=32G                         # Memory per node
#SBATCH --time=24:00:00                   # Time limit (HH:MM:SS)
#SBATCH --partition=gpu                   # Partition name (check IDUN docs)
#SBATCH --gres=gpu:1                      # Number of GPUs
#SBATCH --account=YOUR_ACCOUNT            # Your IDUN account

# Optional: Email notifications
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@ntnu.no

# Load modules
module purge
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Set project directory
PROJECT_DIR="$HOME/projects/cocktail-party"
cd "$PROJECT_DIR"

# Activate virtual environment
source "$PROJECT_DIR/venv/bin/activate"

# Set some PyTorch optimizations
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"  # Adjust based on GPU

# Log system information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Run training with specific configuration
# You can pass the config as a SLURM argument or hardcode it
CONFIG=${1:-"runs/1-offline/tcn"}

echo "Starting training with config: $CONFIG"
python -m src.executables.train --config-name="$CONFIG" \
    training.params.num_workers=8 \
    training.params.batch_size=16 \
    wandb.enabled=true

echo "Job finished at: $(date)"