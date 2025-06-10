#!/bin/bash
# submit_experiments.sh - Submit multiple experiments to SLURM

set -e

PROJECT_DIR="$HOME/projects/cocktail-party"
cd "$PROJECT_DIR"

# Create logs directory
mkdir -p logs

# Define experiments to run
EXPERIMENTS=(
    "runs/1-offline/tcn"
    "runs/1-offline/transformer"
    "runs/1-offline/mamba"
    "runs/1-offline/liquid"
)

# GPU configurations for different experiments
declare -A GPU_CONFIGS
GPU_CONFIGS["tcn"]="gpu:1"              # TCN can run on 1 GPU
GPU_CONFIGS["transformer"]="gpu:1"       # Transformer benefits from more memory
GPU_CONFIGS["mamba"]="gpu:1"            # Mamba requires CUDA capability
GPU_CONFIGS["liquid"]="gpu:1"           # Liquid networks

# Memory configurations
declare -A MEM_CONFIGS
MEM_CONFIGS["tcn"]="32G"
MEM_CONFIGS["transformer"]="48G"        # Transformer needs more memory
MEM_CONFIGS["mamba"]="32G"
MEM_CONFIGS["liquid"]="32G"

# Time configurations
declare -A TIME_CONFIGS
TIME_CONFIGS["tcn"]="24:00:00"
TIME_CONFIGS["transformer"]="36:00:00"   # Transformer might take longer
TIME_CONFIGS["mamba"]="24:00:00"
TIME_CONFIGS["liquid"]="30:00:00"

# Submit each experiment
for exp in "${EXPERIMENTS[@]}"; do
    # Extract experiment type from path
    exp_type=$(basename "$exp")

    # Get configurations
    gpu_config=${GPU_CONFIGS[$exp_type]:-"gpu:1"}
    mem_config=${MEM_CONFIGS[$exp_type]:-"32G"}
    time_config=${TIME_CONFIGS[$exp_type]:-"24:00:00"}

    echo "Submitting experiment: $exp"
    echo "  GPU: $gpu_config"
    echo "  Memory: $mem_config"
    echo "  Time: $time_config"

    # Create a temporary SLURM script for this specific experiment
    cat > "logs/train_${exp_type}.slurm" << EOF
#!/bin/bash
#SBATCH --job-name=cocktail-${exp_type}
#SBATCH --output=logs/train_${exp_type}_%j.out
#SBATCH --error=logs/train_${exp_type}_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=${mem_config}
#SBATCH --time=${time_config}
#SBATCH --partition=gpu
#SBATCH --gres=${gpu_config}
#SBATCH --account=YOUR_ACCOUNT

# Load modules
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Set project directory
PROJECT_DIR="$HOME/projects/cocktail-party"
cd "\$PROJECT_DIR"

# Activate virtual environment
source "\$PROJECT_DIR/venv/bin/activate"

# Set PyTorch optimizations
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK

# Log system information
echo "Experiment: ${exp}"
echo "Job started at: \$(date)"
echo "Running on node: \$(hostname)"
echo "Job ID: \$SLURM_JOB_ID"
nvidia-smi

# Run training
python -m src.executables.train --config-name="${exp}" \\
    training.params.num_workers=8 \\
    training.params.batch_size=16 \\
    wandb.enabled=true

echo "Job finished at: \$(date)"
EOF

    # Submit the job
    job_id=$(sbatch "logs/train_${exp_type}.slurm" | awk '{print $4}')
    echo "  Submitted with Job ID: $job_id"
    echo ""

    # Optional: Add a small delay between submissions
    sleep 2
done

echo "All experiments submitted!"
echo ""
echo "To check job status, run:"
echo "  squeue -u $USER"
echo ""
echo "To check specific job output:"
echo "  tail -f logs/train_<experiment>_<job_id>.out"