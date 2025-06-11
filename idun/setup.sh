#!/bin/bash
# setup_idun.sh - Setup script for NTNU IDUN cluster
# This script creates a virtual environment and installs all dependencies

set -e  # Exit on error

echo "🚀 Setting up Cocktail Party project on IDUN..."

# Load required modules (adjust versions as needed for IDUN)
module purge
module load Python/3.11.3-GCCcore-12.3.0  # Required for mamba-ssm
module load CUDA/12.1.1  # or CUDA/12.6.0 if available - check with 'module avail CUDA'
module load cuDNN/8.9.2.26-CUDA-12.1.1  # For deep learning

# Determine repository root (directory one level up from this script)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Use repository root as project directory
PROJECT_DIR="$REPO_ROOT"
cd "$PROJECT_DIR"

# Create virtual environment
VENV_DIR="$PROJECT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "🐍 Creating virtual environment..."
    python -m venv "$VENV_DIR"
else
    echo "🐍 Virtual environment already exists"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Upgrade pip and install wheel
echo "📦 Upgrading pip and installing wheel..."
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA support (adjust CUDA version as needed)
echo "🔥 Installing PyTorch with CUDA support..."
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# If using CUDA 12.6, you might need to use a different index URL or check PyTorch compatibility

# Install mamba-ssm and causal-conv1d (requires CUDA)
echo "🐍 Installing Mamba dependencies..."
if pip install --no-build-isolation mamba-ssm[causal-conv1d]; then
    echo "✅ Mamba-SSM installed successfully"
else
    echo "⚠️  Mamba-SSM installation failed. This is optional - other models will still work."
    echo "   To use Mamba, ensure Python 3.11+ and CUDA are properly loaded."
fi

# Install remaining requirements
echo "📦 Installing remaining requirements..."
pip install hydra-core==1.3.2
pip install numpy~=2.0.0  # Compatible with Python 3.11
pip install omegaconf==2.3.0
pip install "scipy>=1.10.0"  # More flexible version
pip install soundfile==0.12.1
pip install tqdm==4.67.1
pip install "wandb>=0.15.0"  # More flexible version
pip install "torchmetrics[audio]>=1.0.0"  # More flexible version
pip install "ncps>=0.0.7"  # More flexible version
pip install thop==0.1.1.post2209072238

# Create necessary directories
mkdir -p datasets artifacts outputs wandb logs

# Check if API keys are already set in environment
echo "🔐 Checking environment variables..."
if [ -n "$WANDB_API_KEY" ]; then
    echo "✅ WANDB_API_KEY is already set in environment"
else
    echo "⚠️  WANDB_API_KEY not found. Please add to your ~/.bashrc:"
    echo "    export WANDB_API_KEY='your_key_here'"
fi

# Create a source script for easy activation
cat > "$PROJECT_DIR/activate.sh" << EOF
#!/bin/bash
# Quick activation script
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

source $VENV_DIR/bin/activate

echo "✅ Environment activated!"
echo "📍 Working directory: $PROJECT_DIR"
cd "$PROJECT_DIR"
EOF

chmod +x "$PROJECT_DIR/activate.sh"

echo "✅ Setup complete!"
echo ""
echo "⚠️  Important Notes:"
echo "- Mamba-SSM requires Python 3.11+ and CUDA to be properly loaded"
echo "- If mamba installation failed, you can still use TCN, Transformer, and Liquid models"
echo "- Check available CUDA versions with: module avail CUDA"
echo ""
echo "To activate the environment in future sessions, run:"
echo "  source $PROJECT_DIR/activate.sh"
echo ""
echo "Next steps:"
echo "1. Ensure W&B API key is set in your environment"
echo "2. Test the setup: python test_setup.py"
echo "3. Submit training jobs using the SLURM scripts"
