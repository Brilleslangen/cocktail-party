#!/bin/bash
# setup_idun.sh - Setup script for NTNU IDUN cluster
# This script creates a virtual environment and installs all dependencies

set -e  # Exit on error

echo "🚀 Setting up Cocktail Party project on IDUN..."

# Load required modules (adjust versions as needed for IDUN)
module purge
module load Python/3.10.8-GCCcore-12.2.0  # Adjust version as available
module load CUDA/12.1.1  # Required for mamba-ssm and PyTorch
module load cuDNN/8.9.2.26-CUDA-12.1.1  # For deep learning
module load git/2.38.1-GCCcore-12.2.0

# Create project directory structure
PROJECT_NAME="cocktail-party"
PROJECT_DIR="$HOME/projects/$PROJECT_NAME"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Clone repository if not already present
if [ ! -d ".git" ]; then
    echo "📦 Cloning repository..."
    # Try SSH first (recommended if SSH keys are set up)
    if git clone git@github.com:YOUR_USERNAME/YOUR_REPO.git . 2>/dev/null; then
        echo "✅ Repository cloned via SSH"
    else
        echo "⚠️  SSH clone failed, trying HTTPS..."
        echo "  Enter your GitHub username:"
        read -r github_user
        echo "  Enter your GitHub repo name:"
        read -r github_repo
        git clone "https://github.com/${github_user}/${github_repo}.git" .
    fi
else
    echo "📦 Repository already exists, pulling latest changes..."
    git pull
fi

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
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# Install mamba-ssm and causal-conv1d (requires CUDA)
echo "🐍 Installing Mamba dependencies..."
pip install causal-conv1d==1.4.0
pip install mamba-ssm==2.2.4

# Install remaining requirements
echo "📦 Installing remaining requirements..."
pip install hydra-core==1.3.2
pip install numpy==2.1.3
pip install omegaconf==2.3.0
pip install scipy==1.14.1
pip install soundfile==0.12.1
pip install tqdm==4.67.1
pip install wandb==0.19.1
pip install torchmetrics[audio]==1.6.0
pip install ncps==1.0.1
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
module load Python/3.10.8-GCCcore-12.2.0
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
echo "To activate the environment in future sessions, run:"
echo "  source $PROJECT_DIR/activate.sh"
echo ""
echo "Next steps:"
echo "1. Ensure W&B API key is set in your environment"
echo "2. Test the setup: python test_setup.py"
echo "3. Submit training jobs using the SLURM scripts"