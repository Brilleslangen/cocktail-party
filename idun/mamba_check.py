#!/usr/bin/env python3
"""
check_mamba.py - Quick check if mamba-ssm is properly installed
"""
import sys

print("🐍 Checking Mamba-SSM installation...")
print("-" * 40)

# Check Python version
print(f"Python version: {sys.version}")
if sys.version_info < (3, 11):
    print("❌ Python 3.11+ required for mamba-ssm!")
    sys.exit(1)
else:
    print("✅ Python version OK")

# Check CUDA
try:
    import torch

    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.version.cuda}")
        print(f"   Device: {torch.cuda.get_device_name()}")
    else:
        print("❌ CUDA not available!")
except ImportError:
    print("❌ PyTorch not installed!")
    sys.exit(1)

# Check causal-conv1d
try:
    import causal_conv1d

    print("✅ causal-conv1d imported successfully")
except ImportError as e:
    print(f"❌ causal-conv1d import failed: {e}")

# Check mamba-ssm
try:
    import mamba_ssm

    print("✅ mamba-ssm imported successfully")
    from mamba_ssm import Mamba2

    print("✅ Mamba2 imported successfully")

    # Try to create a simple Mamba2 layer
    layer = Mamba2(d_model=128,d_state=64, d_conv=4, expand=4).to('cuda')    # Block expansion factor
    print(layer.d_model*layer.expand/layer.headdim)

    print("✅ Mamba2 layer created successfully")

    # Test forward pass
    x = torch.randn(8, 320, 128).cuda()
    print(x.shape, x.stride())  # Make sure stride[0] and stride[2] are multiples of 8
    y = layer(x)
    print("✅ Forward pass successful")

    print("\n🎉 Mamba-SSM is properly installed and working!")

except ImportError as e:
    print(f"❌ mamba-ssm import failed: {e}")
    print("\nTo install mamba-ssm:")
    print("1. Ensure Python 3.11+ and CUDA modules are loaded")
    print("2. Run: pip install --no-build-isolation mamba-ssm[causal-conv1d]")
    sys.exit(1)
except Exception as e:
    import traceback
    print("❌ Error testing Mamba:")
    traceback.print_exc()