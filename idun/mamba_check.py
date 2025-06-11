
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
    layer_triton = Mamba2(d_model=128,d_state=64, d_conv=4, expand=4).to('cuda')    # Block expansion factor
    layer = Mamba2(d_model=128, d_state=64, d_conv=4, expand=2).to('cuda')
    print("✅ Mamba2 layers created successfully")
    print('🐞 Triton bug avoidance:', (layer.d_model*layer.expand/layer.headdim) % 8 == 0)

    # Test forward pass
    x = torch.randn(8, 320, 128).cuda()

    y = layer_triton(x)
    print("✅ Forward pass with triton bug avoidance successful")

    try:
        z = layer(x)
        print("✅ Regular Forward pass successful")
    except Exception as e:
        print(f"❌ Mamba-SSM is only working with triton bug avoidance. d_model*expand/headdim must be divisible by 8. "
              f" Error: {e}")

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
