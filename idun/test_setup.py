#!/usr/bin/env python3
"""
test_setup.py - Test script to verify IDUN setup is working correctly
"""
import sys
import os


def test_imports():
    """Test that all required packages can be imported"""
    print("🔍 Testing package imports...")

    packages = [
        ("torch", "PyTorch"),
        ("torchaudio", "TorchAudio"),
        ("hydra", "Hydra"),
        ("wandb", "Weights & Biases"),
        ("torchmetrics", "TorchMetrics"),
        ("ncps", "Liquid Networks"),
        ("omegaconf", "OmegaConf"),
    ]

    failed = []
    for package, name in packages:
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name}: {e}")
            failed.append(name)

    # Try Mamba (optional, GPU-only)
    try:
        import mamba_ssm
        print(f"  ✅ Mamba SSM")
    except ImportError:
        print(f"  ⚠️  Mamba SSM (optional, requires GPU)")

    return len(failed) == 0


def test_cuda():
    """Test CUDA availability"""
    print("\n🎮 Testing CUDA...")

    import torch

    if torch.cuda.is_available():
        print(f"  ✅ CUDA available")
        print(f"  📍 CUDA version: {torch.version.cuda}")
        print(f"  🔢 Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  🎮 GPU {i}: {props.name}")
            print(f"     Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"     Compute Capability: {props.major}.{props.minor}")

        # Test CUDA operations
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.matmul(x, y)
            print(f"  ✅ CUDA operations working")
            return True
        except Exception as e:
            print(f"  ❌ CUDA operations failed: {e}")
            return False
    else:
        print(f"  ❌ CUDA not available")
        return False


def test_environment():
    """Test environment variables"""
    print("\n🔐 Testing environment...")

    required_vars = [
        "WANDB_API_KEY",
        "WANDB_ENTITY",
        "WANDB_PROJECT",
    ]

    missing = []
    for var in required_vars:
        if os.environ.get(var):
            print(f"  ✅ {var} is set")
        else:
            print(f"  ❌ {var} is not set")
            missing.append(var)

    if missing:
        print("\n  ⚠️  Missing environment variables!")
        print("  Please ensure they are set in your ~/.bashrc or shell profile:")
        for var in missing:
            print(f"    export {var}='your_value_here'")
        return False

    return True


def test_wandb_connection():
    """Test W&B connection"""
    print("\n☁️  Testing W&B connection...")

    try:
        import wandb
        api = wandb.Api()
        print(f"  ✅ Connected to W&B")
        print(f"  👤 Entity: {os.environ.get('WANDB_ENTITY', 'Not set')}")
        print(f"  📊 Project: {os.environ.get('WANDB_PROJECT', 'Not set')}")
        return True
    except Exception as e:
        print(f"  ❌ W&B connection failed: {e}")
        return False


def test_model_creation():
    """Test basic model creation"""
    print("\n🏗️  Testing model creation...")

    try:
        import torch
        import torch.nn as nn

        # Create a simple model
        model = nn.Sequential(
            nn.Conv1d(64, 128, 3),
            nn.ReLU(),
            nn.Conv1d(128, 64, 3)
        )

        # Test forward pass
        x = torch.randn(2, 64, 100)
        if torch.cuda.is_available():
            model = model.cuda()
            x = x.cuda()

        y = model(x)
        print(f"  ✅ Model creation and forward pass successful")
        print(f"  📐 Input shape: {list(x.shape)}")
        print(f"  📐 Output shape: {list(y.shape)}")
        return True
    except Exception as e:
        print(f"  ❌ Model test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Testing Cocktail Party setup on IDUN...\n")

    tests = [
        ("Package Imports", test_imports),
        ("CUDA Support", test_cuda),
        ("Environment Variables", test_environment),
        ("W&B Connection", test_wandb_connection),
        ("Model Creation", test_model_creation),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Download datasets: python manage_data.py download static-2-spk-noise:v2")
        print("2. Submit a test job: sbatch train_idun.slurm runs/1-offline/tcn")
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")
        print("\nCommon fixes:")
        print("- Ensure environment variables are set in ~/.bashrc")
        print("- Check that CUDA modules are loaded: module list")
        print("- For missing packages, activate venv and pip install them")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())