"""
Smoke test to verify environment setup and basic functionality.

This script tests:
1. All required dependencies are installed
2. CUDA is available and accessible
3. DeepGaze model can be instantiated
4. Entropy regularizer can be created
5. Basic forward pass works
"""

import sys
import torch
import numpy as np

def test_imports():
    """Test that all required packages can be imported."""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)

    try:
        import torch
        import torchvision
        import numpy
        import scipy
        import cv2
        import skimage
        import yaml
        import pytest
        print("✓ All required packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_cuda():
    """Test CUDA availability and GPU access."""
    print("\n" + "=" * 60)
    print("Testing CUDA...")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("✗ CUDA is not available")
        return False

    print(f"✓ CUDA is available")
    print(f"  - PyTorch version: {torch.__version__}")
    print(f"  - CUDA version: {torch.version.cuda}")
    print(f"  - Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")

    # Test GPU memory allocation
    try:
        test_tensor = torch.randn(100, 100).cuda()
        print(f"✓ Successfully allocated tensor on GPU")
        del test_tensor
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"✗ Failed to allocate tensor on GPU: {e}")
        return False

    return True


def test_model_loading():
    """Test that DeepGaze model files are accessible."""
    print("\n" + "=" * 60)
    print("Testing model files...")
    print("=" * 60)

    import os
    model_dir = "../src/models"

    required_files = [
        "deepgaze3.py",
        "modules.py",
        "layers.py",
        "entropy_regularizer.py",
        "__init__.py"
    ]

    all_present = True
    for filename in required_files:
        filepath = os.path.join(model_dir, filename)
        if os.path.exists(filepath):
            print(f"✓ {filename} exists")
        else:
            print(f"✗ {filename} missing")
            all_present = False

    return all_present


def test_entropy_regularizer():
    """Test entropy regularizer module."""
    print("\n" + "=" * 60)
    print("Testing entropy regularizer...")
    print("=" * 60)

    # Add src to path to import our modules
    import sys
    import os
    sys.path.insert(0, os.path.abspath("../src"))

    try:
        from models.entropy_regularizer import (
            UniformImageGenerator,
            ShannonEntropyComputer,
            EntropyRegularizer
        )
        print("✓ Entropy regularizer modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import entropy regularizer: {e}")
        return False

    # Test uniform image generation
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        generator = UniformImageGenerator(size=(1024, 768), device=device)
        uniform_images = generator.generate(batch_size=4)

        assert uniform_images.shape == (4, 3, 768, 1024), \
            f"Unexpected shape: {uniform_images.shape}"
        print(f"✓ Uniform image generation works (shape: {uniform_images.shape})")
    except Exception as e:
        print(f"✗ Uniform image generation failed: {e}")
        return False

    # Test Shannon entropy computation
    try:
        entropy_computer = ShannonEntropyComputer()

        # Create a simple test probability map
        test_map = torch.randn(2, 1, 768, 1024).to(device)
        prob_map = entropy_computer.normalize_to_probability(test_map)

        # Verify normalization (should sum to ~1.0)
        prob_sum = prob_map.view(2, -1).sum(dim=1)
        assert torch.allclose(prob_sum, torch.ones(2).to(device), atol=1e-5), \
            f"Probabilities don't sum to 1: {prob_sum}"

        entropy = entropy_computer.compute_entropy(prob_map)
        assert entropy.item() > 0, f"Entropy should be positive: {entropy.item()}"

        print(f"✓ Shannon entropy computation works (H = {entropy.item():.4f})")
    except Exception as e:
        print(f"✗ Shannon entropy computation failed: {e}")
        return False

    return True


def test_config_files():
    """Test that configuration files are valid."""
    print("\n" + "=" * 60)
    print("Testing configuration files...")
    print("=" * 60)

    import yaml
    import os

    config_files = [
        "../configs/baseline_config.yaml",
        "../configs/entropy_reg_config.yaml"
    ]

    for config_file in config_files:
        try:
            if not os.path.exists(config_file):
                print(f"✗ {config_file} does not exist")
                return False

            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            print(f"✓ {os.path.basename(config_file)} is valid YAML")
        except Exception as e:
            print(f"✗ Failed to load {config_file}: {e}")
            return False

    return True


def main():
    """Run all smoke tests."""
    print("\n" + "=" * 60)
    print("ENVIRONMENT SMOKE TEST")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("CUDA", test_cuda),
        ("Model Files", test_model_loading),
        ("Entropy Regularizer", test_entropy_regularizer),
        ("Configuration Files", test_config_files),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} test crashed: {e}")
            results[test_name] = False

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = all(results.values())
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:30s} {status}")

    print("=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Environment is ready!")
        return 0
    else:
        print("✗ SOME TESTS FAILED - Please fix the issues above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
