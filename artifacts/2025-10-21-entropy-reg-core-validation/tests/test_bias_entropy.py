"""
Tests for bias entropy measurement.

Tests cover:
- Shannon entropy computation
- Bias map extraction from uniform images
- Entropy measurement on averaged bias maps
- Edge cases (uniform bias, concentrated bias)
- Numerical stability
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestShannonEntropy:
    """Test Shannon entropy computation."""

    def test_entropy_uniform_distribution(self):
        """Test entropy of uniform distribution equals log(N)."""
        from evaluation.metrics import shannon_entropy

        # Uniform distribution over 10000 pixels
        uniform = np.ones((100, 100)) / 10000

        H = shannon_entropy(uniform)

        # Shannon entropy of uniform: H = log(N)
        expected_H = np.log(10000)

        assert np.isclose(H, expected_H, atol=0.01), \
            f"Expected H = log(10000) = {expected_H:.4f}, got {H:.4f}"

    def test_entropy_concentrated_distribution(self):
        """Test entropy of concentrated distribution is low."""
        from evaluation.metrics import shannon_entropy

        # Concentrated distribution (single point)
        concentrated = np.zeros((100, 100))
        concentrated[50, 50] = 1.0

        H = shannon_entropy(concentrated)

        # Entropy of point mass is 0
        assert H < 0.1, f"Expected low entropy for concentrated distribution, got {H:.4f}"

    def test_entropy_increases_with_spread(self):
        """Test that entropy increases as distribution spreads."""
        from evaluation.metrics import shannon_entropy

        # Concentrated (10x10 region)
        concentrated = np.zeros((100, 100))
        concentrated[45:55, 45:55] = 1.0
        concentrated = concentrated / concentrated.sum()

        # More spread (40x40 region)
        spread = np.zeros((100, 100))
        spread[30:70, 30:70] = 1.0
        spread = spread / spread.sum()

        H_concentrated = shannon_entropy(concentrated)
        H_spread = shannon_entropy(spread)

        assert H_spread > H_concentrated, \
            f"Spread distribution should have higher entropy. Concentrated: {H_concentrated:.4f}, Spread: {H_spread:.4f}"

    def test_entropy_numerical_stability(self):
        """Test entropy handles zeros correctly."""
        from evaluation.metrics import shannon_entropy

        # Distribution with many zeros
        sparse = np.random.rand(100, 100)
        sparse[sparse < 0.9] = 0  # 90% zeros
        sparse = sparse / sparse.sum()

        H = shannon_entropy(sparse)

        assert np.isfinite(H), f"Entropy should handle zeros without NaN/Inf, got {H}"
        assert H >= 0, f"Entropy should be non-negative, got {H}"

    def test_entropy_normalized_input(self):
        """Test entropy requires input to sum to 1."""
        from evaluation.metrics import shannon_entropy

        # Non-normalized input
        non_normalized = np.random.rand(100, 100)

        with pytest.raises(ValueError, match="sum to 1"):
            shannon_entropy(non_normalized)


class TestBiasMapExtraction:
    """Test bias map extraction from models."""

    def test_extract_bias_from_uniform_images(self):
        """Test extracting bias maps from uniform images."""
        from evaluation.metrics import extract_bias_maps
        from models.deepgaze3 import DeepGazeIII

        # Create mock model
        model = DeepGazeIII()
        model.eval()

        # Extract bias maps
        bias_maps = extract_bias_maps(
            model,
            image_size=(1024, 768),
            num_samples=4,
            intensities=[0.0, 0.5, 1.0],
            device='cpu'
        )

        # Should return list of bias maps
        assert len(bias_maps) > 0, "Should return at least one bias map"
        assert all(isinstance(bm, np.ndarray) for bm in bias_maps), \
            "All bias maps should be numpy arrays"

    def test_bias_maps_shape(self):
        """Test bias maps have correct shape."""
        from evaluation.metrics import extract_bias_maps
        from models.deepgaze3 import DeepGazeIII

        model = DeepGazeIII()
        model.eval()

        image_size = (1024, 768)
        bias_maps = extract_bias_maps(
            model,
            image_size=image_size,
            num_samples=2,
            device='cpu'
        )

        for bm in bias_maps:
            assert bm.shape == image_size[::-1], \
                f"Expected shape {image_size[::-1]}, got {bm.shape}"

    def test_bias_maps_normalized(self):
        """Test bias maps are normalized to sum to 1."""
        from evaluation.metrics import extract_bias_maps
        from models.deepgaze3 import DeepGazeIII

        model = DeepGazeIII()
        model.eval()

        bias_maps = extract_bias_maps(
            model,
            image_size=(1024, 768),
            num_samples=3,
            device='cpu'
        )

        for bm in bias_maps:
            assert np.isclose(bm.sum(), 1.0, atol=1e-5), \
                f"Bias map should sum to 1, got {bm.sum():.10f}"

    def test_multiple_intensity_values(self):
        """Test bias extraction with multiple uniform intensities."""
        from evaluation.metrics import extract_bias_maps
        from models.deepgaze3 import DeepGazeIII

        model = DeepGazeIII()
        model.eval()

        intensities = [0.0, 0.25, 0.5, 0.75, 1.0]
        bias_maps = extract_bias_maps(
            model,
            image_size=(1024, 768),
            num_samples=2,
            intensities=intensities,
            device='cpu'
        )

        # Should have num_samples * len(intensities) bias maps
        expected_count = 2 * len(intensities)
        assert len(bias_maps) == expected_count, \
            f"Expected {expected_count} bias maps, got {len(bias_maps)}"


class TestAverageBiasMap:
    """Test averaging bias maps."""

    def test_average_bias_maps(self):
        """Test averaging multiple bias maps."""
        from evaluation.metrics import average_bias_maps

        # Create multiple bias maps
        bias_maps = [
            np.random.rand(100, 100) for _ in range(5)
        ]
        # Normalize each
        bias_maps = [bm / bm.sum() for bm in bias_maps]

        avg_bias = average_bias_maps(bias_maps)

        assert avg_bias.shape == (100, 100), \
            f"Expected shape (100, 100), got {avg_bias.shape}"
        assert np.isclose(avg_bias.sum(), 1.0, atol=1e-5), \
            f"Average bias should sum to 1, got {avg_bias.sum():.10f}"

    def test_average_single_bias_map(self):
        """Test averaging single bias map returns itself."""
        from evaluation.metrics import average_bias_maps

        bias_map = np.random.rand(100, 100)
        bias_map = bias_map / bias_map.sum()

        avg_bias = average_bias_maps([bias_map])

        assert np.allclose(avg_bias, bias_map), \
            "Average of single map should equal the map itself"


class TestBiasEntropyMeasurement:
    """Test complete bias entropy measurement pipeline."""

    def test_measure_bias_entropy(self):
        """Test measuring bias entropy from model."""
        from evaluation.metrics import measure_bias_entropy
        from models.deepgaze3 import DeepGazeIII

        model = DeepGazeIII()
        model.eval()

        entropy = measure_bias_entropy(
            model,
            image_size=(1024, 768),
            num_samples=4,
            device='cpu'
        )

        assert isinstance(entropy, float), \
            f"Entropy should be float, got {type(entropy)}"
        assert np.isfinite(entropy), \
            f"Entropy should be finite, got {entropy}"
        assert entropy >= 0, \
            f"Entropy should be non-negative, got {entropy}"

    def test_bias_entropy_range(self):
        """Test bias entropy is in reasonable range."""
        from evaluation.metrics import measure_bias_entropy
        from models.deepgaze3 import DeepGazeIII

        model = DeepGazeIII()
        model.eval()

        entropy = measure_bias_entropy(
            model,
            image_size=(1024, 768),
            num_samples=16,
            device='cpu'
        )

        # For 1024x768 image, maximum entropy is log(1024*768) ≈ 13.56 nats
        max_entropy = np.log(1024 * 768)

        assert 0 <= entropy <= max_entropy + 0.5, \
            f"Entropy should be between 0 and {max_entropy:.2f}, got {entropy:.4f}"

    def test_bias_entropy_comparison(self):
        """Test comparing bias entropy between two models."""
        from evaluation.metrics import measure_bias_entropy, compare_bias_entropy
        from models.deepgaze3 import DeepGazeIII

        model1 = DeepGazeIII()
        model1.eval()

        model2 = DeepGazeIII()
        model2.eval()

        entropy1 = measure_bias_entropy(model1, image_size=(512, 384), num_samples=4, device='cpu')
        entropy2 = measure_bias_entropy(model2, image_size=(512, 384), num_samples=4, device='cpu')

        # Compute percentage increase
        percent_increase = ((entropy2 - entropy1) / entropy1) * 100

        assert np.isfinite(percent_increase), \
            f"Percentage increase should be finite, got {percent_increase}"


class TestEntropyStatistics:
    """Test entropy statistics computation."""

    def test_normalized_entropy(self):
        """Test normalized entropy (H / H_max)."""
        from evaluation.metrics import normalized_entropy

        # Uniform distribution
        uniform = np.ones((100, 100)) / 10000
        H_norm = normalized_entropy(uniform)

        # Normalized entropy of uniform should be 1.0
        assert np.isclose(H_norm, 1.0, atol=0.01), \
            f"Normalized entropy of uniform should be 1.0, got {H_norm:.4f}"

    def test_normalized_entropy_range(self):
        """Test normalized entropy is between 0 and 1."""
        from evaluation.metrics import normalized_entropy

        np.random.seed(42)
        for _ in range(10):
            distribution = np.random.rand(50, 50)
            distribution = distribution / distribution.sum()

            H_norm = normalized_entropy(distribution)

            assert 0 <= H_norm <= 1, \
                f"Normalized entropy should be in [0, 1], got {H_norm:.4f}"
