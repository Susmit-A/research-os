"""
Tests for entropy regularization components.

Tests cover:
- Uniform image generation with constant pixel values
- Shannon entropy computation on toy examples
- Bias map extraction from DeepGaze 3 model
- Integration of entropy regularization loss
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestUniformImageGenerator:
    """Test suite for uniform image generation."""

    def test_generator_initialization(self):
        """Test that generator initializes with correct parameters."""
        from models.entropy_regularizer import UniformImageGenerator

        generator = UniformImageGenerator(size=(1024, 768), device='cpu')

        assert generator.size == (1024, 768)
        assert generator.device == 'cpu'

    def test_generate_correct_shape(self):
        """Test that generated images have correct shape."""
        from models.entropy_regularizer import UniformImageGenerator

        generator = UniformImageGenerator(size=(1024, 768), device='cpu')
        images = generator.generate(batch_size=16)

        # Shape should be (batch, channels, height, width)
        assert images.shape == (16, 3, 768, 1024), \
            f"Expected shape (16, 3, 768, 1024), got {images.shape}"

    def test_generate_different_batch_sizes(self):
        """Test generating different batch sizes."""
        from models.entropy_regularizer import UniformImageGenerator

        generator = UniformImageGenerator(size=(1024, 768), device='cpu')

        for batch_size in [1, 4, 8, 16, 32]:
            images = generator.generate(batch_size=batch_size)
            assert images.shape[0] == batch_size

    def test_uniform_pixel_values(self):
        """Test that images have constant pixel values (uniform)."""
        from models.entropy_regularizer import UniformImageGenerator

        generator = UniformImageGenerator(size=(1024, 768), device='cpu')
        images = generator.generate(batch_size=4)

        # Each image should have constant value across all pixels
        for i in range(images.shape[0]):
            img = images[i]
            # Check that all pixels in each channel are the same
            for c in range(3):
                channel = img[c]
                assert torch.all(channel == channel[0, 0]), \
                    f"Image {i}, channel {c} is not uniform"

    def test_configurable_intensity_values(self):
        """Test that intensity values can be configured."""
        from models.entropy_regularizer import UniformImageGenerator

        generator = UniformImageGenerator(size=(1024, 768), device='cpu')

        # Test with specific intensity values
        intensity_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        images = generator.generate(batch_size=10, intensity_values=intensity_values)

        # Check that generated images use values from the list
        unique_values = set()
        for i in range(images.shape[0]):
            value = images[i, 0, 0, 0].item()
            unique_values.add(value)

        # All unique values should be from intensity_values
        for val in unique_values:
            assert val in intensity_values, \
                f"Found unexpected intensity value: {val}"

    def test_default_intensity_values(self):
        """Test default intensity values [0.0, 0.5, 1.0]."""
        from models.entropy_regularizer import UniformImageGenerator

        generator = UniformImageGenerator(size=(1024, 768), device='cpu')
        images = generator.generate(batch_size=30)

        # Collect all unique values
        unique_values = set()
        for i in range(images.shape[0]):
            value = images[i, 0, 0, 0].item()
            unique_values.add(round(value, 1))  # Round to avoid float precision issues

        # Should only use default values [0.0, 0.5, 1.0]
        assert unique_values.issubset({0.0, 0.5, 1.0}), \
            f"Found unexpected values: {unique_values}"

    def test_device_placement(self):
        """Test that images are created on correct device."""
        from models.entropy_regularizer import UniformImageGenerator

        # Test CPU
        generator_cpu = UniformImageGenerator(size=(1024, 768), device='cpu')
        images_cpu = generator_cpu.generate(batch_size=4)
        assert images_cpu.device.type == 'cpu'

        # Test CUDA if available
        if torch.cuda.is_available():
            generator_cuda = UniformImageGenerator(size=(1024, 768), device='cuda')
            images_cuda = generator_cuda.generate(batch_size=4)
            assert images_cuda.device.type == 'cuda'

    def test_different_image_sizes(self):
        """Test generating images with different sizes."""
        from models.entropy_regularizer import UniformImageGenerator

        sizes = [(512, 384), (1024, 768), (2048, 1536)]

        for width, height in sizes:
            generator = UniformImageGenerator(size=(width, height), device='cpu')
            images = generator.generate(batch_size=4)

            assert images.shape == (4, 3, height, width), \
                f"Expected shape (4, 3, {height}, {width}), got {images.shape}"


class TestShannonEntropyComputer:
    """Test suite for Shannon entropy computation."""

    def test_entropy_computer_initialization(self):
        """Test that entropy computer initializes correctly."""
        from models.entropy_regularizer import ShannonEntropyComputer

        computer = ShannonEntropyComputer(eps=1e-8)
        assert computer.eps == 1e-8

    def test_normalize_to_probability(self):
        """Test normalization of maps to probability distributions."""
        from models.entropy_regularizer import ShannonEntropyComputer

        computer = ShannonEntropyComputer()

        # Create test saliency map
        saliency_map = torch.randn(4, 1, 100, 100)

        # Normalize
        prob_map = computer.normalize_to_probability(saliency_map)

        # Check shape preserved
        assert prob_map.shape == saliency_map.shape

        # Check probabilities sum to 1 (per image)
        for i in range(4):
            prob_sum = prob_map[i].sum().item()
            assert abs(prob_sum - 1.0) < 1e-5, \
                f"Probabilities should sum to 1, got {prob_sum}"

        # Check all probabilities >= 0
        assert torch.all(prob_map >= 0), "Probabilities should be non-negative"

    def test_entropy_computation_uniform(self):
        """Test entropy on uniform distribution (maximum entropy)."""
        from models.entropy_regularizer import ShannonEntropyComputer

        computer = ShannonEntropyComputer()

        # Create uniform distribution (all pixels equal probability)
        n_pixels = 100 * 100
        uniform_prob = torch.ones(1, 1, 100, 100) / n_pixels

        entropy = computer.compute_entropy(uniform_prob)

        # Maximum entropy for uniform distribution = log(n_pixels)
        expected_entropy = np.log(n_pixels)
        assert abs(entropy.item() - expected_entropy) < 0.1, \
            f"Uniform distribution entropy should be ~{expected_entropy}, got {entropy.item()}"

    def test_entropy_computation_peaked(self):
        """Test entropy on peaked distribution (low entropy)."""
        from models.entropy_regularizer import ShannonEntropyComputer

        computer = ShannonEntropyComputer()

        # Create peaked distribution (one pixel has high probability)
        peaked_prob = torch.zeros(1, 1, 100, 100)
        peaked_prob[0, 0, 50, 50] = 0.99
        peaked_prob[0, 0, :, :] += 0.01 / (100 * 100)  # Distribute remaining
        peaked_prob = peaked_prob / peaked_prob.sum()  # Renormalize

        entropy = computer.compute_entropy(peaked_prob)

        # Peaked distribution should have low entropy
        assert entropy.item() < 1.0, \
            f"Peaked distribution should have low entropy, got {entropy.item()}"

    def test_entropy_toy_example(self):
        """Test entropy on known toy example."""
        from models.entropy_regularizer import ShannonEntropyComputer

        computer = ShannonEntropyComputer()

        # Simple 2x2 distribution with known entropy
        # H = -sum(p * log(p)) for p = [0.25, 0.25, 0.25, 0.25] = log(4) = 1.386
        toy_prob = torch.ones(1, 1, 2, 2) * 0.25

        entropy = computer.compute_entropy(toy_prob)

        expected_entropy = np.log(4)
        assert abs(entropy.item() - expected_entropy) < 0.01, \
            f"Expected entropy {expected_entropy}, got {entropy.item()}"

    def test_entropy_batch_processing(self):
        """Test entropy computation on batches."""
        from models.entropy_regularizer import ShannonEntropyComputer

        computer = ShannonEntropyComputer()

        # Create batch of probability maps
        batch_size = 8
        prob_maps = torch.rand(batch_size, 1, 50, 50)
        prob_maps = prob_maps / prob_maps.sum(dim=(2, 3), keepdim=True)

        entropy = computer.compute_entropy(prob_maps)

        # Should return single scalar (mean across batch)
        assert entropy.shape == torch.Size([])
        assert entropy.item() > 0, "Entropy should be positive"

    def test_entropy_zero_handling(self):
        """Test that eps parameter prevents log(0) errors."""
        from models.entropy_regularizer import ShannonEntropyComputer

        computer = ShannonEntropyComputer(eps=1e-8)

        # Create probability map with zeros
        prob_map = torch.zeros(1, 1, 10, 10)
        prob_map[0, 0, 5, 5] = 1.0  # One pixel has all probability

        # Should not raise error due to eps
        entropy = computer.compute_entropy(prob_map)
        assert torch.isfinite(entropy), "Entropy should be finite"


class TestBiasMapExtractor:
    """Test suite for bias map extraction from models."""

    def test_bias_extractor_initialization(self):
        """Test that bias extractor initializes correctly."""
        from models.entropy_regularizer import BiasMapExtractor, UniformImageGenerator

        # Create mock model
        class MockModel(torch.nn.Module):
            def forward(self, x):
                # Return mock saliency maps
                batch_size = x.shape[0]
                return torch.randn(batch_size, 1, 768, 1024)

        model = MockModel()
        generator = UniformImageGenerator(size=(1024, 768), device='cpu')
        extractor = BiasMapExtractor(model, generator, device='cpu')

        assert extractor.model is model
        assert extractor.uniform_generator is generator

    def test_extract_bias_map_shape(self):
        """Test that extracted bias map has correct shape."""
        from models.entropy_regularizer import BiasMapExtractor, UniformImageGenerator

        # Create mock model
        class MockModel(torch.nn.Module):
            def forward(self, x):
                batch_size = x.shape[0]
                return torch.randn(batch_size, 1, 768, 1024)

        model = MockModel()
        generator = UniformImageGenerator(size=(1024, 768), device='cpu')
        extractor = BiasMapExtractor(model, generator, device='cpu')

        bias_map = extractor.extract_bias_map(num_samples=16)

        # Averaged bias map should be (1, 1, 768, 1024)
        assert bias_map.shape == (1, 1, 768, 1024), \
            f"Expected shape (1, 1, 768, 1024), got {bias_map.shape}"

    def test_extract_bias_map_averaging(self):
        """Test that bias maps are averaged correctly."""
        from models.entropy_regularizer import BiasMapExtractor, UniformImageGenerator

        # Create deterministic model
        class DeterministicModel(torch.nn.Module):
            def forward(self, x):
                batch_size = x.shape[0]
                # Return constant maps
                return torch.ones(batch_size, 1, 768, 1024) * 5.0

        model = DeterministicModel()
        generator = UniformImageGenerator(size=(1024, 768), device='cpu')
        extractor = BiasMapExtractor(model, generator, device='cpu')

        bias_map = extractor.extract_bias_map(num_samples=16)

        # Average of all 5.0 should be 5.0
        assert torch.allclose(bias_map, torch.tensor(5.0)), \
            "Averaged bias map should be close to 5.0"

    def test_model_eval_mode(self):
        """Test that model is put in eval mode during extraction."""
        from models.entropy_regularizer import BiasMapExtractor, UniformImageGenerator

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.was_eval = False

            def forward(self, x):
                self.was_eval = not self.training
                return torch.randn(x.shape[0], 1, 768, 1024)

            def eval(self):
                self.training = False
                return self

        model = MockModel()
        model.training = True  # Start in train mode

        generator = UniformImageGenerator(size=(1024, 768), device='cpu')
        extractor = BiasMapExtractor(model, generator, device='cpu')

        _ = extractor.extract_bias_map(num_samples=4)

        assert model.was_eval, "Model should be in eval mode during extraction"


class TestEntropyRegularizer:
    """Test suite for complete entropy regularizer."""

    def test_regularizer_initialization(self):
        """Test that entropy regularizer initializes correctly."""
        from models.entropy_regularizer import EntropyRegularizer

        # Create mock model
        class MockModel(torch.nn.Module):
            def forward(self, x):
                return torch.randn(x.shape[0], 1, 768, 1024)

        model = MockModel()
        regularizer = EntropyRegularizer(
            model,
            image_size=(1024, 768),
            num_samples=16,
            device='cpu'
        )

        assert regularizer.model is model
        assert regularizer.num_samples == 16
        assert regularizer.device == 'cpu'

    def test_compute_entropy_loss(self):
        """Test computation of entropy loss."""
        from models.entropy_regularizer import EntropyRegularizer

        # Create mock model
        class MockModel(torch.nn.Module):
            def forward(self, x):
                return torch.randn(x.shape[0], 1, 768, 1024)
            def eval(self):
                return self

        model = MockModel()
        regularizer = EntropyRegularizer(
            model,
            image_size=(1024, 768),
            num_samples=16,
            device='cpu'
        )

        entropy_loss, entropy_value = regularizer.compute_entropy_loss()

        # Entropy loss should be negative entropy
        assert entropy_loss.item() < 0, "Entropy loss should be negative"
        assert entropy_value > 0, "Entropy value should be positive"

        # They should be negatives of each other
        assert abs(entropy_loss.item() + entropy_value) < 1e-5

    def test_forward_method(self):
        """Test that forward method returns entropy loss."""
        from models.entropy_regularizer import EntropyRegularizer

        class MockModel(torch.nn.Module):
            def forward(self, x):
                return torch.randn(x.shape[0], 1, 768, 1024)
            def eval(self):
                return self

        model = MockModel()
        regularizer = EntropyRegularizer(model, device='cpu')

        result = regularizer()

        assert isinstance(result, tuple)
        assert len(result) == 2
        entropy_loss, entropy_value = result

        assert isinstance(entropy_loss, torch.Tensor)
        assert isinstance(entropy_value, (int, float))

    def test_gradient_flow(self):
        """Test that gradients can flow through entropy loss."""
        from models.entropy_regularizer import EntropyRegularizer

        # Create model with learnable parameters
        class LearnableModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1, 1, 768, 1024))

            def forward(self, x):
                return self.weight.expand(x.shape[0], -1, -1, -1)

            def eval(self):
                return self

        model = LearnableModel()
        regularizer = EntropyRegularizer(model, device='cpu', num_samples=4)

        # Compute loss
        entropy_loss, _ = regularizer.compute_entropy_loss()

        # Backward pass
        entropy_loss.backward()

        # Check that gradients exist
        assert model.weight.grad is not None
        assert not torch.all(model.weight.grad == 0), \
            "Gradients should be non-zero"
