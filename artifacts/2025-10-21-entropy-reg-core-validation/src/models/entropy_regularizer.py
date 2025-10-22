"""
Entropy Regularization Module for DeepGaze 3

This module implements entropy regularization to encourage more uniform bias maps
in saliency models. The approach involves:
1. Generating uniform images with constant pixel values
2. Extracting bias maps from the model using these uniform images
3. Computing Shannon entropy of the bias maps
4. Using entropy as a regularization term in the loss function

Reference: Technical specification for entropy-reg-core-validation experiment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class UniformImageGenerator:
    """Generates uniform images with constant pixel values for bias extraction.

    Uniform images are used to extract the center bias component of saliency models
    by passing images with no semantic content through the model.
    """

    def __init__(self, size=(1024, 768), device='cuda'):
        """Initialize uniform image generator.

        Args:
            size: Tuple of (width, height) for generated images
            device: Device to create tensors on ('cuda' or 'cpu')
        """
        self.size = size
        self.device = device

    def generate(self, batch_size=16, intensity_values=None):
        """Generate batch of uniform images.

        Args:
            batch_size: Number of uniform images to generate
            intensity_values: List of intensity values to use (0.0-1.0).
                             If None, uses [0.0, 0.5, 1.0]

        Returns:
            Tensor of shape (batch_size, 3, height, width) with uniform pixel values
        """
        if intensity_values is None:
            intensity_values = [0.0, 0.5, 1.0]

        uniform_images = []
        for _ in range(batch_size):
            # Randomly select an intensity value
            intensity = np.random.choice(intensity_values)
            # Create image with constant value across all pixels and channels
            img = torch.full((3, self.size[1], self.size[0]), intensity,
                           dtype=torch.float32, device=self.device)
            uniform_images.append(img)

        return torch.stack(uniform_images)


class ShannonEntropyComputer:
    """Computes Shannon entropy of saliency/bias maps.

    Shannon entropy H = -Σ(p * log(p)) measures the uniformity of a probability
    distribution. Higher entropy indicates more uniform (less biased) predictions.
    """

    def __init__(self, eps=1e-8):
        """Initialize entropy computer.

        Args:
            eps: Small constant to avoid log(0) numerical issues
        """
        self.eps = eps

    def normalize_to_probability(self, saliency_map):
        """Normalize saliency map to valid probability distribution.

        Args:
            saliency_map: Tensor of shape (batch, 1, height, width) with log-densities

        Returns:
            Normalized probability distribution of same shape
        """
        # Flatten spatial dimensions
        batch_size = saliency_map.shape[0]
        flat_map = saliency_map.view(batch_size, -1)

        # Convert log-density to probability using softmax
        # This ensures: sum(p) = 1 and p >= 0
        prob_map = F.softmax(flat_map, dim=1)

        # Reshape back to spatial dimensions
        prob_map = prob_map.view_as(saliency_map)

        return prob_map

    def compute_entropy(self, probability_map):
        """Compute Shannon entropy: H = -Σ(p * log(p))

        Args:
            probability_map: Normalized probability distribution

        Returns:
            Scalar tensor with entropy value (bits)
        """
        # Flatten spatial dimensions
        flat_prob = probability_map.view(probability_map.shape[0], -1)

        # Compute H = -Σ(p * log(p))
        # Add eps to avoid log(0)
        log_prob = torch.log(flat_prob + self.eps)
        entropy = -torch.sum(flat_prob * log_prob, dim=1)

        # Average across batch
        mean_entropy = torch.mean(entropy)

        return mean_entropy


class BiasMapExtractor:
    """Extracts bias maps from DeepGaze model using uniform images.

    The bias map represents the model's spatial prior independent of image content.
    It's extracted by passing uniform (contentless) images through the model.
    """

    def __init__(self, model, uniform_generator, device='cuda'):
        """Initialize bias map extractor.

        Args:
            model: DeepGaze model to extract bias from
            uniform_generator: UniformImageGenerator instance
            device: Device for computation
        """
        self.model = model
        self.uniform_generator = uniform_generator
        self.device = device

    def extract_bias_map(self, num_samples=16):
        """Extract bias map by averaging predictions on uniform images.

        Args:
            num_samples: Number of uniform images to average over

        Returns:
            Averaged bias map tensor of shape (1, 1, height, width)
        """
        self.model.eval()

        # Note: No torch.no_grad() here to allow gradient flow for training
        # Generate uniform images
        uniform_images = self.uniform_generator.generate(batch_size=num_samples)

        # Get model predictions (log-densities)
        # For DeepGaze III in saliency-only mode, use dummy fixations
        batch_size = uniform_images.size(0)
        height, width = uniform_images.size(2), uniform_images.size(3)
        centerbias = torch.zeros(batch_size, 1, height, width, device=uniform_images.device)
        # 4 dummy fixations at center (DeepGaze 3 expects 4 previous fixations)
        x_hist = torch.full((batch_size, 4), width / 2.0, device=uniform_images.device)
        y_hist = torch.full((batch_size, 4), height / 2.0, device=uniform_images.device)
        bias_maps = self.model(uniform_images, centerbias, x_hist, y_hist)

        # Average across all samples
        avg_bias_map = torch.mean(bias_maps, dim=0, keepdim=True)

        return avg_bias_map


class EntropyRegularizer(nn.Module):
    """Complete entropy regularization module.

    Combines uniform image generation, bias extraction, and entropy computation
    to provide regularization loss that encourages more uniform bias maps.
    """

    def __init__(self, model, image_size=(1024, 768), num_samples=16, device='cuda'):
        """Initialize entropy regularizer.

        Args:
            model: DeepGaze model to regularize
            image_size: Size of input images (width, height)
            num_samples: Number of uniform images for bias extraction
            device: Device for computation
        """
        super().__init__()

        self.model = model
        self.num_samples = num_samples
        self.device = device

        self.uniform_generator = UniformImageGenerator(size=image_size, device=device)
        self.entropy_computer = ShannonEntropyComputer()
        self.bias_extractor = BiasMapExtractor(model, self.uniform_generator, device)

    def compute_entropy_loss(self):
        """Compute entropy regularization loss.

        Returns:
            Tuple of (entropy_loss, entropy_value)
            - entropy_loss: Negative entropy (to be minimized)
            - entropy_value: Actual entropy value (for logging)
        """
        # Extract bias map from model
        bias_map = self.bias_extractor.extract_bias_map(num_samples=self.num_samples)

        # Normalize to probability distribution
        prob_map = self.entropy_computer.normalize_to_probability(bias_map)

        # Compute Shannon entropy
        entropy = self.entropy_computer.compute_entropy(prob_map)

        # Return negative entropy as loss (maximizing entropy = minimizing -entropy)
        entropy_loss = -entropy

        return entropy_loss, entropy.item()

    def forward(self):
        """Forward pass computes entropy loss."""
        return self.compute_entropy_loss()
