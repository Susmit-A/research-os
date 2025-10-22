"""
Evaluation metrics for saliency prediction.

Implements:
- Information Gain (IG) metric with Gaussian center prior
- Shannon entropy computation
- KL divergence computation
- Bias entropy measurement from uniform images
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Union, List, Tuple
from scipy.ndimage import gaussian_filter


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute KL divergence KL(P || Q) = sum(P * log(P / Q)).

    Parameters
    ----------
    p : np.ndarray
        First probability distribution, shape (H, W) or flattened
    q : np.ndarray
        Second probability distribution, shape (H, W) or flattened
    epsilon : float, default=1e-10
        Small constant for numerical stability

    Returns
    -------
    kl : float
        KL divergence value (non-negative)

    Examples
    --------
    >>> p = np.random.rand(100, 100)
    >>> p = p / p.sum()
    >>> q = np.random.rand(100, 100)
    >>> q = q / q.sum()
    >>> kl = kl_divergence(p, q)
    >>> print(f"KL(P||Q) = {kl:.4f}")
    """
    # Flatten arrays
    p = p.flatten()
    q = q.flatten()

    # Add epsilon for numerical stability
    p = p + epsilon
    q = q + epsilon

    # Re-normalize after adding epsilon
    p = p / p.sum()
    q = q / q.sum()

    # Compute KL divergence
    kl = np.sum(p * np.log(p / q))

    return float(kl)


def shannon_entropy(probability_map: np.ndarray) -> float:
    """
    Compute Shannon entropy H(P) = -∑ P(i) * log(P(i)).

    Parameters
    ----------
    probability_map : np.ndarray
        Probability distribution, shape (H, W), must sum to 1

    Returns
    -------
    H : float
        Shannon entropy in nats (natural logarithm)

    Raises
    ------
    ValueError
        If probability_map does not sum to 1

    Examples
    --------
    >>> uniform = np.ones((100, 100)) / 10000
    >>> H = shannon_entropy(uniform)
    >>> print(f"Entropy: {H:.4f} nats")  # Should be log(10000) ≈ 9.21
    """
    # Validate normalization
    if not np.isclose(probability_map.sum(), 1.0, atol=1e-5):
        raise ValueError(
            f"Probability map must sum to 1, got {probability_map.sum():.10f}"
        )

    # Flatten to 1D
    p = probability_map.flatten()

    # Remove zeros (log(0) undefined)
    p = p[p > 0]

    # Shannon entropy
    H = -np.sum(p * np.log(p))

    return float(H)


def normalized_entropy(probability_map: np.ndarray) -> float:
    """
    Compute normalized Shannon entropy H / H_max where H_max = log(N).

    Parameters
    ----------
    probability_map : np.ndarray
        Probability distribution, shape (H, W), must sum to 1

    Returns
    -------
    H_norm : float
        Normalized entropy between 0 and 1

    Examples
    --------
    >>> uniform = np.ones((100, 100)) / 10000
    >>> H_norm = normalized_entropy(uniform)
    >>> print(f"Normalized entropy: {H_norm:.4f}")  # Should be ≈ 1.0
    """
    H = shannon_entropy(probability_map)
    H_max = np.log(probability_map.size)
    return float(H / H_max)


def create_center_prior(
    height: int,
    width: int,
    sigma: float = None
) -> np.ndarray:
    """
    Create Gaussian center prior (baseline for Information Gain).

    Parameters
    ----------
    height : int
        Image height in pixels
    width : int
        Image width in pixels
    sigma : float, optional
        Gaussian sigma. If None, uses min(height, width) / 8

    Returns
    -------
    prior : np.ndarray
        Center prior probability distribution, shape (height, width), sums to 1

    Examples
    --------
    >>> prior = create_center_prior(100, 150)
    >>> print(prior.shape)  # (100, 150)
    >>> print(prior.sum())  # 1.0
    """
    if sigma is None:
        sigma = min(height, width) / 8.0

    # Create coordinate grids
    y = np.arange(height)
    x = np.arange(width)
    Y, X = np.meshgrid(y, x, indexing='ij')

    # Center coordinates
    cy = height / 2.0
    cx = width / 2.0

    # Gaussian centered at image center
    prior = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))

    # Normalize to sum to 1
    prior = prior / prior.sum()

    return prior


def information_gain(
    fixation_map: np.ndarray,
    prediction: np.ndarray,
    baseline: Union[str, np.ndarray] = 'center',
    epsilon: float = 1e-10
) -> float:
    """
    Compute Information Gain metric.

    IG = KL(fixations || prediction) - KL(fixations || baseline)

    Parameters
    ----------
    fixation_map : np.ndarray
        Ground truth fixation density map, shape (H, W)
    prediction : np.ndarray
        Model's predicted saliency map, shape (H, W)
    baseline : str or np.ndarray, default='center'
        Baseline distribution. If 'center', uses Gaussian center prior.
        If 'uniform', uses uniform distribution. If array, uses directly.
    epsilon : float, default=1e-10
        Small constant for numerical stability

    Returns
    -------
    ig : float
        Information gain value

    Raises
    ------
    ValueError
        If fixation_map and prediction have different shapes

    Examples
    --------
    >>> fixations = np.random.rand(100, 100)
    >>> fixations = fixations / fixations.sum()
    >>> prediction = np.random.rand(100, 100)
    >>> prediction = prediction / prediction.sum()
    >>> ig = information_gain(fixations, prediction, baseline='center')
    >>> print(f"IG: {ig:.4f}")
    """
    # Validate shapes
    if fixation_map.shape != prediction.shape:
        raise ValueError(
            f"Fixation map and prediction must have same shape. "
            f"Got {fixation_map.shape} and {prediction.shape}"
        )

    height, width = fixation_map.shape

    # Normalize inputs to probability distributions
    fixation_map = fixation_map / (fixation_map.sum() + epsilon)
    prediction = prediction / (prediction.sum() + epsilon)

    # Create or use baseline
    if baseline == 'center':
        baseline_dist = create_center_prior(height, width)
    elif baseline == 'uniform':
        baseline_dist = np.ones((height, width)) / (height * width)
    elif isinstance(baseline, np.ndarray):
        baseline_dist = baseline / (baseline.sum() + epsilon)
    else:
        raise ValueError(
            f"Baseline must be 'center', 'uniform', or array. Got {baseline}"
        )

    # Compute KL divergences
    kl_pred = kl_divergence(fixation_map, prediction, epsilon=epsilon)
    kl_baseline = kl_divergence(fixation_map, baseline_dist, epsilon=epsilon)

    # Information Gain
    ig = kl_baseline - kl_pred

    return float(ig)


def information_gain_batch(
    fixation_maps: np.ndarray,
    predictions: np.ndarray,
    baseline: Union[str, np.ndarray] = 'center',
    epsilon: float = 1e-10
) -> np.ndarray:
    """
    Compute Information Gain for batch of images.

    Parameters
    ----------
    fixation_maps : np.ndarray
        Ground truth fixation density maps, shape (N, H, W)
    predictions : np.ndarray
        Model's predicted saliency maps, shape (N, H, W)
    baseline : str or np.ndarray, default='center'
        Baseline distribution
    epsilon : float, default=1e-10
        Small constant for numerical stability

    Returns
    -------
    ig_values : np.ndarray
        Information gain values for each image, shape (N,)

    Examples
    --------
    >>> fixations = np.random.rand(10, 100, 100)
    >>> predictions = np.random.rand(10, 100, 100)
    >>> ig_values = information_gain_batch(fixations, predictions)
    >>> print(f"Mean IG: {ig_values.mean():.4f}")
    >>> print(f"Std IG: {ig_values.std():.4f}")
    """
    batch_size = fixation_maps.shape[0]
    ig_values = np.zeros(batch_size)

    for i in range(batch_size):
        ig_values[i] = information_gain(
            fixation_maps[i],
            predictions[i],
            baseline=baseline,
            epsilon=epsilon
        )

    return ig_values


def extract_bias_maps(
    model: nn.Module,
    image_size: Tuple[int, int] = (1024, 768),
    num_samples: int = 16,
    intensities: List[float] = None,
    device: str = 'cuda'
) -> List[np.ndarray]:
    """
    Extract bias maps from model using uniform images.

    Parameters
    ----------
    model : nn.Module
        Saliency prediction model
    image_size : Tuple[int, int], default=(1024, 768)
        Image size as (width, height)
    num_samples : int, default=16
        Number of uniform samples per intensity
    intensities : List[float], optional
        Uniform intensity values. If None, uses [0.0, 0.5, 1.0]
    device : str, default='cuda'
        Device to run model on

    Returns
    -------
    bias_maps : List[np.ndarray]
        List of bias maps, each shape (height, width), normalized to sum to 1

    Examples
    --------
    >>> from models.deepgaze3 import DeepGazeIII
    >>> model = DeepGazeIII()
    >>> model.eval()
    >>> bias_maps = extract_bias_maps(model, num_samples=4, device='cpu')
    >>> print(f"Extracted {len(bias_maps)} bias maps")
    """
    if intensities is None:
        intensities = [0.0, 0.5, 1.0]

    model = model.to(device)
    model.eval()

    bias_maps = []
    width, height = image_size

    # Create uniform centerbias (required by DeepGaze model)
    centerbias = torch.ones(1, height, width) / (height * width)
    centerbias = centerbias.to(device)

    with torch.no_grad():
        for intensity in intensities:
            for _ in range(num_samples):
                # Create uniform image
                uniform_image = torch.ones(1, 3, height, width) * intensity
                uniform_image = uniform_image.to(device)

                # Forward pass (model outputs log-density predictions)
                # DeepGaze model requires centerbias and scanpath history
                # For saliency-only mode, pass None for scanpath parameters
                output = model(uniform_image, centerbias, x_hist=None, y_hist=None)

                # Convert to probability distribution
                # Assuming output is log-density, apply softmax
                output_np = output.cpu().numpy().squeeze()

                # Apply softmax normalization (exp + normalize)
                bias_map = np.exp(output_np - output_np.max())  # Subtract max for stability
                bias_map = bias_map / bias_map.sum()

                bias_maps.append(bias_map)

    return bias_maps


def average_bias_maps(bias_maps: List[np.ndarray]) -> np.ndarray:
    """
    Average multiple bias maps.

    Parameters
    ----------
    bias_maps : List[np.ndarray]
        List of bias maps to average

    Returns
    -------
    avg_bias : np.ndarray
        Averaged bias map, normalized to sum to 1

    Examples
    --------
    >>> bias_maps = [np.random.rand(100, 100) for _ in range(5)]
    >>> bias_maps = [bm / bm.sum() for bm in bias_maps]
    >>> avg_bias = average_bias_maps(bias_maps)
    >>> print(avg_bias.shape)  # (100, 100)
    >>> print(avg_bias.sum())  # 1.0
    """
    # Stack and average
    stacked = np.stack(bias_maps, axis=0)
    avg_bias = stacked.mean(axis=0)

    # Re-normalize to sum to 1
    avg_bias = avg_bias / avg_bias.sum()

    return avg_bias


def measure_bias_entropy(
    model: nn.Module,
    image_size: Tuple[int, int] = (1024, 768),
    num_samples: int = 16,
    intensities: List[float] = None,
    device: str = 'cuda'
) -> float:
    """
    Measure bias entropy of model using uniform images.

    Parameters
    ----------
    model : nn.Module
        Saliency prediction model
    image_size : Tuple[int, int], default=(1024, 768)
        Image size as (width, height)
    num_samples : int, default=16
        Number of uniform samples per intensity
    intensities : List[float], optional
        Uniform intensity values. If None, uses [0.0, 0.5, 1.0]
    device : str, default='cuda'
        Device to run model on

    Returns
    -------
    entropy : float
        Shannon entropy of averaged bias map in nats

    Examples
    --------
    >>> from models.deepgaze3 import DeepGazeIII
    >>> model = DeepGazeIII()
    >>> model.eval()
    >>> entropy = measure_bias_entropy(model, num_samples=16, device='cpu')
    >>> print(f"Bias entropy: {entropy:.4f} nats")
    """
    # Extract bias maps
    bias_maps = extract_bias_maps(
        model,
        image_size=image_size,
        num_samples=num_samples,
        intensities=intensities,
        device=device
    )

    # Average bias maps
    avg_bias = average_bias_maps(bias_maps)

    # Compute Shannon entropy
    entropy = shannon_entropy(avg_bias)

    return entropy


def compare_bias_entropy(
    model1: nn.Module,
    model2: nn.Module,
    image_size: Tuple[int, int] = (1024, 768),
    num_samples: int = 16,
    device: str = 'cuda'
) -> Tuple[float, float, float]:
    """
    Compare bias entropy between two models.

    Parameters
    ----------
    model1 : nn.Module
        First model (e.g., baseline)
    model2 : nn.Module
        Second model (e.g., entropy-regularized)
    image_size : Tuple[int, int], default=(1024, 768)
        Image size as (width, height)
    num_samples : int, default=16
        Number of uniform samples
    device : str, default='cuda'
        Device to run models on

    Returns
    -------
    entropy1 : float
        Bias entropy of model1
    entropy2 : float
        Bias entropy of model2
    percent_increase : float
        Percentage increase: (entropy2 - entropy1) / entropy1 * 100

    Examples
    --------
    >>> baseline_model = load_model('baseline.pt')
    >>> entropy_model = load_model('entropy_reg.pt')
    >>> e1, e2, increase = compare_bias_entropy(baseline_model, entropy_model)
    >>> print(f"Baseline: {e1:.4f}, Entropy-reg: {e2:.4f}, Increase: {increase:.2f}%")
    """
    entropy1 = measure_bias_entropy(model1, image_size, num_samples, device=device)
    entropy2 = measure_bias_entropy(model2, image_size, num_samples, device=device)

    percent_increase = ((entropy2 - entropy1) / entropy1) * 100.0

    return entropy1, entropy2, percent_increase
