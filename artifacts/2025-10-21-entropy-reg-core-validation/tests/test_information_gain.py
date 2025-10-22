"""
Tests for Information Gain (IG) computation.

Tests cover:
- IG computation with Gaussian center prior
- Perfect prediction cases (IG should approach maximum)
- Uniform prediction cases (IG should be low)
- Edge cases (zero fixations, uniform distributions)
- Numerical stability
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestInformationGain:
    """Test Information Gain metric computation."""

    def test_information_gain_perfect_prediction(self):
        """Test IG for perfect prediction (should be high)."""
        from evaluation.metrics import information_gain

        # Create fixation distribution
        np.random.seed(42)
        fixations = np.random.rand(100, 100)
        fixations = fixations / fixations.sum()

        # Perfect prediction = fixations themselves
        ig = information_gain(fixations, fixations, baseline='center')

        # IG should be positive and substantial for perfect prediction
        assert ig > 0.5, f"Expected IG > 0.5 for perfect prediction, got {ig:.4f}"

    def test_information_gain_uniform_prediction(self):
        """Test IG for uniform prediction (should be negative, worse than baseline)."""
        from evaluation.metrics import information_gain

        # Create fixation distribution (concentrated)
        fixations = np.zeros((100, 100))
        fixations[45:55, 45:55] = 1.0
        fixations = fixations / fixations.sum()

        # Uniform prediction (no information)
        uniform_pred = np.ones((100, 100)) / 10000

        ig = information_gain(fixations, uniform_pred, baseline='center')

        # IG should be negative for uniform prediction (worse than center baseline)
        assert ig < 0, f"Expected negative IG for uniform prediction (worse than baseline), got {ig:.4f}"

    def test_information_gain_better_than_baseline(self):
        """Test that better prediction has higher IG than worse prediction."""
        from evaluation.metrics import information_gain

        # Ground truth fixations (concentrated in center)
        fixations = np.zeros((100, 100))
        fixations[40:60, 40:60] = 1.0
        fixations = fixations / fixations.sum()

        # Good prediction (also concentrated in center)
        good_pred = np.zeros((100, 100))
        good_pred[42:58, 42:58] = 1.0
        good_pred = good_pred / good_pred.sum()

        # Bad prediction (concentrated elsewhere)
        bad_pred = np.zeros((100, 100))
        bad_pred[10:30, 10:30] = 1.0
        bad_pred = bad_pred / bad_pred.sum()

        ig_good = information_gain(fixations, good_pred, baseline='center')
        ig_bad = information_gain(fixations, bad_pred, baseline='center')

        assert ig_good > ig_bad, \
            f"Expected better prediction to have higher IG. Good: {ig_good:.4f}, Bad: {ig_bad:.4f}"

    def test_information_gain_with_center_prior(self):
        """Test IG computation with Gaussian center prior."""
        from evaluation.metrics import information_gain, create_center_prior

        # Create fixations
        fixations = np.zeros((100, 100))
        fixations[40:60, 40:60] = 1.0
        fixations = fixations / fixations.sum()

        # Create prediction
        prediction = np.random.rand(100, 100)
        prediction = prediction / prediction.sum()

        # Compute IG with center prior
        ig = information_gain(fixations, prediction, baseline='center')

        # IG should be a finite number
        assert np.isfinite(ig), f"IG should be finite, got {ig}"

    def test_information_gain_shape_mismatch(self):
        """Test that IG raises error for shape mismatch."""
        from evaluation.metrics import information_gain

        fixations = np.random.rand(100, 100)
        prediction = np.random.rand(50, 50)

        with pytest.raises(ValueError, match="shape"):
            information_gain(fixations, prediction, baseline='center')

    def test_information_gain_numerical_stability(self):
        """Test IG computation handles zero probabilities correctly."""
        from evaluation.metrics import information_gain

        # Create fixations with some zeros
        fixations = np.random.rand(100, 100)
        fixations[fixations < 0.5] = 0  # Introduce zeros
        fixations = fixations / fixations.sum()

        # Create prediction with some zeros
        prediction = np.random.rand(100, 100)
        prediction[prediction < 0.5] = 0
        prediction = prediction / prediction.sum()

        # Should not raise error or return NaN
        ig = information_gain(fixations, prediction, baseline='center')

        assert np.isfinite(ig), f"IG should handle zeros without NaN/Inf, got {ig}"


class TestCenterPrior:
    """Test Gaussian center prior generation."""

    def test_create_center_prior_shape(self):
        """Test center prior has correct shape."""
        from evaluation.metrics import create_center_prior

        prior = create_center_prior(height=100, width=150)

        assert prior.shape == (100, 150), \
            f"Expected shape (100, 150), got {prior.shape}"

    def test_create_center_prior_normalization(self):
        """Test center prior sums to 1."""
        from evaluation.metrics import create_center_prior

        prior = create_center_prior(height=100, width=100)

        assert np.isclose(prior.sum(), 1.0, atol=1e-6), \
            f"Prior should sum to 1, got {prior.sum():.10f}"

    def test_center_prior_peak_at_center(self):
        """Test center prior has maximum value at center."""
        from evaluation.metrics import create_center_prior

        height, width = 100, 100
        prior = create_center_prior(height=height, width=width)

        center_y, center_x = height // 2, width // 2
        center_value = prior[center_y, center_x]

        # Center should have highest or near-highest value
        assert center_value >= np.percentile(prior, 95), \
            f"Center value should be in top 5%, got {center_value:.6f} vs max {prior.max():.6f}"

    def test_center_prior_configurable_sigma(self):
        """Test center prior with different sigma values."""
        from evaluation.metrics import create_center_prior

        prior_narrow = create_center_prior(100, 100, sigma=10)
        prior_wide = create_center_prior(100, 100, sigma=30)

        # Narrow prior should be more concentrated (higher peak)
        assert prior_narrow.max() > prior_wide.max(), \
            f"Narrow prior should have higher peak. Narrow: {prior_narrow.max():.6f}, Wide: {prior_wide.max():.6f}"


class TestKLDivergence:
    """Test KL divergence computation."""

    def test_kl_divergence_identical_distributions(self):
        """Test KL divergence is zero for identical distributions."""
        from evaluation.metrics import kl_divergence

        p = np.random.rand(100, 100)
        p = p / p.sum()

        kl = kl_divergence(p, p)

        assert np.isclose(kl, 0.0, atol=1e-6), \
            f"KL(P||P) should be 0, got {kl:.10f}"

    def test_kl_divergence_asymmetric(self):
        """Test KL divergence is asymmetric: KL(P||Q) ≠ KL(Q||P)."""
        from evaluation.metrics import kl_divergence

        p = np.random.rand(100, 100)
        p = p / p.sum()

        q = np.random.rand(100, 100)
        q = q / q.sum()

        kl_pq = kl_divergence(p, q)
        kl_qp = kl_divergence(q, p)

        # Should be different (unless by rare chance they're symmetric)
        assert not np.isclose(kl_pq, kl_qp, atol=1e-3), \
            f"KL(P||Q) and KL(Q||P) should differ. Got {kl_pq:.6f} and {kl_qp:.6f}"

    def test_kl_divergence_non_negative(self):
        """Test KL divergence is always non-negative."""
        from evaluation.metrics import kl_divergence

        np.random.seed(42)
        for _ in range(10):
            p = np.random.rand(50, 50)
            p = p / p.sum()

            q = np.random.rand(50, 50)
            q = q / q.sum()

            kl = kl_divergence(p, q)

            assert kl >= -1e-6, f"KL divergence should be non-negative, got {kl:.10f}"

    def test_kl_divergence_numerical_stability(self):
        """Test KL divergence handles zeros correctly with epsilon."""
        from evaluation.metrics import kl_divergence

        # Create distributions with zeros
        p = np.random.rand(100, 100)
        p[p < 0.8] = 0  # Many zeros
        p = p / p.sum()

        q = np.random.rand(100, 100)
        q[q < 0.8] = 0
        q = q / q.sum()

        kl = kl_divergence(p, q, epsilon=1e-10)

        assert np.isfinite(kl), f"KL should handle zeros without NaN/Inf, got {kl}"


class TestInformationGainBatch:
    """Test batch computation of Information Gain."""

    def test_information_gain_batch(self):
        """Test IG computation on batch of images."""
        from evaluation.metrics import information_gain_batch

        batch_size = 4
        height, width = 100, 100

        # Create batch of fixations and predictions
        fixations = np.random.rand(batch_size, height, width)
        predictions = np.random.rand(batch_size, height, width)

        # Normalize each sample
        for i in range(batch_size):
            fixations[i] = fixations[i] / fixations[i].sum()
            predictions[i] = predictions[i] / predictions[i].sum()

        ig_values = information_gain_batch(fixations, predictions, baseline='center')

        assert len(ig_values) == batch_size, \
            f"Expected {batch_size} IG values, got {len(ig_values)}"
        assert all(np.isfinite(ig_values)), \
            f"All IG values should be finite, got {ig_values}"

    def test_information_gain_statistics(self):
        """Test computing mean and std of IG across batch."""
        from evaluation.metrics import information_gain_batch

        batch_size = 10
        height, width = 100, 100

        fixations = np.random.rand(batch_size, height, width)
        predictions = np.random.rand(batch_size, height, width)

        for i in range(batch_size):
            fixations[i] = fixations[i] / fixations[i].sum()
            predictions[i] = predictions[i] / predictions[i].sum()

        ig_values = information_gain_batch(fixations, predictions, baseline='center')

        mean_ig = np.mean(ig_values)
        std_ig = np.std(ig_values)

        assert np.isfinite(mean_ig), f"Mean IG should be finite, got {mean_ig}"
        assert np.isfinite(std_ig), f"Std IG should be finite, got {std_ig}"
        assert std_ig >= 0, f"Std should be non-negative, got {std_ig}"
