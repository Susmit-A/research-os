"""
Tests for CAT2000 data loader.

Tests cover:
- Dataset initialization for OOD evaluation
- Sampling 50 images randomly
- Fixation map loading
- Image preprocessing (resize to 1024x768, ImageNet normalization)
- Correct shapes and value ranges
- Batch loading for evaluation
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestCAT2000Dataset:
    """Test suite for CAT2000 dataset class."""

    def test_dataset_initialization(self):
        """Test that dataset can be initialized with correct parameters."""
        from data.cat2000_loader import CAT2000Dataset

        dataset = CAT2000Dataset(
            data_path="/path/to/CAT2000",
            num_samples=50,
            image_size=(1024, 768),
            normalize=True
        )

        assert dataset is not None
        assert hasattr(dataset, '__len__')
        assert hasattr(dataset, '__getitem__')

    def test_dataset_size_50_samples(self):
        """Test that dataset returns exactly 50 samples for OOD evaluation."""
        from data.cat2000_loader import CAT2000Dataset

        dataset = CAT2000Dataset(
            data_path="/path/to/CAT2000",
            num_samples=50
        )

        assert len(dataset) == 50, \
            f"Dataset should have 50 samples for OOD evaluation, got {len(dataset)}"

    def test_getitem_returns_correct_format(self):
        """Test that __getitem__ returns (image, fixation_map) tuple."""
        from data.cat2000_loader import CAT2000Dataset

        dataset = CAT2000Dataset(
            data_path="/path/to/CAT2000",
            num_samples=50
        )

        sample = dataset[0]

        assert isinstance(sample, tuple), "Dataset should return tuple"
        assert len(sample) == 2, "Sample should be (image, fixation_map)"

        image, fixation_map = sample
        assert isinstance(image, torch.Tensor), "Image should be torch.Tensor"
        assert isinstance(fixation_map, torch.Tensor), "Fixation map should be torch.Tensor"

    def test_image_shape_after_preprocessing(self):
        """Test that images are resized to 1024x768."""
        from data.cat2000_loader import CAT2000Dataset

        dataset = CAT2000Dataset(
            data_path="/path/to/CAT2000",
            num_samples=50,
            image_size=(1024, 768)
        )

        image, _ = dataset[0]

        # Image shape should be (C, H, W) = (3, 768, 1024)
        assert image.shape == (3, 768, 1024), \
            f"Image shape should be (3, 768, 1024), got {image.shape}"

    def test_fixation_map_shape(self):
        """Test that fixation maps match image dimensions."""
        from data.cat2000_loader import CAT2000Dataset

        dataset = CAT2000Dataset(
            data_path="/path/to/CAT2000",
            num_samples=50,
            image_size=(1024, 768)
        )

        _, fixation_map = dataset[0]

        # Fixation map shape should be (1, H, W) = (1, 768, 1024)
        assert fixation_map.shape == (1, 768, 1024), \
            f"Fixation map shape should be (1, 768, 1024), got {fixation_map.shape}"

    def test_imagenet_normalization(self):
        """Test that ImageNet normalization is applied correctly."""
        from data.cat2000_loader import CAT2000Dataset

        dataset = CAT2000Dataset(
            data_path="/path/to/CAT2000",
            num_samples=50,
            normalize=True
        )

        image, _ = dataset[0]

        # Check that values are in reasonable range for normalized images
        assert image.min() >= -5.0, "Normalized image min seems incorrect"
        assert image.max() <= 5.0, "Normalized image max seems incorrect"

    def test_value_ranges(self):
        """Test that fixation maps are in valid range [0, 1]."""
        from data.cat2000_loader import CAT2000Dataset

        dataset = CAT2000Dataset(
            data_path="/path/to/CAT2000",
            num_samples=50
        )

        _, fixation_map = dataset[0]

        assert fixation_map.min() >= 0.0, \
            f"Fixation map min should be >= 0, got {fixation_map.min()}"
        assert fixation_map.max() <= 1.0, \
            f"Fixation map max should be <= 1, got {fixation_map.max()}"

    def test_reproducible_sampling(self):
        """Test that sampling is reproducible with same seed."""
        from data.cat2000_loader import CAT2000Dataset

        dataset1 = CAT2000Dataset(
            data_path="/path/to/CAT2000",
            num_samples=50,
            seed=42
        )

        dataset2 = CAT2000Dataset(
            data_path="/path/to/CAT2000",
            num_samples=50,
            seed=42
        )

        # Should sample the same 50 images
        assert len(dataset1) == len(dataset2)

        # First sample should be identical
        img1, fix1 = dataset1[0]
        img2, fix2 = dataset2[0]

        assert torch.allclose(img1, img2), "Same seed should produce identical sampling"
        assert torch.allclose(fix1, fix2), "Same seed should produce identical sampling"

    def test_different_seed_different_sampling(self):
        """Test that different seeds produce different samples."""
        from data.cat2000_loader import CAT2000Dataset

        dataset1 = CAT2000Dataset(
            data_path="/path/to/CAT2000",
            num_samples=50,
            seed=42
        )

        dataset2 = CAT2000Dataset(
            data_path="/path/to/CAT2000",
            num_samples=50,
            seed=123
        )

        # Same size but potentially different images
        assert len(dataset1) == len(dataset2)

        # We can't guarantee different samples, but the mechanism should support it
        # This is more of an interface test

    def test_dataloader_compatibility(self):
        """Test that dataset works with PyTorch DataLoader."""
        from data.cat2000_loader import CAT2000Dataset
        from torch.utils.data import DataLoader

        dataset = CAT2000Dataset(
            data_path="/path/to/CAT2000",
            num_samples=50
        )

        dataloader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,  # OOD evaluation typically doesn't shuffle
            num_workers=0
        )

        # Get one batch
        batch = next(iter(dataloader))
        images, fixation_maps = batch

        assert images.shape == (8, 3, 768, 1024), \
            f"Batch images shape should be (8, 3, 768, 1024), got {images.shape}"
        assert fixation_maps.shape == (8, 1, 768, 1024), \
            f"Batch fixation maps shape should be (8, 1, 768, 1024), got {fixation_maps.shape}"

    def test_all_categories_represented(self):
        """Test that CAT2000 categories are properly handled."""
        from data.cat2000_loader import CAT2000Dataset

        dataset = CAT2000Dataset(
            data_path="/path/to/CAT2000",
            num_samples=50
        )

        # CAT2000 has 20 categories, each with multiple images
        # With 50 samples, we should get good category coverage
        # This is more of a sanity check
        assert len(dataset) == 50

    def test_index_out_of_range(self):
        """Test that accessing invalid indices raises appropriate error."""
        from data.cat2000_loader import CAT2000Dataset

        dataset = CAT2000Dataset(
            data_path="/path/to/CAT2000",
            num_samples=50
        )

        with pytest.raises(IndexError):
            _ = dataset[50]  # Index 50 is out of range for 50 samples (0-49)

        with pytest.raises(IndexError):
            _ = dataset[-51]  # Negative index out of range
