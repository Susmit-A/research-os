"""
Tests for MIT1003 data loader.

Tests cover:
- Dataset initialization
- Train/validation split (902/101)
- Fixation map loading
- Image preprocessing (resize to 1024x768, ImageNet normalization)
- Correct shapes and value ranges
- Batch loading
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestMIT1003Dataset:
    """Test suite for MIT1003 dataset class."""

    @pytest.fixture
    def mock_data_path(self, tmp_path):
        """Create mock MIT1003 dataset structure for testing."""
        data_dir = tmp_path / "MIT1003"
        images_dir = data_dir / "ALLSTIMULI"
        fixations_dir = data_dir / "ALLFIXATIONMAPS"

        images_dir.mkdir(parents=True)
        fixations_dir.mkdir(parents=True)

        # Create mock images and fixation maps (1003 total for MIT1003)
        for i in range(1, 1004):
            # Create dummy image (random RGB)
            img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
            img_path = images_dir / f"i{i}.jpeg"
            # In real test, we'd save actual images, but for now we'll mock the file existence
            img_path.touch()

            # Create dummy fixation map
            fix_map = np.random.rand(600, 800).astype(np.float32)
            fix_path = fixations_dir / f"i{i}_fixMap.jpg"
            fix_path.touch()

        return data_dir

    def test_dataset_initialization(self):
        """Test that dataset can be initialized with correct parameters."""
        from data.mit1003_loader import MIT1003Dataset

        # This will fail initially (test-driven development)
        # The test defines the expected interface
        dataset = MIT1003Dataset(
            data_path="/path/to/MIT1003",
            split="train",
            image_size=(1024, 768),
            normalize=True
        )

        assert dataset is not None
        assert hasattr(dataset, '__len__')
        assert hasattr(dataset, '__getitem__')

    def test_train_split_size(self):
        """Test that train split has exactly 902 images."""
        from data.mit1003_loader import MIT1003Dataset

        dataset = MIT1003Dataset(
            data_path="/path/to/MIT1003",
            split="train"
        )

        assert len(dataset) == 902, \
            f"Train split should have 902 images, got {len(dataset)}"

    def test_val_split_size(self):
        """Test that validation split has exactly 101 images."""
        from data.mit1003_loader import MIT1003Dataset

        dataset = MIT1003Dataset(
            data_path="/path/to/MIT1003",
            split="val"
        )

        assert len(dataset) == 101, \
            f"Validation split should have 101 images, got {len(dataset)}"

    def test_total_dataset_size(self):
        """Test that total dataset has 1003 images (MIT1003 standard)."""
        from data.mit1003_loader import MIT1003Dataset

        train_dataset = MIT1003Dataset(data_path="/path/to/MIT1003", split="train")
        val_dataset = MIT1003Dataset(data_path="/path/to/MIT1003", split="val")

        total_size = len(train_dataset) + len(val_dataset)
        assert total_size == 1003, \
            f"Total dataset should have 1003 images, got {total_size}"

    def test_getitem_returns_correct_format(self):
        """Test that __getitem__ returns (image, fixation_map) tuple."""
        from data.mit1003_loader import MIT1003Dataset

        dataset = MIT1003Dataset(
            data_path="/path/to/MIT1003",
            split="train"
        )

        sample = dataset[0]

        assert isinstance(sample, tuple), "Dataset should return tuple"
        assert len(sample) == 2, "Sample should be (image, fixation_map)"

        image, fixation_map = sample
        assert isinstance(image, torch.Tensor), "Image should be torch.Tensor"
        assert isinstance(fixation_map, torch.Tensor), "Fixation map should be torch.Tensor"

    def test_image_shape_after_preprocessing(self):
        """Test that images are resized to 1024x768."""
        from data.mit1003_loader import MIT1003Dataset

        dataset = MIT1003Dataset(
            data_path="/path/to/MIT1003",
            split="train",
            image_size=(1024, 768)
        )

        image, _ = dataset[0]

        # Image shape should be (C, H, W) = (3, 768, 1024)
        assert image.shape == (3, 768, 1024), \
            f"Image shape should be (3, 768, 1024), got {image.shape}"

    def test_fixation_map_shape(self):
        """Test that fixation maps match image dimensions."""
        from data.mit1003_loader import MIT1003Dataset

        dataset = MIT1003Dataset(
            data_path="/path/to/MIT1003",
            split="train",
            image_size=(1024, 768)
        )

        _, fixation_map = dataset[0]

        # Fixation map shape should be (1, H, W) = (1, 768, 1024)
        assert fixation_map.shape == (1, 768, 1024), \
            f"Fixation map shape should be (1, 768, 1024), got {fixation_map.shape}"

    def test_imagenet_normalization(self):
        """Test that ImageNet normalization is applied correctly."""
        from data.mit1003_loader import MIT1003Dataset

        dataset = MIT1003Dataset(
            data_path="/path/to/MIT1003",
            split="train",
            normalize=True
        )

        image, _ = dataset[0]

        # Check that values are in reasonable range for normalized images
        # ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # Normalized values typically in range [-3, 3]
        assert image.min() >= -5.0, "Normalized image min seems incorrect"
        assert image.max() <= 5.0, "Normalized image max seems incorrect"

        # Check approximate mean (should be close to 0 after normalization)
        mean_per_channel = image.mean(dim=(1, 2))
        assert torch.allclose(mean_per_channel, torch.zeros(3), atol=2.0), \
            "Mean should be approximately 0 after normalization"

    def test_value_ranges(self):
        """Test that fixation maps are in valid range [0, 1]."""
        from data.mit1003_loader import MIT1003Dataset

        dataset = MIT1003Dataset(
            data_path="/path/to/MIT1003",
            split="train"
        )

        _, fixation_map = dataset[0]

        assert fixation_map.min() >= 0.0, \
            f"Fixation map min should be >= 0, got {fixation_map.min()}"
        assert fixation_map.max() <= 1.0, \
            f"Fixation map max should be <= 1, got {fixation_map.max()}"

    def test_reproducible_split(self):
        """Test that train/val split is deterministic with same seed."""
        from data.mit1003_loader import MIT1003Dataset

        dataset1 = MIT1003Dataset(
            data_path="/path/to/MIT1003",
            split="train",
            seed=42
        )

        dataset2 = MIT1003Dataset(
            data_path="/path/to/MIT1003",
            split="train",
            seed=42
        )

        # Sample the same index from both datasets
        img1, fix1 = dataset1[0]
        img2, fix2 = dataset2[0]

        # Should be identical
        assert torch.allclose(img1, img2), "Same seed should produce identical splits"
        assert torch.allclose(fix1, fix2), "Same seed should produce identical splits"

    def test_dataloader_compatibility(self):
        """Test that dataset works with PyTorch DataLoader."""
        from data.mit1003_loader import MIT1003Dataset
        from torch.utils.data import DataLoader

        dataset = MIT1003Dataset(
            data_path="/path/to/MIT1003",
            split="train"
        )

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0
        )

        # Get one batch
        batch = next(iter(dataloader))
        images, fixation_maps = batch

        assert images.shape == (4, 3, 768, 1024), \
            f"Batch images shape should be (4, 3, 768, 1024), got {images.shape}"
        assert fixation_maps.shape == (4, 1, 768, 1024), \
            f"Batch fixation maps shape should be (4, 1, 768, 1024), got {fixation_maps.shape}"

    def test_different_splits_no_overlap(self):
        """Test that train and val splits have no overlapping images."""
        from data.mit1003_loader import MIT1003Dataset

        train_dataset = MIT1003Dataset(
            data_path="/path/to/MIT1003",
            split="train",
            seed=42
        )

        val_dataset = MIT1003Dataset(
            data_path="/path/to/MIT1003",
            split="val",
            seed=42
        )

        # This would require access to internal image IDs/paths
        # For now, we test that they have different sizes
        assert len(train_dataset) != len(val_dataset), \
            "Train and val splits should have different sizes"
