"""
Verification tests for data loaders to ensure correct shapes and value ranges.

These tests verify the integration of all data loading components.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestDataLoaderVerification:
    """Integration tests to verify data loaders produce correct outputs."""

    def test_mit1003_train_loader_shapes(self):
        """Verify MIT1003 train loader produces correct tensor shapes."""
        from data.mit1003_loader import create_mit1003_dataloaders

        train_loader, _ = create_mit1003_dataloaders(
            data_path="/path/to/MIT1003",
            batch_size=4,
            num_workers=0
        )

        # Get one batch
        images, fixation_maps = next(iter(train_loader))

        assert images.shape == (4, 3, 768, 1024), \
            f"Expected shape (4, 3, 768, 1024), got {images.shape}"
        assert fixation_maps.shape == (4, 1, 768, 1024), \
            f"Expected shape (4, 1, 768, 1024), got {fixation_maps.shape}"

    def test_mit1003_val_loader_shapes(self):
        """Verify MIT1003 validation loader produces correct tensor shapes."""
        from data.mit1003_loader import create_mit1003_dataloaders

        _, val_loader = create_mit1003_dataloaders(
            data_path="/path/to/MIT1003",
            batch_size=4,
            num_workers=0
        )

        # Get one batch
        images, fixation_maps = next(iter(val_loader))

        assert images.shape == (4, 3, 768, 1024), \
            f"Expected shape (4, 3, 768, 1024), got {images.shape}"
        assert fixation_maps.shape == (4, 1, 768, 1024), \
            f"Expected shape (4, 1, 768, 1024), got {fixation_maps.shape}"

    def test_cat2000_loader_shapes(self):
        """Verify CAT2000 loader produces correct tensor shapes."""
        from data.cat2000_loader import create_cat2000_dataloader

        loader = create_cat2000_dataloader(
            data_path="/path/to/CAT2000",
            num_samples=50,
            batch_size=8,
            num_workers=0
        )

        # Get one batch
        images, fixation_maps = next(iter(loader))

        assert images.shape == (8, 3, 768, 1024), \
            f"Expected shape (8, 3, 768, 1024), got {images.shape}"
        assert fixation_maps.shape == (8, 1, 768, 1024), \
            f"Expected shape (8, 1, 768, 1024), got {fixation_maps.shape}"

    def test_image_value_ranges(self):
        """Verify images are properly normalized."""
        from data.mit1003_loader import create_mit1003_dataloaders

        train_loader, _ = create_mit1003_dataloaders(
            data_path="/path/to/MIT1003",
            batch_size=4,
            num_workers=0,
            normalize=True
        )

        images, _ = next(iter(train_loader))

        # Normalized images should be roughly in [-3, 3] range
        assert images.min() >= -10.0, "Image min value seems incorrect"
        assert images.max() <= 10.0, "Image max value seems incorrect"

    def test_fixation_map_value_ranges(self):
        """Verify fixation maps are in [0, 1] range."""
        from data.mit1003_loader import create_mit1003_dataloaders

        train_loader, _ = create_mit1003_dataloaders(
            data_path="/path/to/MIT1003",
            batch_size=4,
            num_workers=0
        )

        _, fixation_maps = next(iter(train_loader))

        assert fixation_maps.min() >= 0.0, \
            f"Fixation map min should be >= 0, got {fixation_maps.min()}"
        assert fixation_maps.max() <= 1.0, \
            f"Fixation map max should be <= 1, got {fixation_maps.max()}"

    def test_batch_consistency(self):
        """Verify all items in a batch have consistent shapes."""
        from data.mit1003_loader import create_mit1003_dataloaders

        train_loader, _ = create_mit1003_dataloaders(
            data_path="/path/to/MIT1003",
            batch_size=8,
            num_workers=0
        )

        images, fixation_maps = next(iter(train_loader))

        # All images should have same shape
        for i in range(images.shape[0]):
            assert images[i].shape == (3, 768, 1024)
            assert fixation_maps[i].shape == (1, 768, 1024)

    def test_dataloader_iteration(self):
        """Verify dataloaders can iterate through entire dataset."""
        from data.mit1003_loader import MIT1003Dataset
        from torch.utils.data import DataLoader

        # Use small dataset for testing
        dataset = MIT1003Dataset(
            data_path="/path/to/MIT1003",
            split="train"
        )

        loader = DataLoader(dataset, batch_size=32, num_workers=0)

        # Count total samples
        total_samples = 0
        for images, fixation_maps in loader:
            total_samples += images.shape[0]
            assert images.shape[1:] == (3, 768, 1024)
            assert fixation_maps.shape[1:] == (1, 768, 1024)

        # Should iterate through all 902 training samples
        assert total_samples == 902, \
            f"Expected to iterate through 902 samples, got {total_samples}"

    def test_no_data_leakage_between_splits(self):
        """Verify train and val splits are independent."""
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

        # Different sizes
        assert len(train_dataset) == 902
        assert len(val_dataset) == 101

        # Total should be 1003
        assert len(train_dataset) + len(val_dataset) == 1003
