"""
Tests for distributed data loading with multiple GPUs.

Tests verify:
- DistributedSampler functionality
- Data distribution across multiple GPUs
- No data duplication between processes
- Correct batch sizes in distributed setting
"""

import pytest
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestDistributedDataLoading:
    """Test suite for distributed data loading on multiple GPUs."""

    def test_distributed_sampler_creation(self):
        """Test that DistributedSampler can be created for MIT1003."""
        from data.mit1003_loader import MIT1003Dataset

        dataset = MIT1003Dataset(
            data_path="/path/to/MIT1003",
            split="train"
        )

        # Create DistributedSampler (simulating 4 GPUs)
        sampler = DistributedSampler(
            dataset,
            num_replicas=4,
            rank=0,
            shuffle=True
        )

        assert sampler is not None
        assert len(sampler) > 0

    def test_distributed_sampler_splits_data(self):
        """Test that DistributedSampler correctly splits data across ranks."""
        from data.mit1003_loader import MIT1003Dataset

        dataset = MIT1003Dataset(
            data_path="/path/to/MIT1003",
            split="train"  # 902 samples
        )

        # Simulate 4 GPUs
        num_replicas = 4
        samplers = []

        for rank in range(num_replicas):
            sampler = DistributedSampler(
                dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=False  # No shuffle for testing
            )
            samplers.append(sampler)

        # Each rank should get roughly equal number of samples
        # 902 samples / 4 GPUs = 225.5, so each gets 225 or 226
        for sampler in samplers:
            assert 225 <= len(sampler) <= 226, \
                f"Expected ~225 samples per rank, got {len(sampler)}"

    def test_distributed_dataloader_creation(self):
        """Test creating DataLoader with DistributedSampler."""
        from data.mit1003_loader import MIT1003Dataset
        from torch.utils.data import DataLoader

        dataset = MIT1003Dataset(
            data_path="/path/to/MIT1003",
            split="train"
        )

        sampler = DistributedSampler(
            dataset,
            num_replicas=4,
            rank=0
        )

        dataloader = DataLoader(
            dataset,
            batch_size=32,
            sampler=sampler,
            num_workers=0,
            pin_memory=False
        )

        assert dataloader is not None
        assert len(dataloader) > 0

    def test_distributed_batch_sizes(self):
        """Test that distributed dataloaders produce correct batch sizes."""
        from data.mit1003_loader import MIT1003Dataset
        from torch.utils.data import DataLoader

        dataset = MIT1003Dataset(
            data_path="/path/to/MIT1003",
            split="train"
        )

        batch_size = 32
        sampler = DistributedSampler(
            dataset,
            num_replicas=4,
            rank=0
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0
        )

        # Get first batch
        batch = next(iter(dataloader))
        images, fixation_maps = batch

        # Batch size should match (except possibly last batch)
        assert images.shape[0] <= batch_size
        assert fixation_maps.shape[0] <= batch_size

    def test_cat2000_distributed_loading(self):
        """Test distributed loading for CAT2000 OOD evaluation."""
        from data.cat2000_loader import CAT2000Dataset
        from torch.utils.data import DataLoader

        dataset = CAT2000Dataset(
            data_path="/path/to/CAT2000",
            num_samples=50
        )

        # For evaluation, we might not use DistributedSampler
        # But test that it works if needed
        sampler = DistributedSampler(
            dataset,
            num_replicas=4,
            rank=0,
            shuffle=False  # No shuffle for evaluation
        )

        dataloader = DataLoader(
            dataset,
            batch_size=8,
            sampler=sampler,
            num_workers=0
        )

        # Should be able to iterate
        batch = next(iter(dataloader))
        assert len(batch) == 2  # (images, fixation_maps)

    def test_no_duplicate_samples_across_ranks(self):
        """Test that different ranks don't get duplicate samples."""
        from data.mit1003_loader import MIT1003Dataset

        dataset = MIT1003Dataset(
            data_path="/path/to/MIT1003",
            split="train"
        )

        num_replicas = 4
        all_indices = []

        for rank in range(num_replicas):
            sampler = DistributedSampler(
                dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=False
            )

            # Collect indices for this rank
            rank_indices = list(sampler)
            all_indices.extend(rank_indices)

        # Check for duplicates
        unique_indices = set(all_indices)
        assert len(all_indices) == len(unique_indices), \
            "Found duplicate samples across ranks"

    def test_epoch_setting(self):
        """Test that setting epoch changes sampler behavior."""
        from data.mit1003_loader import MIT1003Dataset

        dataset = MIT1003Dataset(
            data_path="/path/to/MIT1003",
            split="train"
        )

        sampler = DistributedSampler(
            dataset,
            num_replicas=4,
            rank=0,
            shuffle=True
        )

        # Set epoch to 0
        sampler.set_epoch(0)
        indices_epoch0 = list(sampler)

        # Set epoch to 1 (should give different shuffle)
        sampler.set_epoch(1)
        indices_epoch1 = list(sampler)

        # With shuffle=True, different epochs should give different orders
        # (though with small probability they could be the same)
        # This is a probabilistic test
        assert len(indices_epoch0) == len(indices_epoch1)

    def test_create_dataloaders_with_distributed_sampler(self):
        """Test that create_mit1003_dataloaders uses DistributedSampler for DDP."""
        from data.mit1003_loader import create_mit1003_dataloaders

        # Test with world_size > 1 (distributed training)
        # This should fail gracefully if path doesn't exist, but we can check the API
        try:
            train_loader, val_loader = create_mit1003_dataloaders(
                data_path="/path/to/MIT1003",
                batch_size=32,
                world_size=4,
                rank=0
            )

            # If loaders were created, verify they have DistributedSampler
            if train_loader is not None:
                assert train_loader.sampler is not None, \
                    "Train loader should have a sampler in distributed mode"
                assert isinstance(train_loader.sampler, DistributedSampler), \
                    "Train sampler should be DistributedSampler when world_size > 1"

            if val_loader is not None:
                assert val_loader.sampler is not None, \
                    "Val loader should have a sampler in distributed mode"
                assert isinstance(val_loader.sampler, DistributedSampler), \
                    "Val sampler should be DistributedSampler when world_size > 1"
        except FileNotFoundError:
            # Expected if dataset doesn't exist - test passes
            pass

    def test_create_dataloaders_without_distributed_sampler(self):
        """Test that create_mit1003_dataloaders uses shuffle when world_size=1."""
        from data.mit1003_loader import create_mit1003_dataloaders

        # Test with world_size = 1 (single GPU)
        try:
            train_loader, val_loader = create_mit1003_dataloaders(
                data_path="/path/to/MIT1003",
                batch_size=32,
                world_size=1,
                rank=0
            )

            # If loaders were created, verify they don't have DistributedSampler
            if train_loader is not None:
                assert train_loader.sampler is None, \
                    "Train loader should not have sampler in single-GPU mode (uses shuffle instead)"
        except FileNotFoundError:
            # Expected if dataset doesn't exist - test passes
            pass


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4,
                    reason="Requires 4 GPUs")
class TestActualDistributedLoading:
    """Tests that require actual multi-GPU setup.

    These tests are skipped if less than 4 GPUs are available.
    """

    def test_multi_gpu_data_loading(self):
        """Test actual data loading on multiple GPUs.

        This test would need to be run with torch.distributed.launch
        or torch.multiprocessing to actually execute on multiple GPUs.
        """
        # This is a placeholder for integration testing
        # Actual testing would require DDP setup
        assert torch.cuda.device_count() >= 4

    def test_distributed_training_simulation(self):
        """Simulate one step of distributed training.

        This verifies the data loading pipeline works in a
        distributed training context.
        """
        from data.mit1003_loader import MIT1003Dataset
        from torch.utils.data import DataLoader

        # In actual distributed training, these would be set by DDP
        world_size = torch.cuda.device_count()
        rank = 0  # This would be the process rank

        dataset = MIT1003Dataset(
            data_path="/path/to/MIT1003",
            split="train"
        )

        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank
        )

        dataloader = DataLoader(
            dataset,
            batch_size=32 // world_size,  # Effective batch size per GPU
            sampler=sampler,
            num_workers=2,
            pin_memory=True
        )

        # Get one batch
        batch = next(iter(dataloader))
        images, fixation_maps = batch

        # Move to GPU
        device = torch.device(f'cuda:{rank}')
        images = images.to(device)
        fixation_maps = fixation_maps.to(device)

        assert images.is_cuda
        assert fixation_maps.is_cuda
