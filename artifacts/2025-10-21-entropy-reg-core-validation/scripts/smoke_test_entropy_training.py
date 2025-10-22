"""
Smoke test for entropy-regularized training pipeline.

Tests that training with entropy regularization can run for 1-2 epochs without errors.
Verifies that entropy loss is computed and logged correctly.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from training.trainer import Trainer
from models.deepgaze3 import DeepGazeIII
from models.entropy_regularizer import EntropyRegularizer


class MockDataset(Dataset):
    """Mock dataset for smoke testing."""

    def __init__(self, num_samples=50, image_size=(1024, 768)):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create random image
        image = torch.randn(3, self.image_size[1], self.image_size[0])

        # Create random fixation map
        fixation_map = torch.abs(torch.randn(1, self.image_size[1], self.image_size[0]))

        return image, fixation_map


def test_entropy_regularizer_creation():
    """Test that entropy regularizer is created."""
    print("\n[1/4] Testing entropy regularizer creation...")

    config_path = Path(__file__).parent.parent / 'configs' / 'entropy_reg_config.yaml'

    try:
        trainer = Trainer(
            config_path=str(config_path),
            rank=0,
            world_size=1,
            use_entropy_regularization=True
        )

        assert trainer.entropy_regularizer is not None
        assert isinstance(trainer.entropy_regularizer, EntropyRegularizer)
        print("✓ Entropy regularizer created successfully")
        return True
    except Exception as e:
        print(f"✗ Entropy regularizer creation failed: {e}")
        return False


def test_entropy_loss_computation():
    """Test that entropy loss can be computed."""
    print("\n[2/4] Testing entropy loss computation...")

    config_path = Path(__file__).parent.parent / 'configs' / 'entropy_reg_config.yaml'

    try:
        trainer = Trainer(
            config_path=str(config_path),
            rank=0,
            world_size=1,
            use_entropy_regularization=True
        )

        # Move to GPU if available
        if torch.cuda.is_available():
            trainer.model = trainer.model.cuda()
            trainer.entropy_regularizer = EntropyRegularizer(
                model=trainer.model,
                image_size=(1024, 768),
                num_samples=16,
                device='cuda'
            )
            trainer.device = torch.device('cuda')

        # Compute entropy loss
        entropy_loss, entropy_value = trainer.entropy_regularizer.compute_entropy_loss()

        assert entropy_loss.dim() == 0, "Entropy loss should be scalar"
        assert not torch.isnan(entropy_loss), "Entropy loss should not be NaN"
        assert not torch.isnan(torch.tensor(entropy_value)), "Entropy value should not be NaN"

        print(f"✓ Entropy loss computed successfully: {entropy_value:.4f} bits")
        return True
    except Exception as e:
        print(f"✗ Entropy loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_combined_loss():
    """Test combined NLL + entropy loss."""
    print("\n[3/4] Testing combined loss computation...")

    config_path = Path(__file__).parent.parent / 'configs' / 'entropy_reg_config.yaml'

    try:
        trainer = Trainer(
            config_path=str(config_path),
            rank=0,
            world_size=1,
            use_entropy_regularization=True
        )

        # Create mock data
        predictions = torch.randn(4, 1, 768, 1024)
        fixation_maps = torch.abs(torch.randn(4, 1, 768, 1024))

        # Compute NLL loss
        nll_loss = trainer.compute_nll_loss(predictions, fixation_maps)

        # Mock entropy loss
        entropy_loss = torch.tensor(-10.0)  # Negative because we maximize entropy
        entropy_value = 10.0

        # Combined loss
        lambda_entropy = trainer.config['training']['loss']['entropy_weight']
        total_loss = nll_loss + lambda_entropy * entropy_loss

        assert total_loss.dim() == 0, "Total loss should be scalar"
        assert not torch.isnan(total_loss), "Total loss should not be NaN"

        print(f"✓ Combined loss computed successfully")
        print(f"  NLL: {nll_loss.item():.4f}, Entropy: {entropy_value:.4f}, Total: {total_loss.item():.4f}")
        return True
    except Exception as e:
        print(f"✗ Combined loss computation failed: {e}")
        return False


def test_entropy_training_epoch():
    """Test that one epoch with entropy regularization completes."""
    print("\n[4/4] Testing one epoch with entropy regularization...")

    config_path = Path(__file__).parent.parent / 'configs' / 'entropy_reg_config.yaml'

    try:
        trainer = Trainer(
            config_path=str(config_path),
            rank=0,
            world_size=1,
            use_entropy_regularization=True
        )

        # Create mock data loaders
        mock_dataset = MockDataset(num_samples=20)
        mock_loader = DataLoader(mock_dataset, batch_size=4, shuffle=True)

        # Override trainer's data loaders
        trainer.train_loader = mock_loader
        trainer.val_loader = mock_loader

        # Move model to GPU if available
        if torch.cuda.is_available():
            trainer.model = trainer.model.cuda()
            trainer.entropy_regularizer = EntropyRegularizer(
                model=trainer.model,
                image_size=(1024, 768),
                num_samples=16,
                device='cuda'
            )
            trainer.device = torch.device('cuda')
            print("  Using CUDA device")

        # Run one epoch with entropy regularization
        train_metrics = trainer.train_epoch(epoch=0)
        val_metrics = trainer.validate(epoch=0)

        # Verify entropy metrics are present
        assert 'train_entropy_value' in train_metrics, "Entropy value should be in training metrics"

        print(f"✓ One epoch with entropy regularization completed successfully")
        print(f"  Train loss: {train_metrics['train_loss']:.4f}")
        print(f"  Train NLL: {train_metrics['train_nll']:.4f}")
        print(f"  Train entropy: {train_metrics['train_entropy_value']:.4f}")
        print(f"  Val loss: {val_metrics['val_loss']:.4f}")
        return True
    except Exception as e:
        print(f"✗ Entropy training epoch failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("Entropy-Regularized Training Pipeline Smoke Test")
    print("=" * 60)

    tests = [
        test_entropy_regularizer_creation,
        test_entropy_loss_computation,
        test_combined_loss,
        test_entropy_training_epoch,
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n✓ All entropy-regularized training smoke tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit(main())
