"""
Smoke test for training pipeline.

Tests that training can run for 1-2 epochs without errors.
Uses a small mock dataset to verify the pipeline works.
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


def test_trainer_initialization():
    """Test that trainer initializes correctly."""
    print("\n[1/5] Testing trainer initialization...")

    config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'

    try:
        trainer = Trainer(
            config_path=str(config_path),
            rank=0,
            world_size=1,
            use_entropy_regularization=False
        )
        print("✓ Trainer initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Trainer initialization failed: {e}")
        return False


def test_model_forward_pass():
    """Test that model can perform forward pass."""
    print("\n[2/5] Testing model forward pass...")

    try:
        model = DeepGazeIII(pretrained=False)
        model.eval()

        # Create dummy input
        batch_size = 2
        images = torch.randn(batch_size, 3, 768, 1024)
        centerbias = torch.zeros(batch_size, 1, 768, 1024)
        # 4 dummy fixations at center (DeepGaze 3 expects 4 previous fixations)
        x_hist = torch.full((batch_size, 4), 512.0)  # width/2
        y_hist = torch.full((batch_size, 4), 384.0)  # height/2

        with torch.no_grad():
            predictions = model(images, centerbias, x_hist, y_hist)

        assert predictions.shape == (batch_size, 1, 768, 1024), \
            f"Expected shape (2, 1, 768, 1024), got {predictions.shape}"

        print("✓ Model forward pass successful")
        return True
    except Exception as e:
        print(f"✗ Model forward pass failed: {e}")
        return False


def test_nll_loss_computation():
    """Test that NLL loss can be computed."""
    print("\n[3/5] Testing NLL loss computation...")

    config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'

    try:
        trainer = Trainer(
            config_path=str(config_path),
            rank=0,
            world_size=1,
            use_entropy_regularization=False
        )

        # Create dummy data
        predictions = torch.randn(4, 1, 768, 1024)
        fixation_maps = torch.abs(torch.randn(4, 1, 768, 1024))

        loss = trainer.compute_nll_loss(predictions, fixation_maps)

        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "NLL loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"

        print(f"✓ NLL loss computed successfully: {loss.item():.4f}")
        return True
    except Exception as e:
        print(f"✗ NLL loss computation failed: {e}")
        return False


def test_training_step():
    """Test that a single training step can be executed."""
    print("\n[4/5] Testing single training step...")

    config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'

    try:
        trainer = Trainer(
            config_path=str(config_path),
            rank=0,
            world_size=1,
            use_entropy_regularization=False
        )

        # Create mock data loader
        mock_dataset = MockDataset(num_samples=10)
        mock_loader = DataLoader(mock_dataset, batch_size=2, shuffle=False)

        # Override trainer's data loaders
        trainer.train_loader = mock_loader
        trainer.val_loader = mock_loader

        # Run one training step manually
        trainer.model.train()
        images, fixation_maps = next(iter(mock_loader))

        if torch.cuda.is_available():
            images = images.cuda()
            fixation_maps = fixation_maps.cuda()
            trainer.model = trainer.model.cuda()
            trainer.device = torch.device('cuda')

        trainer.optimizer.zero_grad()
        batch_size = images.size(0)
        height, width = images.size(2), images.size(3)
        centerbias = torch.zeros_like(fixation_maps)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 4 dummy fixations at center (DeepGaze 3 expects 4 previous fixations)
        x_hist = torch.full((batch_size, 4), width / 2.0, device=device)
        y_hist = torch.full((batch_size, 4), height / 2.0, device=device)
        if torch.cuda.is_available():
            centerbias = centerbias.cuda()
        predictions = trainer.model(images, centerbias, x_hist, y_hist)
        loss = trainer.compute_nll_loss(predictions, fixation_maps)
        loss.backward()
        trainer.optimizer.step()

        print(f"✓ Training step executed successfully, loss={loss.item():.4f}")
        return True
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_one_epoch_training():
    """Test that one epoch of training can complete."""
    print("\n[5/5] Testing one epoch training...")

    config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'

    try:
        trainer = Trainer(
            config_path=str(config_path),
            rank=0,
            world_size=1,
            use_entropy_regularization=False
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
            trainer.device = torch.device('cuda')
            print("  Using CUDA device")

        # Run one epoch
        train_metrics = trainer.train_epoch(epoch=0)
        val_metrics = trainer.validate(epoch=0)

        print(f"✓ One epoch completed successfully")
        print(f"  Train loss: {train_metrics['train_loss']:.4f}")
        print(f"  Val loss: {val_metrics['val_loss']:.4f}")
        return True
    except Exception as e:
        print(f"✗ One epoch training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("Training Pipeline Smoke Test")
    print("=" * 60)

    tests = [
        test_trainer_initialization,
        test_model_forward_pass,
        test_nll_loss_computation,
        test_training_step,
        test_one_epoch_training,
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
        print("\n✓ All smoke tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit(main())
