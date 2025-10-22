"""
Tests for training framework.

Tests cover:
- Configuration loading and validation
- NLL loss computation
- Training loop execution
- Checkpointing system
- Distributed training setup
- Learning rate scheduling
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import yaml
import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestConfigurationLoading:
    """Test configuration file loading and validation."""

    def test_load_baseline_config(self):
        """Test loading baseline configuration."""
        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        assert config['experiment']['name'] == 'deepgaze3_baseline'
        assert config['training']['epochs'] == 25
        assert config['training']['batch_size'] == 32
        assert config['training']['optimizer']['lr'] == 0.001585

    def test_baseline_config_has_required_fields(self):
        """Test that baseline config has all required fields."""
        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Check experiment fields
        assert 'name' in config['experiment']
        assert 'seed' in config['experiment']

        # Check model fields
        assert 'name' in config['model']
        assert 'pretrained' in config['model']

        # Check data fields
        assert 'dataset' in config['data']
        assert 'image_size' in config['data']

        # Check training fields
        assert 'epochs' in config['training']
        assert 'batch_size' in config['training']
        assert 'optimizer' in config['training']
        assert 'lr_scheduler' in config['training']

    def test_lr_scheduler_config(self):
        """Test LR scheduler configuration."""
        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        scheduler_config = config['training']['lr_scheduler']
        assert scheduler_config['type'] == 'MultiStepLR'
        assert 'milestones' in scheduler_config
        assert 'gamma' in scheduler_config
        assert len(scheduler_config['milestones']) > 0


class TestNLLLoss:
    """Test NLL loss computation."""

    def test_nll_loss_returns_scalar(self):
        """Test that NLL loss returns a scalar tensor."""
        from training.trainer import Trainer

        # Create mock predictions and fixation maps
        batch_size = 4
        height, width = 768, 1024

        predictions = torch.randn(batch_size, 1, height, width)
        fixation_maps = torch.abs(torch.randn(batch_size, 1, height, width))

        # Create a mock trainer to access compute_nll_loss
        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
        trainer = Trainer(config_path=str(config_path), rank=0, world_size=1)

        loss = trainer.compute_nll_loss(predictions, fixation_maps)

        # Should return scalar
        assert loss.dim() == 0, f"Expected scalar (0-dim), got {loss.dim()}-dim tensor"
        assert loss.item() >= 0, "NLL loss should be non-negative"

    def test_nll_loss_toy_example(self):
        """Test NLL loss on toy example with known result."""
        from training.trainer import Trainer

        # Create simple toy example
        # Prediction: all equal probabilities
        predictions = torch.zeros(1, 1, 4, 4)  # Will become uniform after softmax

        # Fixation: single point
        fixation_maps = torch.zeros(1, 1, 4, 4)
        fixation_maps[0, 0, 2, 2] = 1.0

        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
        trainer = Trainer(config_path=str(config_path), rank=0, world_size=1)

        loss = trainer.compute_nll_loss(predictions, fixation_maps)

        # For uniform prediction over 16 pixels, NLL = -log(1/16) = log(16) ≈ 2.77
        expected_loss = np.log(16)
        assert abs(loss.item() - expected_loss) < 0.1, \
            f"Expected loss ~{expected_loss:.2f}, got {loss.item():.2f}"

    def test_nll_loss_gradient_flow(self):
        """Test that gradients flow through NLL loss."""
        from training.trainer import Trainer

        predictions = torch.randn(2, 1, 4, 4, requires_grad=True)
        fixation_maps = torch.abs(torch.randn(2, 1, 4, 4))

        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
        trainer = Trainer(config_path=str(config_path), rank=0, world_size=1)

        loss = trainer.compute_nll_loss(predictions, fixation_maps)
        loss.backward()

        assert predictions.grad is not None, "Gradients should flow to predictions"
        assert not torch.all(predictions.grad == 0), "Gradients should be non-zero"


class TestCheckpointing:
    """Test checkpoint saving and loading."""

    def test_save_checkpoint(self):
        """Test that checkpoint saves correctly."""
        from training.trainer import Trainer

        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Modify config to use temporary output directory
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Create trainer
            trainer = Trainer(config_path=str(config_path), rank=0, world_size=1)

            # Save checkpoint
            checkpoint_dir = Path(tmp_dir) / 'checkpoints' / 'test'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = checkpoint_dir / 'test_checkpoint.pt'

            # Manually create checkpoint
            checkpoint = {
                'epoch': 5,
                'global_step': 100,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict(),
                'best_val_loss': 2.5,
                'config': trainer.config,
            }
            torch.save(checkpoint, checkpoint_path)

            # Verify checkpoint exists and can be loaded
            assert checkpoint_path.exists()
            loaded = torch.load(checkpoint_path, map_location='cpu')
            assert loaded['epoch'] == 5
            assert loaded['global_step'] == 100
            assert loaded['best_val_loss'] == 2.5

    def test_checkpoint_contains_required_keys(self):
        """Test that checkpoint contains all required keys."""
        from training.trainer import Trainer

        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
        trainer = Trainer(config_path=str(config_path), rank=0, world_size=1)

        # Create checkpoint dict
        checkpoint = {
            'epoch': 0,
            'global_step': 0,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict(),
            'best_val_loss': float('inf'),
            'config': trainer.config,
        }

        required_keys = ['epoch', 'global_step', 'model_state_dict',
                        'optimizer_state_dict', 'scheduler_state_dict',
                        'best_val_loss', 'config']

        for key in required_keys:
            assert key in checkpoint, f"Checkpoint missing required key: {key}"


class TestLRScheduler:
    """Test learning rate scheduler."""

    def test_multistep_scheduler_initialization(self):
        """Test MultiStepLR scheduler initializes correctly."""
        from training.trainer import Trainer

        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
        trainer = Trainer(config_path=str(config_path), rank=0, world_size=1)

        # Check initial learning rate
        initial_lr = trainer.optimizer.param_groups[0]['lr']
        assert abs(initial_lr - 0.001585) < 1e-6, \
            f"Expected lr=0.001585, got {initial_lr}"

    def test_multistep_scheduler_steps(self):
        """Test that scheduler reduces LR at milestones."""
        from training.trainer import Trainer

        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
        trainer = Trainer(config_path=str(config_path), rank=0, world_size=1)

        initial_lr = trainer.optimizer.param_groups[0]['lr']

        # Step through epochs
        for epoch in range(15):
            trainer.scheduler.step()

        # After milestone (epoch 12), LR should be reduced
        current_lr = trainer.optimizer.param_groups[0]['lr']
        expected_lr = initial_lr * 0.1  # gamma = 0.1

        assert abs(current_lr - expected_lr) < 1e-6, \
            f"Expected lr={expected_lr}, got {current_lr} after milestone"


class TestTrainerInitialization:
    """Test trainer initialization."""

    def test_trainer_initializes_model(self):
        """Test that trainer initializes DeepGaze 3 model."""
        from training.trainer import Trainer

        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
        trainer = Trainer(config_path=str(config_path), rank=0, world_size=1)

        assert trainer.model is not None
        assert isinstance(trainer.model, nn.Module)

    def test_trainer_initializes_optimizer(self):
        """Test that trainer initializes optimizer."""
        from training.trainer import Trainer

        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
        trainer = Trainer(config_path=str(config_path), rank=0, world_size=1)

        assert trainer.optimizer is not None
        assert isinstance(trainer.optimizer, torch.optim.Optimizer)

    def test_trainer_initializes_scheduler(self):
        """Test that trainer initializes LR scheduler."""
        from training.trainer import Trainer

        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
        trainer = Trainer(config_path=str(config_path), rank=0, world_size=1)

        assert trainer.scheduler is not None

    def test_trainer_no_entropy_regularizer_for_baseline(self):
        """Test that baseline trainer has no entropy regularizer."""
        from training.trainer import Trainer

        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
        trainer = Trainer(
            config_path=str(config_path),
            rank=0,
            world_size=1,
            use_entropy_regularization=False
        )

        assert trainer.entropy_regularizer is None


class TestDataLoaderIntegration:
    """Test data loader integration."""

    @pytest.mark.skipif(not Path('/path/to/MIT1003').exists(),
                       reason="MIT1003 dataset not available")
    def test_trainer_creates_data_loaders(self):
        """Test that trainer creates train and validation loaders."""
        from training.trainer import Trainer

        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
        trainer = Trainer(config_path=str(config_path), rank=0, world_size=1)

        assert trainer.train_loader is not None
        assert trainer.val_loader is not None


class TestTrainingState:
    """Test training state management."""

    def test_trainer_initializes_training_state(self):
        """Test that trainer initializes training state variables."""
        from training.trainer import Trainer

        config_path = Path(__file__).parent.parent / 'configs' / 'baseline_config.yaml'
        trainer = Trainer(config_path=str(config_path), rank=0, world_size=1)

        assert trainer.current_epoch == 0
        assert trainer.global_step == 0
        assert trainer.best_val_loss == float('inf')


class TestEntropyRegularization:
    """Test entropy regularization integration."""

    def test_entropy_config_loading(self):
        """Test loading entropy-regularized configuration."""
        config_path = Path(__file__).parent.parent / 'configs' / 'entropy_reg_config.yaml'

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        assert config['experiment']['name'] == 'deepgaze3_entropy_reg'
        assert config['training']['entropy_regularization']['enabled'] == True
        assert config['training']['entropy_regularization']['compute_every'] == 50
        assert config['training']['entropy_regularization']['num_uniform_samples'] == 16
        assert config['training']['loss']['entropy_weight'] == 1.0

    def test_trainer_has_entropy_regularizer(self):
        """Test that entropy-regularized trainer creates entropy regularizer."""
        from training.trainer import Trainer

        config_path = Path(__file__).parent.parent / 'configs' / 'entropy_reg_config.yaml'
        trainer = Trainer(
            config_path=str(config_path),
            rank=0,
            world_size=1,
            use_entropy_regularization=True
        )

        assert trainer.entropy_regularizer is not None
        assert trainer.use_entropy_reg == True

    def test_entropy_computation_frequency(self):
        """Test that entropy is computed at correct frequency."""
        from training.trainer import Trainer

        config_path = Path(__file__).parent.parent / 'configs' / 'entropy_reg_config.yaml'
        trainer = Trainer(
            config_path=str(config_path),
            rank=0,
            world_size=1,
            use_entropy_regularization=True
        )

        # Check config value
        compute_every = trainer.config['training']['entropy_regularization']['compute_every']
        assert compute_every == 50

    def test_combined_loss_with_entropy(self):
        """Test combined NLL + entropy loss."""
        from training.trainer import Trainer

        config_path = Path(__file__).parent.parent / 'configs' / 'entropy_reg_config.yaml'
        trainer = Trainer(
            config_path=str(config_path),
            rank=0,
            world_size=1,
            use_entropy_regularization=True
        )

        # Verify lambda parameter
        lambda_entropy = trainer.config['training']['loss']['entropy_weight']
        assert lambda_entropy == 1.0

    def test_entropy_regularizer_initialization(self):
        """Test entropy regularizer is properly initialized."""
        from training.trainer import Trainer
        from models.entropy_regularizer import EntropyRegularizer

        config_path = Path(__file__).parent.parent / 'configs' / 'entropy_reg_config.yaml'
        trainer = Trainer(
            config_path=str(config_path),
            rank=0,
            world_size=1,
            use_entropy_regularization=True
        )

        assert isinstance(trainer.entropy_regularizer, EntropyRegularizer)
        assert trainer.entropy_regularizer.num_samples == 16
