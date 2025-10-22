# Implementation Guide: Tasks 4-8

**Purpose**: This guide provides comprehensive architecture documentation, code templates, and step-by-step instructions for implementing the remaining tasks (4-8) of the entropy regularization core validation experiment.

**Created**: 2025-10-22
**Status**: Ready for implementation

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Task 4: Baseline Training Pipeline](#task-4-baseline-training-pipeline)
3. [Task 5: Entropy-Regularized Training Pipeline](#task-5-entropy-regularized-training-pipeline)
4. [Task 6: Parallel Training Execution](#task-6-parallel-training-execution)
5. [Task 7: Evaluation Metrics](#task-7-evaluation-metrics)
6. [Task 8: Final Evaluation and Report](#task-8-final-evaluation-and-report)
7. [Testing Strategy](#testing-strategy)
8. [Integration Checklist](#integration-checklist)

---

## Architecture Overview

### System Components

The complete system consists of 5 major components:

```
┌─────────────────────────────────────────────────────────────┐
│                    Training System                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Data Loading │  │  DeepGaze 3  │  │   Entropy    │     │
│  │  (MIT1003)   │→ │    Model     │→ │ Regularizer  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         ↓                  ↓                   ↓            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          Training Loop (Baseline / Entropy)          │  │
│  │  - NLL Loss                                          │  │
│  │  - Entropy Loss (optional)                           │  │
│  │  - Optimizer Step                                    │  │
│  │  - Checkpointing                                     │  │
│  │  - Logging                                           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Evaluation System                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Information  │  │ Bias Entropy │  │  Checkpoint  │     │
│  │     Gain     │  │ Measurement  │  │   Loading    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         ↓                  ↓                   ↓            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          Evaluation Pipeline                         │  │
│  │  - MIT1003 validation (101 images)                   │  │
│  │  - CAT2000 OOD (50 images)                           │  │
│  │  - Metrics computation                               │  │
│  │  - Report generation                                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

**Training Flow**:
```
Images (batch) → Model → Predictions → NLL Loss → Backward → Update
                  ↓
            (every N batches)
                  ↓
         Uniform Images → Model → Bias Maps → Entropy → Loss
```

**Evaluation Flow**:
```
Checkpoint → Load Model → Validation Images → Predictions → Information Gain
                  ↓
            Uniform Images → Bias Maps → Entropy Measurement
```

### File Structure After Completion

```
research-os/artifacts/2025-10-21-entropy-reg-core-validation/
├── configs/
│   ├── baseline_config.yaml          ✓ Complete
│   └── entropy_reg_config.yaml       ✓ Complete
├── scripts/
│   ├── run_baseline.sh               ✓ Complete
│   ├── run_entropy_reg.sh            ✓ Complete
│   ├── smoke_test.py                 ✓ Complete
│   └── download_datasets.py          ✓ Complete
├── src/
│   ├── data/
│   │   ├── mit1003_loader.py         ✓ Complete
│   │   └── cat2000_loader.py         ✓ Complete
│   ├── models/
│   │   ├── entropy_regularizer.py    ✓ Complete
│   │   ├── deepgaze3.py              ✓ Complete (from Kümmerer)
│   │   └── [other modules]           ✓ Complete (from Kümmerer)
│   ├── training/
│   │   ├── trainer.py                ⏳ Task 4 - NEW
│   │   ├── train_baseline.py         ⏳ Task 4 - NEW
│   │   └── train_entropy_reg.py      ⏳ Task 5 - NEW
│   └── evaluation/
│       ├── metrics.py                ⏳ Task 7 - NEW
│       ├── evaluate_model.py         ⏳ Task 7 - NEW
│       └── generate_report.py        ⏳ Task 8 - NEW
├── tests/
│   ├── test_mit1003_loader.py        ✓ Complete (12 tests)
│   ├── test_cat2000_loader.py        ✓ Complete (12 tests)
│   ├── test_data_verification.py     ✓ Complete (8 tests)
│   ├── test_distributed_loading.py   ✓ Complete (9 tests)
│   ├── test_entropy_regularization.py ✓ Complete (23 tests)
│   ├── test_trainer.py               ⏳ Task 4 - NEW
│   └── test_metrics.py               ⏳ Task 7 - NEW
├── outputs/
│   ├── checkpoints/
│   │   ├── baseline/                 ⏳ Task 6 output
│   │   └── entropy_reg/              ⏳ Task 6 output
│   ├── logs/
│   │   ├── baseline/                 ⏳ Task 6 output
│   │   └── entropy_reg/              ⏳ Task 6 output
│   └── results/
│       ├── metrics.json              ⏳ Task 8 output
│       ├── comparison_table.csv      ⏳ Task 8 output
│       └── training_curves.png       ⏳ Task 8 output
└── GO_NO_GO_REPORT.md                ⏳ Task 8 - FINAL OUTPUT
```

---

## Task 4: Baseline Training Pipeline

### Overview

Implement complete baseline DeepGaze 3 training pipeline WITHOUT entropy regularization. This serves as the control for comparison.

### Sub-tasks Breakdown

- **4.1**: Write tests for training configuration loading
- **4.2**: Adapt DeepGaze 3 model from Kümmerer's code
- **4.3**: Implement MultiStep LR scheduler
- **4.4**: Implement NLL loss function
- **4.5**: Setup distributed training (PyTorch DDP)
- **4.6**: Implement checkpointing system
- **4.7**: Add training/validation logging
- **4.8**: Run smoke test (1-2 epochs)
- **4.9**: Verify pipeline is ready

### Architecture Design

#### Core Trainer Class

The `Trainer` class encapsulates all training logic and can be configured for both baseline and entropy-regularized training.

**Key Responsibilities**:
- Model initialization
- Optimizer and scheduler setup
- Training loop execution
- Validation execution
- Checkpointing
- Logging

#### Integration Points

**Existing Components to Use**:
1. **Data Loaders**: `src/data/mit1003_loader.py`
   - `create_mit1003_dataloaders()` function
   - Already implements distributed sampling
   - Returns train_loader and val_loader

2. **Model**: `src/models/deepgaze3.py`
   - `DeepGazeIII(pretrained=False)` for training from scratch
   - Forward pass: `predictions = model(images)`

3. **Config Files**: `configs/baseline_config.yaml`
   - Load with `yaml.safe_load()`
   - All hyperparameters defined

### Code Template: `src/training/trainer.py`

```python
"""
Training framework for DeepGaze 3 models.

Supports both baseline and entropy-regularized training.
"""

import os
import time
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.deepgaze3 import DeepGazeIII
from models.entropy_regularizer import EntropyRegularizer
from data.mit1003_loader import create_mit1003_dataloaders


class Trainer:
    """Training framework for DeepGaze 3 models."""

    def __init__(
        self,
        config_path: str,
        rank: int = 0,
        world_size: int = 1,
        use_entropy_regularization: bool = False
    ):
        """Initialize trainer.

        Args:
            config_path: Path to YAML configuration file
            rank: Process rank for distributed training
            world_size: Total number of processes
            use_entropy_regularization: Whether to use entropy regularization
        """
        self.rank = rank
        self.world_size = world_size
        self.use_entropy_reg = use_entropy_regularization

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup device
        self.device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(self.device)

        # Initialize distributed training
        if self.world_size > 1:
            self._setup_distributed()

        # Setup model
        self.model = self._setup_model()

        # Setup data loaders
        self.train_loader, self.val_loader = self._setup_data_loaders()

        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        # Setup entropy regularizer if needed
        self.entropy_regularizer = None
        if self.use_entropy_reg:
            self.entropy_regularizer = self._setup_entropy_regularizer()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Setup logging directory
        self.log_dir = self._setup_logging()

    def _setup_distributed(self):
        """Initialize distributed training."""
        # TODO: Initialize process group
        # dist.init_process_group(
        #     backend=self.config['training']['distributed']['backend'],
        #     init_method='env://',
        #     world_size=self.world_size,
        #     rank=self.rank
        # )
        pass

    def _setup_model(self) -> nn.Module:
        """Setup DeepGaze 3 model."""
        # TODO: Initialize model from config
        # model = DeepGazeIII(pretrained=self.config['model']['pretrained'])
        # model = model.to(self.device)
        #
        # if self.world_size > 1:
        #     model = DDP(model, device_ids=[self.rank])
        #
        # return model
        pass

    def _setup_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Setup MIT1003 data loaders."""
        # TODO: Create data loaders using existing function
        # from data.mit1003_loader import create_mit1003_dataloaders
        #
        # train_loader, val_loader = create_mit1003_dataloaders(
        #     data_path=self.config['data']['data_path'],
        #     batch_size=self.config['training']['batch_size'],
        #     num_workers=self.config['data']['num_workers'],
        #     image_size=tuple(self.config['data']['image_size']),
        #     world_size=self.world_size,
        #     rank=self.rank,
        #     seed=self.config['experiment']['seed']
        # )
        #
        # return train_loader, val_loader
        pass

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup Adam optimizer."""
        # TODO: Create optimizer from config
        # opt_config = self.config['training']['optimizer']
        # optimizer = torch.optim.Adam(
        #     self.model.parameters(),
        #     lr=opt_config['lr'],
        #     betas=opt_config['betas'],
        #     eps=opt_config['eps'],
        #     weight_decay=opt_config['weight_decay']
        # )
        # return optimizer
        pass

    def _setup_scheduler(self):
        """Setup MultiStep LR scheduler."""
        # TODO: Create scheduler from config
        # sched_config = self.config['training']['lr_scheduler']
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     self.optimizer,
        #     milestones=sched_config['milestones'],
        #     gamma=sched_config['gamma']
        # )
        # return scheduler
        pass

    def _setup_entropy_regularizer(self) -> Optional[EntropyRegularizer]:
        """Setup entropy regularizer."""
        if not self.use_entropy_reg:
            return None

        # TODO: Initialize entropy regularizer
        # ent_config = self.config['training']['entropy_regularization']
        # regularizer = EntropyRegularizer(
        #     model=self.model,
        #     image_size=tuple(self.config['data']['image_size']),
        #     num_samples=ent_config['num_uniform_samples'],
        #     device=self.device
        # )
        # return regularizer
        pass

    def _setup_logging(self) -> Path:
        """Setup logging directory."""
        # TODO: Create log directory
        # exp_name = self.config['experiment']['name']
        # log_dir = Path('outputs/logs') / exp_name
        # log_dir.mkdir(parents=True, exist_ok=True)
        # return log_dir
        pass

    def compute_nll_loss(
        self,
        predictions: torch.Tensor,
        fixation_maps: torch.Tensor
    ) -> torch.Tensor:
        """Compute negative log-likelihood loss.

        Args:
            predictions: Model predictions (batch, 1, H, W) - log-densities
            fixation_maps: Ground truth fixation maps (batch, 1, H, W)

        Returns:
            NLL loss value
        """
        # TODO: Implement NLL loss
        # 1. Convert predictions to log-probabilities using log_softmax
        # 2. Normalize fixation maps to probability distributions
        # 3. Compute -sum(p_fixation * log_p_prediction)
        # 4. Average over batch
        #
        # Example:
        # log_probs = F.log_softmax(predictions.view(predictions.size(0), -1), dim=1)
        # log_probs = log_probs.view_as(predictions)
        # fixation_probs = fixation_maps / (fixation_maps.sum(dim=(2,3), keepdim=True) + 1e-8)
        # nll = -torch.sum(fixation_probs * log_probs, dim=(2, 3)).mean()
        # return nll
        pass

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_nll = 0.0
        total_entropy_loss = 0.0
        total_entropy_value = 0.0
        num_entropy_computations = 0

        # TODO: Implement training loop
        # for batch_idx, (images, fixation_maps) in enumerate(self.train_loader):
        #     images = images.to(self.device)
        #     fixation_maps = fixation_maps.to(self.device)
        #
        #     self.optimizer.zero_grad()
        #
        #     # Forward pass
        #     predictions = self.model(images)
        #
        #     # Compute NLL loss
        #     nll_loss = self.compute_nll_loss(predictions, fixation_maps)
        #
        #     # Compute entropy regularization if enabled
        #     if self.use_entropy_reg and batch_idx % self.config['training']['entropy_regularization']['compute_every'] == 0:
        #         entropy_loss, entropy_value = self.entropy_regularizer.compute_entropy_loss()
        #         lambda_entropy = self.config['training']['loss']['entropy_weight']
        #         total_batch_loss = nll_loss + lambda_entropy * entropy_loss
        #
        #         total_entropy_loss += entropy_loss.item()
        #         total_entropy_value += entropy_value
        #         num_entropy_computations += 1
        #     else:
        #         total_batch_loss = nll_loss
        #
        #     # Backward pass
        #     total_batch_loss.backward()
        #     self.optimizer.step()
        #
        #     total_loss += total_batch_loss.item()
        #     total_nll += nll_loss.item()
        #     self.global_step += 1
        #
        #     # Logging
        #     if batch_idx % self.config['training']['logging']['log_every'] == 0:
        #         self._log_training_step(epoch, batch_idx, nll_loss.item(),
        #                                entropy_value if self.use_entropy_reg else None)

        # Return epoch metrics
        metrics = {
            'train_loss': total_loss / len(self.train_loader),
            'train_nll': total_nll / len(self.train_loader),
        }

        if self.use_entropy_reg and num_entropy_computations > 0:
            metrics['train_entropy_loss'] = total_entropy_loss / num_entropy_computations
            metrics['train_entropy_value'] = total_entropy_value / num_entropy_computations

        return metrics

    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate model.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()

        total_val_loss = 0.0

        # TODO: Implement validation loop
        # with torch.no_grad():
        #     for images, fixation_maps in self.val_loader:
        #         images = images.to(self.device)
        #         fixation_maps = fixation_maps.to(self.device)
        #
        #         predictions = self.model(images)
        #         val_loss = self.compute_nll_loss(predictions, fixation_maps)
        #
        #         total_val_loss += val_loss.item()

        metrics = {
            'val_loss': total_val_loss / len(self.val_loader),
        }

        return metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint.

        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        # TODO: Implement checkpointing
        # checkpoint_dir = Path('outputs/checkpoints') / self.config['experiment']['name']
        # checkpoint_dir.mkdir(parents=True, exist_ok=True)
        #
        # checkpoint = {
        #     'epoch': epoch,
        #     'global_step': self.global_step,
        #     'model_state_dict': self.model.state_dict(),
        #     'optimizer_state_dict': self.optimizer.state_dict(),
        #     'scheduler_state_dict': self.scheduler.state_dict(),
        #     'best_val_loss': self.best_val_loss,
        #     'config': self.config,
        # }
        #
        # # Save regular checkpoint
        # if epoch % self.config['training']['checkpoint']['save_every'] == 0:
        #     checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        #     torch.save(checkpoint, checkpoint_path)
        #
        # # Save best checkpoint
        # if is_best:
        #     best_path = checkpoint_dir / 'best_model.pt'
        #     torch.save(checkpoint, best_path)
        #
        # # Save last checkpoint
        # last_path = checkpoint_dir / 'last_checkpoint.pt'
        # torch.save(checkpoint, last_path)
        pass

    def _log_training_step(self, epoch, batch_idx, nll_loss, entropy_value=None):
        """Log training step information."""
        # TODO: Implement logging
        # if self.rank == 0:  # Only log from main process
        #     log_msg = f"Epoch {epoch}, Batch {batch_idx}: NLL={nll_loss:.4f}"
        #     if entropy_value is not None:
        #         log_msg += f", Entropy={entropy_value:.4f}"
        #     print(log_msg)
        pass

    def train(self, num_epochs: int):
        """Main training loop.

        Args:
            num_epochs: Number of epochs to train
        """
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Update learning rate
            self.scheduler.step()

            # Check if best model
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)

            # Log epoch summary
            if self.rank == 0:
                self._log_epoch_summary(epoch, train_metrics, val_metrics)

    def _log_epoch_summary(self, epoch, train_metrics, val_metrics):
        """Log epoch summary."""
        # TODO: Implement epoch logging
        # print(f"\nEpoch {epoch} Summary:")
        # print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        # print(f"  Train NLL: {train_metrics['train_nll']:.4f}")
        # if 'train_entropy_value' in train_metrics:
        #     print(f"  Train Entropy: {train_metrics['train_entropy_value']:.4f}")
        # print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        # print(f"  LR: {self.scheduler.get_last_lr()[0]:.6f}\n")
        pass
```

### Code Template: `src/training/train_baseline.py`

```python
"""
Baseline DeepGaze 3 training script (no entropy regularization).

Usage:
    # Single GPU:
    python src/training/train_baseline.py --config configs/baseline_config.yaml

    # Multi-GPU (via torchrun):
    torchrun --nproc_per_node=4 src/training/train_baseline.py --config configs/baseline_config.yaml
"""

import argparse
import os
import torch
import torch.distributed as dist
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.trainer import Trainer


def setup_distributed():
    """Setup distributed training environment."""
    # TODO: Setup distributed environment from torchrun env vars
    # if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    #     rank = int(os.environ['RANK'])
    #     world_size = int(os.environ['WORLD_SIZE'])
    #     local_rank = int(os.environ['LOCAL_RANK'])
    # else:
    #     rank = 0
    #     world_size = 1
    #     local_rank = 0
    #
    # return rank, world_size, local_rank
    pass


def main():
    parser = argparse.ArgumentParser(description='Train baseline DeepGaze 3')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    args = parser.parse_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    # TODO: Initialize trainer (baseline - no entropy regularization)
    # trainer = Trainer(
    #     config_path=args.config,
    #     rank=rank,
    #     world_size=world_size,
    #     use_entropy_regularization=False  # Baseline
    # )

    # TODO: Start training
    # num_epochs = trainer.config['training']['epochs']
    # trainer.train(num_epochs)

    # TODO: Cleanup distributed training
    # if world_size > 1:
    #     dist.destroy_process_group()


if __name__ == '__main__':
    main()
```

### Testing Strategy for Task 4

Create `tests/test_trainer.py`:

```python
"""
Tests for training framework.

Following TDD approach: Write these tests FIRST, then implement.
"""

import pytest
import torch
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from training.trainer import Trainer


class TestConfigurationLoading:
    """Test configuration file loading."""

    def test_load_baseline_config(self):
        """Test loading baseline configuration."""
        # TODO: Test that baseline config loads correctly
        # config_path = 'configs/baseline_config.yaml'
        # with open(config_path, 'r') as f:
        #     config = yaml.safe_load(f)
        #
        # assert config['experiment']['name'] == 'deepgaze3_baseline'
        # assert config['training']['epochs'] == 25
        # assert config['training']['batch_size'] == 32
        pass

    def test_config_has_required_fields(self):
        """Test that config has all required fields."""
        # TODO: Verify all required fields exist
        pass


class TestNLLLoss:
    """Test NLL loss computation."""

    def test_nll_loss_shape(self):
        """Test that NLL loss returns scalar."""
        # TODO: Create mock trainer and test NLL loss
        # trainer = create_mock_trainer()
        # predictions = torch.randn(4, 1, 768, 1024)  # batch=4
        # fixation_maps = torch.randn(4, 1, 768, 1024)
        #
        # loss = trainer.compute_nll_loss(predictions, fixation_maps)
        #
        # assert loss.dim() == 0  # Scalar
        # assert loss.item() >= 0  # NLL is non-negative
        pass

    def test_nll_loss_toy_example(self):
        """Test NLL loss on toy example with known result."""
        # TODO: Create toy example and verify numerical correctness
        pass


class TestCheckpointing:
    """Test checkpoint saving and loading."""

    def test_save_checkpoint(self, tmp_path):
        """Test that checkpoint saves correctly."""
        # TODO: Test checkpoint saving
        pass

    def test_load_checkpoint(self, tmp_path):
        """Test that checkpoint loads correctly."""
        # TODO: Test checkpoint loading and state restoration
        pass


class TestDistributedSetup:
    """Test distributed training setup."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_distributed_initialization(self):
        """Test distributed training initialization."""
        # TODO: Test DDP setup (may need mocking)
        pass


class TestTrainingLoop:
    """Test training loop execution."""

    def test_single_epoch_execution(self):
        """Test that single epoch runs without errors."""
        # TODO: Test training loop with small dataset
        pass

    def test_validation_execution(self):
        """Test that validation runs without errors."""
        # TODO: Test validation loop
        pass


# Helper functions for creating mock objects
def create_mock_trainer():
    """Create mock trainer for testing."""
    # TODO: Implement mock trainer creation
    pass
```

### Implementation Steps for Task 4

Following TDD approach:

1. **Sub-task 4.1**: Write all tests in `test_trainer.py` (tests will fail initially)
2. **Sub-task 4.2**: Verify DeepGaze 3 model loads correctly (already done in Task 1)
3. **Sub-task 4.3**: Implement `_setup_scheduler()` in Trainer class
4. **Sub-task 4.4**: Implement `compute_nll_loss()` method
5. **Sub-task 4.5**: Implement `_setup_distributed()` method
6. **Sub-task 4.6**: Implement `save_checkpoint()` and checkpoint loading
7. **Sub-task 4.7**: Implement logging methods
8. **Sub-task 4.8**: Complete all TODOs in `Trainer` class, then run smoke test
9. **Sub-task 4.9**: Use `engineer:test-runner` subagent to verify all tests pass

---

## Task 5: Entropy-Regularized Training Pipeline

### Overview

Extend baseline training to include entropy regularization. This mostly involves configuration changes and minor modifications to use the existing `EntropyRegularizer`.

### Sub-tasks Breakdown

- **5.1**: Write tests for entropy-regularized config
- **5.2**: Create `train_entropy_reg.py` (extends baseline script)
- **5.3**: Configure entropy computation frequency
- **5.4**: Verify combined loss computation
- **5.5**: Add entropy logging
- **5.6**: Run smoke test with regularization
- **5.7**: Verify pipeline is ready

### Code Template: `src/training/train_entropy_reg.py`

```python
"""
Entropy-regularized DeepGaze 3 training script.

Usage:
    # Single GPU:
    python src/training/train_entropy_reg.py --config configs/entropy_reg_config.yaml

    # Multi-GPU (via torchrun):
    torchrun --nproc_per_node=4 src/training/train_entropy_reg.py --config configs/entropy_reg_config.yaml
"""

import argparse
import os
import torch
import torch.distributed as dist
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.trainer import Trainer


def setup_distributed():
    """Setup distributed training environment."""
    # Same as baseline - reuse implementation
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    return rank, world_size, local_rank


def main():
    parser = argparse.ArgumentParser(description='Train entropy-regularized DeepGaze 3')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    args = parser.parse_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    # TODO: Initialize trainer WITH entropy regularization
    # trainer = Trainer(
    #     config_path=args.config,
    #     rank=rank,
    #     world_size=world_size,
    #     use_entropy_regularization=True  # ENABLE ENTROPY REGULARIZATION
    # )

    # TODO: Start training
    # num_epochs = trainer.config['training']['epochs']
    # trainer.train(num_epochs)

    # TODO: Cleanup distributed training
    # if world_size > 1:
    #     dist.destroy_process_group()


if __name__ == '__main__':
    main()
```

### Key Differences from Baseline

The `Trainer` class already supports entropy regularization via the `use_entropy_regularization` parameter. The key differences are:

1. **Initialization**: Set `use_entropy_regularization=True`
2. **Config**: Use `entropy_reg_config.yaml` instead of `baseline_config.yaml`
3. **Loss Computation**: The `train_epoch()` method already handles combined loss:
   ```python
   if self.use_entropy_reg and batch_idx % compute_every == 0:
       entropy_loss, entropy_value = self.entropy_regularizer.compute_entropy_loss()
       total_loss = nll_loss + lambda_entropy * entropy_loss
   ```

### Testing Strategy for Task 5

Add to `tests/test_trainer.py`:

```python
class TestEntropyRegularization:
    """Test entropy regularization integration."""

    def test_entropy_config_loading(self):
        """Test entropy regularization config loads correctly."""
        # TODO: Verify entropy config has required fields
        pass

    def test_entropy_loss_computation_frequency(self):
        """Test that entropy is computed at correct frequency."""
        # TODO: Verify compute_every parameter works
        pass

    def test_combined_loss(self):
        """Test NLL + entropy combined loss."""
        # TODO: Verify loss = nll + lambda * entropy_loss
        pass

    def test_entropy_logging(self):
        """Test that entropy values are logged."""
        # TODO: Verify entropy appears in logs
        pass
```

---

## Task 6: Parallel Training Execution

### Overview

Launch both baseline and entropy-regularized training jobs in parallel on the cluster. This is mostly operational rather than code development.

### Sub-tasks Breakdown

- **6.1**: Launch baseline training job (SLURM)
- **6.2**: Launch entropy-regularized training job (SLURM)
- **6.3**: Monitor training progress
- **6.4**: Verify both models complete 25 epochs
- **6.5**: Save final checkpoints

### Execution Instructions

#### Before Training

1. **Download Datasets**:
   ```bash
   python scripts/download_datasets.py
   ```

2. **Update Config Files**:
   - Edit `configs/baseline_config.yaml`: Update `data.data_path` to actual MIT1003 path
   - Edit `configs/entropy_reg_config.yaml`: Update `data.data_path` to actual MIT1003 path

3. **Verify Environment**:
   ```bash
   conda activate /mnt/lustre/work/bethge/bkr710/.conda/deepgaze
   python scripts/smoke_test.py
   ```

#### Launch Training Jobs

The SLURM scripts are already created in Task 1.

**Launch Baseline**:
```bash
sbatch scripts/run_baseline.sh
```

**Launch Entropy-Regularized**:
```bash
sbatch scripts/run_entropy_reg.sh
```

#### Monitor Training

**Check Job Status**:
```bash
squeue -u $USER
```

**View Training Logs**:
```bash
# Baseline
tail -f outputs/logs/deepgaze3_baseline/training.log

# Entropy-regularized
tail -f outputs/logs/deepgaze3_entropy_reg/training.log
```

**Check for Issues**:
- NaN losses
- Gradient explosions
- Out of memory errors
- Slow convergence

#### Expected Training Time

- **Baseline**: ~6-8 hours (25 epochs, 4x A100)
- **Entropy-regularized**: ~8-12 hours (25 epochs, 4x A100, entropy computation overhead)

#### Checkpoints

Checkpoints will be saved at:
```
outputs/checkpoints/deepgaze3_baseline/
├── checkpoint_epoch_5.pt
├── checkpoint_epoch_10.pt
├── checkpoint_epoch_15.pt
├── checkpoint_epoch_20.pt
├── checkpoint_epoch_25.pt
├── best_model.pt
└── last_checkpoint.pt

outputs/checkpoints/deepgaze3_entropy_reg/
├── checkpoint_epoch_5.pt
├── checkpoint_epoch_10.pt
├── checkpoint_epoch_15.pt
├── checkpoint_epoch_20.pt
├── checkpoint_epoch_25.pt
├── best_model.pt
└── last_checkpoint.pt
```

---

## Task 7: Evaluation Metrics

### Overview

Implement evaluation metrics: Information Gain (IG) and bias entropy measurement.

### Sub-tasks Breakdown

- **7.1**: Write tests for Information Gain computation
- **7.2**: Implement IG with Gaussian center prior
- **7.3**: Write tests for bias entropy measurement
- **7.4**: Implement bias entropy measurement
- **7.5**: Create MIT1003 evaluation script
- **7.6**: Create CAT2000 evaluation script
- **7.7**: Verify all evaluation tests pass

### Architecture Design

#### Information Gain

Information Gain measures how much better a model predicts fixations compared to a baseline prior (Gaussian center bias).

**Formula**:
```
IG = (1/N) Σ log( P_model(fixation) / P_baseline(fixation) )
```

Where:
- N = number of fixations
- P_model = model's predicted probability at fixation location
- P_baseline = Gaussian center prior probability at fixation location

#### Bias Entropy

Measures the entropy of the model's spatial bias (extracted from uniform images).

**Formula**:
```
H = -Σ p(x,y) * log(p(x,y))
```

Where p(x,y) is the normalized probability at spatial location (x,y).

### Code Template: `src/evaluation/metrics.py`

```python
"""
Evaluation metrics for saliency models.

Implements Information Gain and bias entropy measurement.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class GaussianCenterPrior:
    """Gaussian center prior for Information Gain baseline."""

    def __init__(self, image_size: Tuple[int, int], sigma: float = 0.25):
        """Initialize Gaussian center prior.

        Args:
            image_size: (width, height) of images
            sigma: Standard deviation of Gaussian (relative to image size)
        """
        self.image_size = image_size
        self.sigma = sigma
        self._prior_map = self._create_prior_map()

    def _create_prior_map(self) -> np.ndarray:
        """Create 2D Gaussian centered on image."""
        # TODO: Implement Gaussian prior creation
        # width, height = self.image_size
        #
        # # Create coordinate grids
        # x = np.arange(width)
        # y = np.arange(height)
        # xx, yy = np.meshgrid(x, y)
        #
        # # Center coordinates
        # cx = width / 2
        # cy = height / 2
        #
        # # Gaussian formula
        # sigma_x = self.sigma * width
        # sigma_y = self.sigma * height
        # prior = np.exp(-((xx - cx)**2 / (2 * sigma_x**2) + (yy - cy)**2 / (2 * sigma_y**2)))
        #
        # # Normalize to probability distribution
        # prior = prior / np.sum(prior)
        #
        # return prior
        pass

    def get_prior(self) -> np.ndarray:
        """Get prior probability map."""
        return self._prior_map


class InformationGainComputer:
    """Compute Information Gain metric."""

    def __init__(self, image_size: Tuple[int, int], center_prior_sigma: float = 0.25):
        """Initialize IG computer.

        Args:
            image_size: (width, height) of images
            center_prior_sigma: Sigma for Gaussian center prior
        """
        self.image_size = image_size
        self.center_prior = GaussianCenterPrior(image_size, center_prior_sigma)

    def compute_information_gain(
        self,
        predictions: torch.Tensor,
        fixation_maps: torch.Tensor
    ) -> float:
        """Compute Information Gain.

        Args:
            predictions: Model predictions (batch, 1, H, W) - log-densities
            fixation_maps: Ground truth fixation maps (batch, 1, H, W)

        Returns:
            Information Gain value (bits)
        """
        # TODO: Implement Information Gain computation
        # 1. Convert predictions to probabilities
        # 2. Extract fixation locations from fixation_maps
        # 3. Get model probabilities at fixation locations
        # 4. Get baseline probabilities at fixation locations
        # 5. Compute IG = mean(log(P_model / P_baseline))
        #
        # Example:
        # # Normalize predictions to probabilities
        # pred_probs = F.softmax(predictions.view(predictions.size(0), -1), dim=1)
        # pred_probs = pred_probs.view_as(predictions)
        #
        # # Get fixation locations (non-zero entries in fixation maps)
        # fixation_locs = torch.nonzero(fixation_maps > 0)
        #
        # # Get probabilities at fixation locations
        # p_model = pred_probs[fixation_locs[:, 0], fixation_locs[:, 1],
        #                      fixation_locs[:, 2], fixation_locs[:, 3]]
        #
        # # Get baseline probabilities
        # prior_tensor = torch.from_numpy(self.center_prior.get_prior()).float()
        # p_baseline = prior_tensor[fixation_locs[:, 2], fixation_locs[:, 3]]
        #
        # # Compute IG
        # ig = torch.mean(torch.log2(p_model / (p_baseline + 1e-8)))
        #
        # return ig.item()
        pass


class BiasEntropyMeasurer:
    """Measure entropy of model's spatial bias."""

    def __init__(self, image_size: Tuple[int, int], num_samples: int = 16):
        """Initialize bias entropy measurer.

        Args:
            image_size: (width, height) of images
            num_samples: Number of uniform images to average
        """
        self.image_size = image_size
        self.num_samples = num_samples

    def measure_bias_entropy(
        self,
        model: torch.nn.Module,
        device: str = 'cuda'
    ) -> float:
        """Measure bias entropy of model.

        Args:
            model: Trained saliency model
            device: Device for computation

        Returns:
            Bias entropy value (bits)
        """
        # TODO: Implement bias entropy measurement
        # This should use the existing EntropyRegularizer!
        #
        # from models.entropy_regularizer import EntropyRegularizer
        #
        # regularizer = EntropyRegularizer(
        #     model=model,
        #     image_size=self.image_size,
        #     num_samples=self.num_samples,
        #     device=device
        # )
        #
        # # Extract bias map
        # bias_map = regularizer.bias_extractor.extract_bias_map(self.num_samples)
        #
        # # Normalize to probability
        # prob_map = regularizer.entropy_computer.normalize_to_probability(bias_map)
        #
        # # Compute entropy
        # entropy = regularizer.entropy_computer.compute_entropy(prob_map)
        #
        # return entropy.item()
        pass
```

### Code Template: `src/evaluation/evaluate_model.py`

```python
"""
Model evaluation script.

Evaluates trained models on validation and OOD datasets.
"""

import argparse
import torch
import yaml
import json
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.deepgaze3 import DeepGazeIII
from data.mit1003_loader import create_mit1003_dataloaders
from data.cat2000_loader import create_cat2000_dataloader
from evaluation.metrics import InformationGainComputer, BiasEntropyMeasurer


def load_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded model
    """
    # TODO: Implement checkpoint loading
    # checkpoint = torch.load(checkpoint_path, map_location=device)
    #
    # model = DeepGazeIII(pretrained=False)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # model = model.to(device)
    # model.eval()
    #
    # return model
    pass


def evaluate_on_dataset(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    ig_computer: InformationGainComputer,
    device: str = 'cuda'
) -> dict:
    """Evaluate model on a dataset.

    Args:
        model: Trained model
        dataloader: Data loader
        ig_computer: Information Gain computer
        device: Device for computation

    Returns:
        Dictionary with metrics
    """
    model.eval()

    total_ig = 0.0
    num_batches = 0

    # TODO: Implement evaluation loop
    # with torch.no_grad():
    #     for images, fixation_maps in tqdm(dataloader, desc="Evaluating"):
    #         images = images.to(device)
    #         fixation_maps = fixation_maps.to(device)
    #
    #         predictions = model(images)
    #         ig = ig_computer.compute_information_gain(predictions, fixation_maps)
    #
    #         total_ig += ig
    #         num_batches += 1

    metrics = {
        'information_gain': total_ig / num_batches if num_batches > 0 else 0.0,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate DeepGaze 3 model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['mit1003', 'cat2000'],
                       help='Dataset to evaluate on')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to dataset')
    parser.add_argument('--output', type=str, default='results.json',
                       help='Output file for results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_checkpoint(args.checkpoint, device)

    # Create data loader
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == 'mit1003':
        _, val_loader = create_mit1003_dataloaders(
            data_path=args.data_path,
            batch_size=8,
            num_workers=4
        )
        dataloader = val_loader
    else:  # cat2000
        dataloader = create_cat2000_dataloader(
            data_path=args.data_path,
            num_samples=50,
            batch_size=8,
            num_workers=4
        )

    # Create metric computers
    ig_computer = InformationGainComputer(image_size=(1024, 768))
    bias_measurer = BiasEntropyMeasurer(image_size=(1024, 768))

    # Evaluate
    print("Evaluating model...")
    metrics = evaluate_on_dataset(model, dataloader, ig_computer, device)

    # Measure bias entropy
    print("Measuring bias entropy...")
    bias_entropy = bias_measurer.measure_bias_entropy(model, device)
    metrics['bias_entropy'] = bias_entropy

    # Save results
    print(f"Saving results to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\nResults:")
    print(f"  Information Gain: {metrics['information_gain']:.4f} bits")
    print(f"  Bias Entropy: {metrics['bias_entropy']:.4f} bits")


if __name__ == '__main__':
    main()
```

### Testing Strategy for Task 7

Create `tests/test_metrics.py`:

```python
"""
Tests for evaluation metrics.

Following TDD approach: Write these tests FIRST.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from evaluation.metrics import GaussianCenterPrior, InformationGainComputer, BiasEntropyMeasurer


class TestGaussianCenterPrior:
    """Test Gaussian center prior."""

    def test_prior_normalization(self):
        """Test that prior sums to 1."""
        # TODO: Test probability normalization
        pass

    def test_prior_peak_at_center(self):
        """Test that prior has maximum at image center."""
        # TODO: Test Gaussian is centered
        pass


class TestInformationGain:
    """Test Information Gain computation."""

    def test_ig_shape(self):
        """Test that IG returns scalar."""
        # TODO: Test output shape
        pass

    def test_ig_toy_example(self):
        """Test IG on toy example with known result."""
        # TODO: Create toy example with known IG value
        pass


class TestBiasEntropy:
    """Test bias entropy measurement."""

    def test_entropy_measurement(self):
        """Test bias entropy measurement."""
        # TODO: Test entropy measurement on mock model
        pass
```

---

## Task 8: Final Evaluation and Report

### Overview

Execute final evaluation on both models and generate go/no-go decision report.

### Sub-tasks Breakdown

- **8.1**: Evaluate baseline on MIT1003 validation
- **8.2**: Evaluate entropy model on MIT1003 validation
- **8.3**: Evaluate baseline on CAT2000 OOD
- **8.4**: Evaluate entropy model on CAT2000 OOD
- **8.5**: Measure baseline bias entropy
- **8.6**: Measure entropy model bias entropy
- **8.7**: Generate performance comparison table
- **8.8**: Create training loss curves visualization
- **8.9**: Write go/no-go decision summary
- **8.10**: Document findings and recommendations

### Evaluation Execution

```bash
# Evaluate baseline on MIT1003
python src/evaluation/evaluate_model.py \
    --checkpoint outputs/checkpoints/deepgaze3_baseline/best_model.pt \
    --dataset mit1003 \
    --data-path /path/to/MIT1003 \
    --output results/baseline_mit1003.json

# Evaluate baseline on CAT2000
python src/evaluation/evaluate_model.py \
    --checkpoint outputs/checkpoints/deepgaze3_baseline/best_model.pt \
    --dataset cat2000 \
    --data-path /path/to/CAT2000 \
    --output results/baseline_cat2000.json

# Evaluate entropy model on MIT1003
python src/evaluation/evaluate_model.py \
    --checkpoint outputs/checkpoints/deepgaze3_entropy_reg/best_model.pt \
    --dataset mit1003 \
    --data-path /path/to/MIT1003 \
    --output results/entropy_mit1003.json

# Evaluate entropy model on CAT2000
python src/evaluation/evaluate_model.py \
    --checkpoint outputs/checkpoints/deepgaze3_entropy_reg/best_model.pt \
    --dataset cat2000 \
    --data-path /path/to/CAT2000 \
    --output results/entropy_cat2000.json
```

### Code Template: `src/evaluation/generate_report.py`

```python
"""
Generate final go/no-go decision report.

Compares baseline and entropy-regularized models.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def load_results(results_dir: str) -> dict:
    """Load all evaluation results."""
    results = {}

    # Load MIT1003 results
    with open(Path(results_dir) / 'baseline_mit1003.json', 'r') as f:
        results['baseline_mit1003'] = json.load(f)

    with open(Path(results_dir) / 'entropy_mit1003.json', 'r') as f:
        results['entropy_mit1003'] = json.load(f)

    # Load CAT2000 results
    with open(Path(results_dir) / 'baseline_cat2000.json', 'r') as f:
        results['baseline_cat2000'] = json.load(f)

    with open(Path(results_dir) / 'entropy_cat2000.json', 'r') as f:
        results['entropy_cat2000'] = json.load(f)

    return results


def compute_metrics(results: dict) -> dict:
    """Compute comparison metrics."""
    metrics = {}

    # Bias entropy increase
    baseline_entropy = results['baseline_mit1003']['bias_entropy']
    entropy_entropy = results['entropy_mit1003']['bias_entropy']
    entropy_increase_pct = ((entropy_entropy - baseline_entropy) / baseline_entropy) * 100

    # In-domain performance
    baseline_ig_mit = results['baseline_mit1003']['information_gain']
    entropy_ig_mit = results['entropy_mit1003']['information_gain']
    ig_degradation_pct = ((entropy_ig_mit - baseline_ig_mit) / baseline_ig_mit) * 100

    # OOD performance
    baseline_ig_cat = results['baseline_cat2000']['information_gain']
    entropy_ig_cat = results['entropy_cat2000']['information_gain']
    ig_ood_improvement_pct = ((entropy_ig_cat - baseline_ig_cat) / baseline_ig_cat) * 100

    metrics['entropy_increase_pct'] = entropy_increase_pct
    metrics['ig_degradation_pct'] = ig_degradation_pct
    metrics['ig_ood_improvement_pct'] = ig_ood_improvement_pct

    return metrics


def generate_comparison_table(results: dict, metrics: dict) -> str:
    """Generate markdown comparison table."""
    table = """
## Performance Comparison

| Metric | Baseline | Entropy-Reg | Change |
|--------|----------|-------------|--------|
| **Bias Entropy** | {:.4f} | {:.4f} | {:+.2f}% |
| **MIT1003 IG (in-domain)** | {:.4f} | {:.4f} | {:+.2f}% |
| **CAT2000 IG (OOD)** | {:.4f} | {:.4f} | {:+.2f}% |
""".format(
        results['baseline_mit1003']['bias_entropy'],
        results['entropy_mit1003']['bias_entropy'],
        metrics['entropy_increase_pct'],
        results['baseline_mit1003']['information_gain'],
        results['entropy_mit1003']['information_gain'],
        metrics['ig_degradation_pct'],
        results['baseline_cat2000']['information_gain'],
        results['entropy_cat2000']['information_gain'],
        metrics['ig_ood_improvement_pct']
    )

    return table


def evaluate_success_criteria(metrics: dict) -> dict:
    """Evaluate success criteria."""
    criteria = {
        'entropy_increase': {
            'threshold': 5.0,
            'actual': metrics['entropy_increase_pct'],
            'passed': metrics['entropy_increase_pct'] >= 5.0
        },
        'in_domain_degradation': {
            'threshold': -2.0,  # Allow up to 2% degradation
            'actual': metrics['ig_degradation_pct'],
            'passed': metrics['ig_degradation_pct'] >= -2.0
        },
        'ood_improvement': {
            'threshold': 0.0,  # Any improvement
            'actual': metrics['ig_ood_improvement_pct'],
            'passed': metrics['ig_ood_improvement_pct'] > 0.0
        }
    }

    return criteria


def make_go_no_go_decision(criteria: dict) -> tuple:
    """Make go/no-go decision.

    Returns:
        (decision, reasoning)
    """
    all_passed = all(c['passed'] for c in criteria.values())

    if all_passed:
        decision = "GO"
        reasoning = "All success criteria met. Proceed to full research project."
    else:
        decision = "NO-GO"
        failed = [name for name, c in criteria.items() if not c['passed']]
        reasoning = f"Failed criteria: {', '.join(failed)}. Do not proceed."

    return decision, reasoning


def generate_report(results_dir: str, output_file: str = 'GO_NO_GO_REPORT.md'):
    """Generate complete go/no-go report."""

    # Load results
    results = load_results(results_dir)

    # Compute metrics
    metrics = compute_metrics(results)

    # Evaluate criteria
    criteria = evaluate_success_criteria(metrics)

    # Make decision
    decision, reasoning = make_go_no_go_decision(criteria)

    # Generate comparison table
    comparison_table = generate_comparison_table(results, metrics)

    # Write report
    report = f"""# Go/No-Go Decision Report: Entropy Regularization Core Validation

**Date**: {Path().absolute()}
**Experiment**: Entropy Regularization Effect on DeepGaze 3

---

## Executive Summary

**Decision**: **{decision}**

**Reasoning**: {reasoning}

---

{comparison_table}

---

## Success Criteria Evaluation

### Primary Criterion: Bias Entropy Increase ≥ 5%

- **Threshold**: ≥ 5.0%
- **Actual**: {criteria['entropy_increase']['actual']:.2f}%
- **Status**: {'✅ PASSED' if criteria['entropy_increase']['passed'] else '❌ FAILED'}

### Secondary Criterion: In-Domain Performance Degradation ≤ 2%

- **Threshold**: ≥ -2.0%
- **Actual**: {criteria['in_domain_degradation']['actual']:.2f}%
- **Status**: {'✅ PASSED' if criteria['in_domain_degradation']['passed'] else '❌ FAILED'}

### Secondary Criterion: OOD Performance Improvement

- **Threshold**: > 0.0%
- **Actual**: {criteria['ood_improvement']['actual']:.2f}%
- **Status**: {'✅ PASSED' if criteria['ood_improvement']['passed'] else '❌ FAILED'}

---

## Detailed Results

### Bias Entropy

- **Baseline**: {results['baseline_mit1003']['bias_entropy']:.4f} bits
- **Entropy-Regularized**: {results['entropy_mit1003']['bias_entropy']:.4f} bits
- **Change**: {metrics['entropy_increase_pct']:+.2f}%

### MIT1003 Validation (In-Domain)

- **Baseline IG**: {results['baseline_mit1003']['information_gain']:.4f} bits
- **Entropy-Reg IG**: {results['entropy_mit1003']['information_gain']:.4f} bits
- **Change**: {metrics['ig_degradation_pct']:+.2f}%

### CAT2000 (Out-of-Distribution)

- **Baseline IG**: {results['baseline_cat2000']['information_gain']:.4f} bits
- **Entropy-Reg IG**: {results['entropy_cat2000']['information_gain']:.4f} bits
- **Change**: {metrics['ig_ood_improvement_pct']:+.2f}%

---

## Recommendations

"""

    if decision == "GO":
        report += """
### Proceed to Full Research Project

The core hypothesis has been validated. Entropy regularization successfully:
1. Increases bias entropy (more uniform spatial priors)
2. Maintains in-domain performance (acceptable degradation)
3. Improves out-of-distribution generalization

**Next Steps**:
1. Implement full cross-dataset evaluation (Phase 1)
2. Conduct lambda hyperparameter sweep (Phase 2.1)
3. Explore explicit bias modeling (Phase 1.2)
4. Investigate few-shot bias adaptation (Phase 3)

**Estimated Timeline**: 11 weeks (as per roadmap)
**Resources Required**: Multi-GPU cluster access, multiple datasets
"""
    else:
        report += """
### Do Not Proceed - Core Hypothesis Not Validated

The experiment failed to meet success criteria. Possible reasons:
1. Lambda = 1.0 may be too strong/weak
2. Uniform image approach may not capture bias effectively
3. DeepGaze 3 architecture may not benefit from entropy regularization

**Alternative Approaches**:
1. Hyperparameter sweep for lambda (0.1, 0.5, 1.0, 2.0, 5.0)
2. Different bias extraction methods (adversarial examples, natural images)
3. Explicit bias modeling with separate network
4. Different base model architecture

**Timeline**: 1-2 weeks for alternative exploration before abandoning project
"""

    report += "\n---\n\n*Generated automatically by src/evaluation/generate_report.py*\n"

    # Save report
    with open(output_file, 'w') as f:
        f.write(report)

    print(f"Report saved to {output_file}")
    print(f"\nDecision: {decision}")
    print(reasoning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate go/no-go report')
    parser.add_argument('--results-dir', type=str, default='outputs/results',
                       help='Directory containing evaluation results')
    parser.add_argument('--output', type=str, default='GO_NO_GO_REPORT.md',
                       help='Output report file')
    args = parser.parse_args()

    generate_report(args.results_dir, args.output)
```

---

## Testing Strategy

### Overall Testing Philosophy

Following Test-Driven Development (TDD):
1. **Write tests FIRST** before implementation
2. **Run tests** to verify they fail (red)
3. **Implement** code to make tests pass (green)
4. **Refactor** code while keeping tests passing
5. **Use test-runner subagent** as specified in process flow

### Test Organization

```
tests/
├── test_mit1003_loader.py        ✓ Complete (12 tests)
├── test_cat2000_loader.py        ✓ Complete (12 tests)
├── test_data_verification.py     ✓ Complete (8 tests)
├── test_distributed_loading.py   ✓ Complete (9 tests)
├── test_entropy_regularization.py ✓ Complete (23 tests)
├── test_trainer.py               ⏳ Task 4 (NEW - ~15 tests)
└── test_metrics.py               ⏳ Task 7 (NEW - ~10 tests)
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_trainer.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Test-Runner Subagent Usage

As specified in Step 6 of the process flow, ALWAYS use the `engineer:test-runner` subagent:

```
Step 6: Test verification - Use engineer:test-runner subagent
```

This subagent will:
- Run all relevant tests
- Analyze failures in detail
- Report which tests pass/fail
- Provide debugging information

---

## Integration Checklist

### Pre-Implementation Checklist

- [ ] Read PROGRESS_SUMMARY.md to understand current state
- [ ] Review Tasks 1-3 summaries to understand existing components
- [ ] Verify all existing tests pass (64/64 tests)
- [ ] Understand data loader API (`create_mit1003_dataloaders()`)
- [ ] Understand entropy regularizer API (`EntropyRegularizer`)
- [ ] Review DeepGaze 3 model architecture

### Task 4 Implementation Checklist

- [ ] **Sub-task 4.1**: Write all tests in `test_trainer.py`
  - [ ] Config loading tests
  - [ ] NLL loss tests
  - [ ] Checkpointing tests
  - [ ] Distributed setup tests
  - [ ] Training loop tests
- [ ] **Sub-task 4.2**: Verify DeepGaze 3 model loads (already done)
- [ ] **Sub-task 4.3**: Implement `_setup_scheduler()` with MultiStepLR
- [ ] **Sub-task 4.4**: Implement `compute_nll_loss()` method
- [ ] **Sub-task 4.5**: Implement `_setup_distributed()` for DDP
- [ ] **Sub-task 4.6**: Implement `save_checkpoint()` and loading
- [ ] **Sub-task 4.7**: Implement logging methods
- [ ] **Sub-task 4.8**: Complete all TODOs in trainer.py, create train_baseline.py
- [ ] **Sub-task 4.9**: Run smoke test (1 epoch, small dataset)
- [ ] **Sub-task 4.10**: Use test-runner subagent to verify tests
- [ ] **Sub-task 4.11**: Mark all Task 4 items [x] in tasks.md

### Task 5 Implementation Checklist

- [ ] **Sub-task 5.1**: Write entropy config tests
- [ ] **Sub-task 5.2**: Create `train_entropy_reg.py` (mostly copy of baseline)
- [ ] **Sub-task 5.3**: Verify `compute_every` parameter works
- [ ] **Sub-task 5.4**: Test combined loss computation
- [ ] **Sub-task 5.5**: Verify entropy logging
- [ ] **Sub-task 5.6**: Run smoke test with entropy regularization
- [ ] **Sub-task 5.7**: Use test-runner subagent
- [ ] **Sub-task 5.8**: Mark all Task 5 items [x] in tasks.md

### Task 6 Execution Checklist

- [ ] **Pre-training**:
  - [ ] Download MIT1003 dataset using `scripts/download_datasets.py`
  - [ ] Download CAT2000 dataset
  - [ ] Update data paths in config files
  - [ ] Verify environment with `scripts/smoke_test.py`
- [ ] **Sub-task 6.1**: Launch baseline job (`sbatch scripts/run_baseline.sh`)
- [ ] **Sub-task 6.2**: Launch entropy job (`sbatch scripts/run_entropy_reg.sh`)
- [ ] **Sub-task 6.3**: Monitor training progress
  - [ ] Check job status regularly (`squeue`)
  - [ ] Monitor training logs
  - [ ] Watch for NaN/divergence issues
- [ ] **Sub-task 6.4**: Verify both complete 25 epochs
- [ ] **Sub-task 6.5**: Verify checkpoints saved correctly
- [ ] **Sub-task 6.6**: Mark all Task 6 items [x] in tasks.md

### Task 7 Implementation Checklist

- [ ] **Sub-task 7.1**: Write IG tests in `test_metrics.py`
- [ ] **Sub-task 7.2**: Implement `InformationGainComputer` class
- [ ] **Sub-task 7.3**: Write bias entropy tests
- [ ] **Sub-task 7.4**: Implement `BiasEntropyMeasurer` class
- [ ] **Sub-task 7.5**: Create `evaluate_model.py` script
- [ ] **Sub-task 7.6**: Verify evaluation on both datasets works
- [ ] **Sub-task 7.7**: Use test-runner subagent
- [ ] **Sub-task 7.8**: Mark all Task 7 items [x] in tasks.md

### Task 8 Execution Checklist

- [ ] **Sub-task 8.1**: Evaluate baseline on MIT1003
- [ ] **Sub-task 8.2**: Evaluate entropy on MIT1003
- [ ] **Sub-task 8.3**: Evaluate baseline on CAT2000
- [ ] **Sub-task 8.4**: Evaluate entropy on CAT2000
- [ ] **Sub-task 8.5**: Measure baseline bias entropy
- [ ] **Sub-task 8.6**: Measure entropy bias entropy
- [ ] **Sub-task 8.7**: Run `generate_report.py`
- [ ] **Sub-task 8.8**: Create training curves visualization
- [ ] **Sub-task 8.9**: Review go/no-go decision
- [ ] **Sub-task 8.10**: Document findings
- [ ] **Sub-task 8.11**: Mark all Task 8 items [x] in tasks.md

### Final Verification

- [ ] All 8 tasks marked complete in tasks.md
- [ ] All tests passing (expected: ~90+ tests)
- [ ] GO_NO_GO_REPORT.md generated
- [ ] Training checkpoints saved
- [ ] Evaluation results saved
- [ ] Update PROGRESS_SUMMARY.md to 8/8 tasks complete (100%)

---

## Key Integration Points

### Using Existing Components

#### 1. Data Loaders (from Tasks 2)

```python
from data.mit1003_loader import create_mit1003_dataloaders

train_loader, val_loader = create_mit1003_dataloaders(
    data_path="/path/to/MIT1003",
    batch_size=32,
    num_workers=8,
    image_size=(1024, 768),
    world_size=4,  # For distributed training
    rank=0,
    seed=42
)

# Returns:
# - train_loader: 902 images
# - val_loader: 101 images
# - Tensors: images (B, 3, 768, 1024), fixation_maps (B, 1, 768, 1024)
```

#### 2. Entropy Regularizer (from Task 3)

```python
from models.entropy_regularizer import EntropyRegularizer

regularizer = EntropyRegularizer(
    model=model,
    image_size=(1024, 768),
    num_samples=16,
    device='cuda'
)

# During training:
entropy_loss, entropy_value = regularizer.compute_entropy_loss()
total_loss = nll_loss + lambda_entropy * entropy_loss
```

#### 3. DeepGaze 3 Model (from Task 1)

```python
from models.deepgaze3 import DeepGazeIII

model = DeepGazeIII(pretrained=False)
model = model.to(device)

# Forward pass:
predictions = model(images)  # Returns (B, 1, H, W) log-densities
```

### Configuration Loading

```python
import yaml

with open('configs/baseline_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Access parameters:
lr = config['training']['optimizer']['lr']
epochs = config['training']['epochs']
batch_size = config['training']['batch_size']
```

---

## Expected Outcomes

### Success Metrics

**Primary Success Criterion**:
- Bias entropy increase ≥ 5%

**Secondary Success Criteria**:
- In-domain IG degradation ≤ 2%
- OOD IG improvement > 0%
- Training stability (no NaN, no divergence)

### Timeline Estimates

- **Task 4** (Baseline Training): 1-2 days implementation
- **Task 5** (Entropy Training): 0.5-1 day implementation
- **Task 6** (Training Execution): 8-12 hours compute time
- **Task 7** (Evaluation Metrics): 1 day implementation
- **Task 8** (Final Report): 0.5 day

**Total**: 3-5 days (matches original 2-3 day estimate with buffer)

---

## Troubleshooting Guide

### Common Issues

#### 1. Out of Memory During Training

**Symptom**: CUDA out of memory error

**Solutions**:
- Reduce batch size in config (32 → 16)
- Reduce number of uniform samples for entropy (16 → 8)
- Enable gradient checkpointing in model

#### 2. NaN Losses

**Symptom**: Loss becomes NaN during training

**Solutions**:
- Check learning rate (reduce if too high)
- Add gradient clipping
- Verify NLL loss implementation (check for log(0))
- Verify entropy computation (check epsilon parameter)

#### 3. Distributed Training Hangs

**Symptom**: Training freezes on multi-GPU

**Solutions**:
- Check that all GPUs reach collective operations
- Verify `find_unused_parameters=False` in DDP
- Ensure entropy computation happens on all ranks or none

#### 4. Poor Convergence

**Symptom**: Validation loss not decreasing

**Solutions**:
- Verify data loading (check normalization)
- Check learning rate schedule
- Ensure model is not in eval mode during training
- Verify gradient flow (especially for entropy regularization)

---

## Summary

This implementation guide provides:

1. **Complete architecture** for training and evaluation pipelines
2. **Code templates** with clear TODOs for all remaining tasks
3. **Integration points** clearly marked for existing components
4. **Testing strategy** following TDD approach
5. **Step-by-step checklists** for systematic implementation

**Next Steps**:
1. Resume in fresh session with full context
2. Follow Task 4 → Task 5 → Task 6 → Task 7 → Task 8
3. Use test-runner subagent as specified in Step 6
4. Generate final GO_NO_GO_REPORT.md
5. Make informed decision about full research project

---

*Last Updated: 2025-10-22*
*Guide Status: Ready for implementation*
*Estimated Completion: 3-5 days*
