"""
Training framework for DeepGaze 3 models.

Supports both baseline and entropy-regularized training.
"""

import os
import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{rank}')
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cpu')

        # Initialize distributed training if needed
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
        dist.init_process_group(
            backend=self.config['training']['distributed']['backend'],
            init_method='env://',
            world_size=self.world_size,
            rank=self.rank
        )

    def _setup_model(self) -> nn.Module:
        """Setup DeepGaze 3 model."""
        model = DeepGazeIII(pretrained=self.config['model']['pretrained'])
        model = model.to(self.device)

        if self.world_size > 1:
            model = DDP(
                model,
                device_ids=[self.rank],
                find_unused_parameters=self.config['training']['distributed'].get(
                    'find_unused_parameters', False
                )
            )

        return model

    def _setup_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Setup MIT1003 data loaders."""
        data_config = self.config['data']

        # Skip actual data loading if path doesn't exist (for testing)
        if not Path(data_config['data_path']).exists():
            return None, None

        train_loader, val_loader = create_mit1003_dataloaders(
            data_path=data_config['data_path'],
            batch_size=self.config['training']['batch_size'],
            num_workers=data_config.get('num_workers', 8),
            image_size=tuple(data_config['image_size']),
            world_size=self.world_size,
            rank=self.rank,
            seed=self.config['experiment']['seed']
        )

        return train_loader, val_loader

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup Adam optimizer."""
        opt_config = self.config['training']['optimizer']
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=opt_config['lr'],
            betas=opt_config['betas'],
            eps=opt_config['eps'],
            weight_decay=opt_config['weight_decay']
        )
        return optimizer

    def _setup_scheduler(self):
        """Setup MultiStep LR scheduler."""
        sched_config = self.config['training']['lr_scheduler']
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=sched_config['milestones'],
            gamma=sched_config['gamma']
        )
        return scheduler

    def _setup_entropy_regularizer(self) -> Optional[EntropyRegularizer]:
        """Setup entropy regularizer."""
        if not self.use_entropy_reg:
            return None

        ent_config = self.config['training']['entropy_regularization']
        regularizer = EntropyRegularizer(
            model=self.model,
            image_size=tuple(self.config['data']['image_size']),
            num_samples=ent_config['num_uniform_samples'],
            device=self.device
        )
        return regularizer

    def _setup_logging(self) -> Path:
        """Setup logging directory."""
        exp_name = self.config['experiment']['name']
        log_dir = Path('outputs/logs') / exp_name
        if self.rank == 0:  # Only main process creates directories
            log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

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
        # Convert predictions to log-probabilities using log_softmax
        batch_size = predictions.size(0)
        log_probs = F.log_softmax(predictions.view(batch_size, -1), dim=1)
        log_probs = log_probs.view_as(predictions)

        # Normalize fixation maps to probability distributions
        fixation_sum = fixation_maps.sum(dim=(2, 3), keepdim=True) + 1e-8
        fixation_probs = fixation_maps / fixation_sum

        # Compute NLL: -sum(p_fixation * log_p_prediction)
        nll = -torch.sum(fixation_probs * log_probs, dim=(2, 3)).mean()

        return nll

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

        # Skip if no data loaders (testing mode)
        if self.train_loader is None:
            return {'train_loss': 0.0, 'train_nll': 0.0}

        for batch_idx, (images, fixation_maps) in enumerate(self.train_loader):
            images = images.to(self.device)
            fixation_maps = fixation_maps.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            # For saliency-only mode, use default inputs
            centerbias = torch.zeros_like(fixation_maps)
            # Use dummy fixation history (DeepGaze 3 expects 4 previous fixations)
            batch_size = images.size(0)
            height, width = images.size(2), images.size(3)
            # 4 dummy fixations at center for each image
            x_hist = torch.full((batch_size, 4), width / 2.0, device=self.device)
            y_hist = torch.full((batch_size, 4), height / 2.0, device=self.device)
            predictions = self.model(images, centerbias, x_hist, y_hist)

            # Compute NLL loss
            nll_loss = self.compute_nll_loss(predictions, fixation_maps)

            # Compute entropy regularization if enabled
            if self.use_entropy_reg and batch_idx % self.config['training']['entropy_regularization']['compute_every'] == 0:
                entropy_loss, entropy_value = self.entropy_regularizer.compute_entropy_loss()
                lambda_entropy = self.config['training']['loss']['entropy_weight']
                total_batch_loss = nll_loss + lambda_entropy * entropy_loss

                total_entropy_loss += entropy_loss.item()
                total_entropy_value += entropy_value
                num_entropy_computations += 1
            else:
                total_batch_loss = nll_loss

            # Backward pass
            total_batch_loss.backward()
            self.optimizer.step()

            total_loss += total_batch_loss.item()
            total_nll += nll_loss.item()
            self.global_step += 1

            # Logging
            if batch_idx % self.config['training']['logging']['log_every'] == 0:
                self._log_training_step(
                    epoch, batch_idx, nll_loss.item(),
                    entropy_value if self.use_entropy_reg and num_entropy_computations > 0 else None
                )

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

        # Skip if no data loaders (testing mode)
        if self.val_loader is None:
            return {'val_loss': 0.0}

        with torch.no_grad():
            for images, fixation_maps in self.val_loader:
                images = images.to(self.device)
                fixation_maps = fixation_maps.to(self.device)

                # For saliency-only mode, use default inputs
                centerbias = torch.zeros_like(fixation_maps)
                # Use dummy fixation history (DeepGaze 3 expects 4 previous fixations)
                batch_size = images.size(0)
                height, width = images.size(2), images.size(3)
                # 4 dummy fixations at center for each image
                x_hist = torch.full((batch_size, 4), width / 2.0, device=self.device)
                y_hist = torch.full((batch_size, 4), height / 2.0, device=self.device)
                predictions = self.model(images, centerbias, x_hist, y_hist)
                val_loss = self.compute_nll_loss(predictions, fixation_maps)

                total_val_loss += val_loss.item()

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
        if self.rank != 0:  # Only main process saves
            return

        checkpoint_dir = Path('outputs/checkpoints') / self.config['experiment']['name']
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Get model state dict (unwrap DDP if needed)
        if isinstance(self.model, DDP):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }

        # Save regular checkpoint
        if epoch % self.config['training']['checkpoint']['save_every'] == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best and self.config['training']['checkpoint'].get('save_best', True):
            best_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)

        # Save last checkpoint
        if self.config['training']['checkpoint'].get('save_last', True):
            last_path = checkpoint_dir / 'last_checkpoint.pt'
            torch.save(checkpoint, last_path)

    def _log_training_step(self, epoch, batch_idx, nll_loss, entropy_value=None):
        """Log training step information."""
        if self.rank == 0:  # Only log from main process
            log_msg = f"Epoch {epoch}, Batch {batch_idx}: NLL={nll_loss:.4f}"
            if entropy_value is not None:
                log_msg += f", Entropy={entropy_value:.4f}"
            print(log_msg)

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

            self.current_epoch = epoch + 1

    def _log_epoch_summary(self, epoch, train_metrics, val_metrics):
        """Log epoch summary."""
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
        print(f"  Train NLL: {train_metrics['train_nll']:.4f}")
        if 'train_entropy_value' in train_metrics:
            print(f"  Train Entropy: {train_metrics['train_entropy_value']:.4f}")
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"  LR: {self.scheduler.get_last_lr()[0]:.6f}\n")
