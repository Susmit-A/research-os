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
    parser = argparse.ArgumentParser(description='Train baseline DeepGaze 3')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    args = parser.parse_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        print("=" * 60)
        print("Baseline DeepGaze 3 Training")
        print("=" * 60)
        print(f"Config: {args.config}")
        print(f"World Size: {world_size}")
        print(f"Rank: {rank}")
        print("=" * 60)

    # Initialize trainer (baseline - no entropy regularization)
    trainer = Trainer(
        config_path=args.config,
        rank=rank,
        world_size=world_size,
        use_entropy_regularization=False  # Baseline
    )

    # Start training
    num_epochs = args.epochs if args.epochs is not None else trainer.config['training']['epochs']

    if rank == 0:
        print(f"\nStarting training for {num_epochs} epochs...")

    trainer.train(num_epochs)

    if rank == 0:
        print("\nTraining complete!")
        print(f"Best validation loss: {trainer.best_val_loss:.4f}")

    # Cleanup distributed training
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
