"""
Example of integrating entropy regularization into training loop.

This demonstrates how to use the EntropyRegularizer module during training
to maximize the entropy of the model's center bias.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# Add models to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.entropy_regularizer import EntropyRegularizer


def train_with_entropy_regularization(
    model,
    train_loader,
    optimizer,
    num_epochs=25,
    lambda_entropy=1.0,
    device='cuda'
):
    """Training loop with entropy regularization.

    Args:
        model: DeepGaze model
        train_loader: DataLoader for training data
        optimizer: Optimizer (e.g., Adam)
        num_epochs: Number of training epochs
        lambda_entropy: Weight for entropy regularization loss
        device: Device to train on
    """
    # Initialize entropy regularizer
    entropy_regularizer = EntropyRegularizer(
        model=model,
        image_size=(1024, 768),
        num_samples=16,
        device=device
    )

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_nll = 0.0
        total_entropy_reg = 0.0

        for batch_idx, (images, fixation_maps) in enumerate(train_loader):
            images = images.to(device)
            fixation_maps = fixation_maps.to(device)

            optimizer.zero_grad()

            # Forward pass
            predictions = model(images)

            # Compute NLL loss (negative log-likelihood)
            nll_loss = compute_nll_loss(predictions, fixation_maps)

            # Compute entropy regularization loss (every N batches to save computation)
            if batch_idx % 50 == 0:  # Compute entropy every 50 batches
                entropy_loss, entropy_value = entropy_regularizer.compute_entropy_loss()

                # Total loss: NLL + lambda * (-Entropy)
                # Note: entropy_loss is already negative, so we minimize it
                total_batch_loss = nll_loss + lambda_entropy * entropy_loss

                # Log entropy value
                print(f"Epoch {epoch}, Batch {batch_idx}: "
                      f"NLL={nll_loss.item():.4f}, "
                      f"Entropy={entropy_value:.4f}, "
                      f"Total Loss={total_batch_loss.item():.4f}")

                total_entropy_reg += entropy_value
            else:
                # Use only NLL loss for other batches
                total_batch_loss = nll_loss

            # Backward pass
            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item()
            total_nll += nll_loss.item()

        # Epoch summary
        avg_loss = total_loss / len(train_loader)
        avg_nll = total_nll / len(train_loader)
        avg_entropy = total_entropy_reg / (len(train_loader) // 50)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Average Total Loss: {avg_loss:.4f}")
        print(f"  Average NLL Loss: {avg_nll:.4f}")
        print(f"  Average Entropy: {avg_entropy:.4f}\n")


def compute_nll_loss(predictions, fixation_maps):
    """Compute negative log-likelihood loss on fixation data.

    Args:
        predictions: Model predictions (log-densities)
        fixation_maps: Ground truth fixation maps

    Returns:
        NLL loss value
    """
    # Convert predictions to probabilities
    log_probs = torch.log_softmax(predictions.view(predictions.size(0), -1), dim=1)
    log_probs = log_probs.view_as(predictions)

    # Normalize fixation maps to sum to 1 (probability distribution)
    fixation_probs = fixation_maps / (fixation_maps.sum(dim=(2, 3), keepdim=True) + 1e-8)

    # Compute negative log-likelihood
    nll = -torch.sum(fixation_probs * log_probs, dim=(2, 3)).mean()

    return nll


def example_training_script():
    """Example script showing complete training setup."""
    import torch.optim as optim
    from models.deepgaze3 import DeepGazeIII
    from data.mit1003_loader import create_mit1003_dataloaders

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = DeepGazeIII(pretrained=False).to(device)

    # Create data loaders
    train_loader, val_loader = create_mit1003_dataloaders(
        data_path="/path/to/MIT1003",
        batch_size=32,
        num_workers=8
    )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001585)

    # Train with entropy regularization
    train_with_entropy_regularization(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        num_epochs=25,
        lambda_entropy=1.0,  # Entropy regularization weight
        device=device
    )


if __name__ == "__main__":
    # This is just an example - actual training scripts are in:
    # - src/training/train_baseline.py (no entropy regularization)
    # - src/training/train_entropy_reg.py (with entropy regularization)
    print("This is an example training integration.")
    print("See src/training/train_baseline.py and train_entropy_reg.py for actual scripts.")
