"""
Evaluation script for MIT1003 validation set.

Computes Information Gain metric on 101 validation images.

Usage:
    python scripts/evaluate_mit1003.py --checkpoint path/to/model.pt --output results.json
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.deepgaze3 import DeepGazeIII
from data.mit1003_loader_hdf5 import create_mit1003_dataloaders_hdf5
from evaluation.metrics import information_gain


def evaluate_mit1003(
    checkpoint_path: str,
    data_path: str,
    output_path: str = None,
    batch_size: int = 8,
    device: str = 'cuda'
):
    """
    Evaluate model on MIT1003 validation set.

    Parameters
    ----------
    checkpoint_path : str
        Path to model checkpoint
    data_path : str
        Path to MIT1003 dataset
    output_path : str, optional
        Path to save results JSON
    batch_size : int, default=8
        Batch size for evaluation
    device : str, default='cuda'
        Device to run evaluation on

    Returns
    -------
    results : dict
        Dictionary containing evaluation metrics
    """
    print("=" * 60)
    print("MIT1003 Validation Set Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data path: {data_path}")
    print(f"Device: {device}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model = DeepGazeIII()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    print("Model loaded successfully")

    # Create validation data loader
    print("\nLoading validation data...")
    _, val_loader = create_mit1003_dataloaders_hdf5(
        data_path=data_path,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True
    )
    print(f"Validation set size: {len(val_loader.dataset)} images")

    # Evaluate
    print("\nEvaluating...")
    ig_values = []
    image_ids = []

    with torch.no_grad():
        for images, fixation_maps, ids in tqdm(val_loader, desc="Evaluating"):
            # Move to device
            images = images.to(device)

            # Forward pass
            predictions = model(images)

            # Move to CPU and convert to numpy
            predictions_np = predictions.cpu().numpy()
            fixation_maps_np = fixation_maps.numpy()

            # Compute IG for each image in batch
            for i in range(len(images)):
                pred = predictions_np[i]
                fixmap = fixation_maps_np[i, 0]  # Remove channel dim

                # Normalize to probability distributions
                pred_prob = np.exp(pred - pred.max())  # Log-density to density
                pred_prob = pred_prob / pred_prob.sum()
                fixmap_prob = fixmap / (fixmap.sum() + 1e-10)

                # Compute IG
                ig = information_gain(fixmap_prob, pred_prob, baseline='center')
                ig_values.append(ig)
                image_ids.append(ids[i])

    # Compute statistics
    ig_values = np.array(ig_values)
    mean_ig = np.mean(ig_values)
    std_ig = np.std(ig_values)
    median_ig = np.median(ig_values)

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Number of images: {len(ig_values)}")
    print(f"Mean Information Gain: {mean_ig:.4f}")
    print(f"Std Information Gain: {std_ig:.4f}")
    print(f"Median Information Gain: {median_ig:.4f}")
    print(f"Min Information Gain: {ig_values.min():.4f}")
    print(f"Max Information Gain: {ig_values.max():.4f}")
    print("=" * 60)

    # Prepare results
    results = {
        'dataset': 'MIT1003',
        'split': 'validation',
        'num_images': len(ig_values),
        'checkpoint': checkpoint_path,
        'metrics': {
            'information_gain': {
                'mean': float(mean_ig),
                'std': float(std_ig),
                'median': float(median_ig),
                'min': float(ig_values.min()),
                'max': float(ig_values.max())
            }
        },
        'per_image': {
            'image_ids': image_ids,
            'ig_values': ig_values.tolist()
        }
    }

    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate model on MIT1003 validation set'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to MIT1003 dataset'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save results JSON'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run evaluation on'
    )

    args = parser.parse_args()

    results = evaluate_mit1003(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device
    )

    return 0


if __name__ == '__main__':
    exit(main())
