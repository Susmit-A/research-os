"""
Evaluation script for CAT2000 OOD set.

Computes Information Gain metric on 50 randomly sampled CAT2000 images
for out-of-distribution generalization assessment.

Usage:
    python scripts/evaluate_cat2000.py --checkpoint path/to/model.pt --output results.json
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
from data.cat2000_loader import create_cat2000_dataloader
from evaluation.metrics import information_gain


def evaluate_cat2000(
    checkpoint_path: str,
    data_path: str,
    output_path: str = None,
    num_samples: int = 50,
    batch_size: int = 8,
    seed: int = 42,
    device: str = 'cuda'
):
    """
    Evaluate model on CAT2000 out-of-distribution set.

    Parameters
    ----------
    checkpoint_path : str
        Path to model checkpoint
    data_path : str
        Path to CAT2000 dataset
    output_path : str, optional
        Path to save results JSON
    num_samples : int, default=50
        Number of images to sample from CAT2000
    batch_size : int, default=8
        Batch size for evaluation
    seed : int, default=42
        Random seed for reproducible sampling
    device : str, default='cuda'
        Device to run evaluation on

    Returns
    -------
    results : dict
        Dictionary containing evaluation metrics
    """
    print("=" * 60)
    print("CAT2000 Out-of-Distribution Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data path: {data_path}")
    print(f"Num samples: {num_samples}")
    print(f"Seed: {seed}")
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

    # Create CAT2000 data loader
    print(f"\nLoading CAT2000 data (sampling {num_samples} images)...")
    cat_loader = create_cat2000_dataloader(
        data_path=data_path,
        num_samples=num_samples,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        seed=seed
    )
    print(f"CAT2000 sample size: {len(cat_loader.dataset)} images")

    # Evaluate
    print("\nEvaluating...")
    ig_values = []
    image_ids = []
    categories = []

    with torch.no_grad():
        for images, fixation_maps, ids in tqdm(cat_loader, desc="Evaluating"):
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

                # Extract category from image ID (if available)
                # CAT2000 format: category_imagename.jpg
                if '_' in ids[i]:
                    category = ids[i].split('_')[0]
                    categories.append(category)

    # Compute overall statistics
    ig_values = np.array(ig_values)
    mean_ig = np.mean(ig_values)
    std_ig = np.std(ig_values)
    median_ig = np.median(ig_values)

    print("\n" + "=" * 60)
    print("Overall Results")
    print("=" * 60)
    print(f"Number of images: {len(ig_values)}")
    print(f"Mean Information Gain: {mean_ig:.4f}")
    print(f"Std Information Gain: {std_ig:.4f}")
    print(f"Median Information Gain: {median_ig:.4f}")
    print(f"Min Information Gain: {ig_values.min():.4f}")
    print(f"Max Information Gain: {ig_values.max():.4f}")

    # Compute per-category statistics if categories available
    category_stats = {}
    if categories:
        unique_categories = list(set(categories))
        print("\n" + "=" * 60)
        print("Per-Category Results")
        print("=" * 60)

        for cat in sorted(unique_categories):
            cat_indices = [i for i, c in enumerate(categories) if c == cat]
            cat_ig = ig_values[cat_indices]

            category_stats[cat] = {
                'count': len(cat_ig),
                'mean': float(np.mean(cat_ig)),
                'std': float(np.std(cat_ig))
            }

            print(f"{cat:15s}: n={len(cat_ig):2d}, mean={np.mean(cat_ig):.4f}, std={np.std(cat_ig):.4f}")

    print("=" * 60)

    # Prepare results
    results = {
        'dataset': 'CAT2000',
        'split': 'OOD',
        'num_images': len(ig_values),
        'num_samples': num_samples,
        'seed': seed,
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
            'ig_values': ig_values.tolist(),
            'categories': categories if categories else None
        },
        'per_category': category_stats if category_stats else None
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
        description='Evaluate model on CAT2000 out-of-distribution set'
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
        help='Path to CAT2000 dataset'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save results JSON'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=50,
        help='Number of images to sample from CAT2000'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible sampling'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run evaluation on'
    )

    args = parser.parse_args()

    results = evaluate_cat2000(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        output_path=args.output,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device
    )

    return 0


if __name__ == '__main__':
    exit(main())
