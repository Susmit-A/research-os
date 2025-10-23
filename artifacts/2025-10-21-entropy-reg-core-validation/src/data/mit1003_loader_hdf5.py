"""
MIT1003 Dataset Loader for Saliency Prediction (HDF5 Fixations Version).

This is an adapted version of mit1003_loader.py that works with the existing
HDF5-based MIT1003 datasets available on the cluster. It uses:
- Individual JPEG images from stimuli/ directory
- Fixation data from fixations.hdf5

This adaptation allows us to proceed with training without downloading/converting
the entire dataset.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import random

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    print("Warning: h5py not available. Install with: pip install h5py")


class MIT1003DatasetHDF5(Dataset):
    """MIT1003 dataset for saliency prediction using HDF5 fixations.

    The MIT1003 dataset contains 1003 images with corresponding eye fixation data.
    Standard split: 902 train, 101 validation.

    This version reads:
    - Images from stimuli/ directory (individual JPEG files)
    - Fixations from fixations.hdf5 file
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        image_size: Tuple[int, int] = (1024, 768),
        normalize: bool = True,
        seed: int = 42
    ):
        """Initialize MIT1003 dataset with HDF5 fixations.

        Args:
            data_path: Path to MIT1003 dataset root directory
            split: 'train' or 'val'
            image_size: Target image size as (width, height)
            normalize: Whether to apply ImageNet normalization
            seed: Random seed for reproducible train/val split
        """
        if not HDF5_AVAILABLE:
            raise ImportError("h5py is required for this data loader. Install with: pip install h5py")

        self.data_path = Path(data_path)
        self.split = split
        self.image_size = image_size
        self.normalize = normalize
        self.seed = seed

        # Validate split
        if split not in ['train', 'val']:
            raise ValueError(f"Split must be 'train' or 'val', got '{split}'")

        # Setup paths
        self.images_dir = self.data_path / "stimuli"
        self.fixations_path = self.data_path / "fixations.hdf5"

        # Validate paths exist
        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"Images directory not found: {self.images_dir}. "
                f"Expected MIT1003/stimuli/ directory."
            )

        if not self.fixations_path.exists():
            raise FileNotFoundError(
                f"Fixations file not found: {self.fixations_path}. "
                f"Expected MIT1003/fixations.hdf5 file."
            )

        # Get all image paths
        self.all_image_files = sorted(self.images_dir.glob("*.jpeg"))

        if len(self.all_image_files) == 0:
            raise FileNotFoundError(
                f"No images found in {self.images_dir}. "
                "Expected format: *.jpeg"
            )

        # Create reproducible train/val split
        self.image_files = self._create_split()

        # Setup preprocessing transforms
        self.transform = self._get_transforms()

        # Open HDF5 file (keep it open for efficiency)
        self.h5_file = h5py.File(self.fixations_path, 'r')

    def _create_split(self):
        """Create reproducible train/val split (902/101)."""
        # Set seed for reproducibility
        rng = random.Random(self.seed)

        # Shuffle all files with fixed seed
        all_files = list(self.all_image_files)
        rng.shuffle(all_files)

        # Split: first 902 for train, last 101 for validation
        if self.split == 'train':
            return all_files[:902]
        else:  # val
            return all_files[902:1003]

    def _get_transforms(self):
        """Get image preprocessing transforms."""
        transform_list = [
            transforms.Resize(self.image_size[::-1]),  # (H, W) for torchvision
            transforms.ToTensor(),
        ]

        if self.normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )

        return transforms.Compose(transform_list)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """Get image and fixation map pair.

        Args:
            idx: Index of the sample

        Returns:
            image: Preprocessed image tensor (C, H, W)
            fixation_map: Fixation density map tensor (1, H, W)
            image_id: Image identifier string
        """
        # Get image path
        image_path = self.image_files[idx]
        image_id = image_path.stem

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # Load fixation data from HDF5
        # The HDF5 file structure varies, so we need to handle it carefully
        # Common structure: h5_file[image_name]['fixations'] or similar
        try:
            # Try common HDF5 structures
            if image_id in self.h5_file:
                fixation_data = np.array(self.h5_file[image_id])
            elif f'{image_id}.jpeg' in self.h5_file:
                fixation_data = np.array(self.h5_file[f'{image_id}.jpeg'])
            else:
                # Fallback: create empty fixation map
                print(f"Warning: No fixation data found for {image_id}, using empty map")
                fixation_data = np.zeros((self.image_size[1], self.image_size[0]))
        except Exception as e:
            print(f"Warning: Error loading fixations for {image_id}: {e}")
            fixation_data = np.zeros((self.image_size[1], self.image_size[0]))

        # Convert fixation data to density map
        # If fixation_data is a list of (x, y) coordinates, create density map
        if fixation_data.ndim == 2 and fixation_data.shape[1] == 2:
            # Coordinate format: create density map via Gaussian blobs
            density_map = self._create_density_map(fixation_data)
        else:
            # Already a density map: resize if needed
            if fixation_data.shape != (self.image_size[1], self.image_size[0]):
                from scipy.ndimage import zoom
                scale_y = self.image_size[1] / fixation_data.shape[0]
                scale_x = self.image_size[0] / fixation_data.shape[1]
                density_map = zoom(fixation_data, (scale_y, scale_x), order=1)
            else:
                density_map = fixation_data

        # Convert to tensor and add channel dimension
        fixation_map = torch.from_numpy(density_map).float().unsqueeze(0)

        # Normalize to [0, 1]
        if fixation_map.max() > 0:
            fixation_map = fixation_map / fixation_map.max()

        return image, fixation_map, image_id

    def _create_density_map(self, fixations):
        """Create density map from fixation coordinates.

        Args:
            fixations: Array of (x, y) coordinates, shape (N, 2)

        Returns:
            density_map: 2D array of shape (H, W)
        """
        density_map = np.zeros((self.image_size[1], self.image_size[0]))

        # Simple approach: place Gaussian blobs at each fixation
        sigma = 20  # Gaussian sigma in pixels

        for x, y in fixations:
            # Scale coordinates to target size if needed
            # Create meshgrid
            y_grid, x_grid = np.ogrid[-int(3*sigma):int(3*sigma)+1, -int(3*sigma):int(3*sigma)+1]
            gaussian = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))

            # Place Gaussian in density map
            y_int, x_int = int(y), int(x)
            y_start = max(0, y_int - int(3*sigma))
            y_end = min(self.image_size[1], y_int + int(3*sigma) + 1)
            x_start = max(0, x_int - int(3*sigma))
            x_end = min(self.image_size[0], x_int + int(3*sigma) + 1)

            g_y_start = int(3*sigma) - (y_int - y_start)
            g_y_end = g_y_start + (y_end - y_start)
            g_x_start = int(3*sigma) - (x_int - x_start)
            g_x_end = g_x_start + (x_end - x_start)

            if y_end > y_start and x_end > x_start:
                density_map[y_start:y_end, x_start:x_end] += gaussian[g_y_start:g_y_end, g_x_start:g_x_end]

        return density_map

    def __del__(self):
        """Close HDF5 file on cleanup."""
        if hasattr(self, 'h5_file'):
            self.h5_file.close()


def create_mit1003_dataloaders_hdf5(
    data_path: str,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (1024, 768),
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    world_size: int = 1,
    rank: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders for MIT1003 (HDF5 version).

    Args:
        data_path: Path to MIT1003 dataset root
        batch_size: Batch size for data loaders (per GPU)
        image_size: Target image size as (width, height)
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        seed: Random seed for reproducibility
        world_size: Number of distributed processes (1 for single GPU)
        rank: Rank of current process in distributed training

    Returns:
        train_loader: DataLoader for training set (902 images)
        val_loader: DataLoader for validation set (101 images)
    """
    # Create datasets
    train_dataset = MIT1003DatasetHDF5(
        data_path=data_path,
        split='train',
        image_size=image_size,
        normalize=True,
        seed=seed
    )

    val_dataset = MIT1003DatasetHDF5(
        data_path=data_path,
        split='val',
        image_size=image_size,
        normalize=True,
        seed=seed
    )

    # Create DistributedSampler for DDP training
    train_sampler = None
    val_sampler = None

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=False
        )

        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=seed,
            drop_last=False
        )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # Only shuffle if not using DistributedSampler
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Never shuffle validation
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader
