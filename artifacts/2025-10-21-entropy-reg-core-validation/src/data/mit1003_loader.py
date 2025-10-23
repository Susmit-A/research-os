"""
MIT1003 Dataset Loader for Saliency Prediction.

Implements PyTorch Dataset class for MIT1003 benchmark dataset with:
- 902/101 train/validation split
- Fixation map loading
- Image preprocessing (resize to 1024×768, ImageNet normalization)
- Reproducible splits with seed control
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import random


class MIT1003Dataset(Dataset):
    """MIT1003 dataset for saliency prediction.

    The MIT1003 dataset contains 1003 images with corresponding eye fixation data.
    Standard split: 902 train, 101 validation.
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        image_size: Tuple[int, int] = (1024, 768),
        normalize: bool = True,
        seed: int = 42
    ):
        """Initialize MIT1003 dataset.

        Args:
            data_path: Path to MIT1003 dataset root directory
            split: 'train' or 'val'
            image_size: Target image size as (width, height)
            normalize: Whether to apply ImageNet normalization
            seed: Random seed for reproducible train/val split
        """
        self.data_path = Path(data_path)
        self.split = split
        self.image_size = image_size
        self.normalize = normalize
        self.seed = seed

        # Validate split
        if split not in ['train', 'val']:
            raise ValueError(f"Split must be 'train' or 'val', got '{split}'")

        # Setup paths
        self.images_dir = self.data_path / "ALLSTIMULI"
        self.fixations_dir = self.data_path / "ALLFIXATIONMAPS"

        # Get all image paths
        self.all_image_files = sorted(self.images_dir.glob("i*.jpeg"))

        if len(self.all_image_files) == 0:
            raise FileNotFoundError(
                f"No images found in {self.images_dir}. "
                "Expected format: i1.jpeg, i2.jpeg, ..."
            )

        # Create reproducible train/val split
        self.image_files = self._create_split()

        # Setup preprocessing transforms
        self.transform = self._get_transforms()

    def _create_split(self):
        """Create reproducible train/val split (902/101)."""
        # Set seed for reproducibility
        rng = random.Random(self.seed)

        # Shuffle all files with fixed seed
        all_files = list(self.all_image_files)
        rng.shuffle(all_files)

        # Split: first 902 for train, last 101 for val
        if self.split == "train":
            return all_files[:902]
        else:  # val
            return all_files[902:1003]

    def _get_transforms(self):
        """Get image preprocessing transforms."""
        transform_list = [
            transforms.Resize(self.image_size[::-1]),  # PIL uses (height, width)
            transforms.ToTensor(),  # Converts to [0, 1] and (C, H, W)
        ]

        if self.normalize:
            # ImageNet normalization
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )

        return transforms.Compose(transform_list)

    def _load_fixation_map(self, image_filename: str) -> torch.Tensor:
        """Load corresponding fixation map for an image.

        Args:
            image_filename: Name of image file (e.g., 'i1.jpeg')

        Returns:
            Fixation map as torch.Tensor of shape (1, H, W) in range [0, 1]
        """
        # Extract image number from filename (e.g., 'i1.jpeg' -> '1')
        img_num = image_filename.stem[1:]  # Remove 'i' prefix and extension

        # Fixation map filename format: i{num}_fixMap.jpg
        fix_filename = f"i{img_num}_fixMap.jpg"
        fix_path = self.fixations_dir / fix_filename

        if not fix_path.exists():
            raise FileNotFoundError(
                f"Fixation map not found: {fix_path}"
            )

        # Load fixation map
        fix_map = Image.open(fix_path).convert('L')  # Convert to grayscale

        # Resize to match target image size
        fix_map = fix_map.resize(self.image_size, Image.BILINEAR)

        # Convert to numpy array and normalize to [0, 1]
        fix_array = np.array(fix_map, dtype=np.float32) / 255.0

        # Convert to torch tensor and add channel dimension (1, H, W)
        fix_tensor = torch.from_numpy(fix_array).unsqueeze(0)

        return fix_tensor

    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (image, fixation_map)
            - image: torch.Tensor of shape (3, H, W)
            - fixation_map: torch.Tensor of shape (1, H, W)
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        # Get image path
        img_path = self.image_files[idx]

        # Load and preprocess image
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")

        # Load corresponding fixation map
        try:
            fixation_map = self._load_fixation_map(img_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load fixation map for {img_path}: {e}")

        return image, fixation_map

    def get_image_id(self, idx: int) -> str:
        """Get the image ID for a given index.

        Args:
            idx: Index of the sample

        Returns:
            Image ID (e.g., 'i1')
        """
        return self.image_files[idx].stem


def create_mit1003_dataloaders(
    data_path: str,
    batch_size: int = 32,
    num_workers: int = 8,
    image_size: Tuple[int, int] = (1024, 768),
    normalize: bool = True,
    seed: int = 42,
    world_size: int = 1,
    rank: int = 0
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation dataloaders for MIT1003.

    Args:
        data_path: Path to MIT1003 dataset
        batch_size: Batch size for dataloaders (per GPU)
        num_workers: Number of worker processes for data loading
        image_size: Target image size (width, height)
        normalize: Whether to apply ImageNet normalization
        seed: Random seed for reproducibility
        world_size: Number of distributed processes (1 for single GPU)
        rank: Rank of current process in distributed training

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    train_dataset = MIT1003Dataset(
        data_path=data_path,
        split="train",
        image_size=image_size,
        normalize=normalize,
        seed=seed
    )

    val_dataset = MIT1003Dataset(
        data_path=data_path,
        split="val",
        image_size=image_size,
        normalize=normalize,
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

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # Only shuffle if not using DistributedSampler
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Never shuffle validation
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_dataloader, val_dataloader
