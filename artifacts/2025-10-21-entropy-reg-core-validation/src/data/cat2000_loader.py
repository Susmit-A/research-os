"""
CAT2000 Dataset Loader for Out-of-Distribution (OOD) Evaluation.

Implements PyTorch Dataset class for CAT2000 dataset with:
- Random sampling of 50 images for OOD evaluation
- Fixation map loading
- Image preprocessing (resize to 1024×768, ImageNet normalization)
- Reproducible sampling with seed control
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple, List
import random


class CAT2000Dataset(Dataset):
    """CAT2000 dataset for OOD saliency evaluation.

    The CAT2000 dataset contains 2000 images across 20 categories.
    For OOD evaluation, we randomly sample 50 images.
    """

    def __init__(
        self,
        data_path: str,
        num_samples: int = 50,
        image_size: Tuple[int, int] = (1024, 768),
        normalize: bool = True,
        seed: int = 42
    ):
        """Initialize CAT2000 dataset for OOD evaluation.

        Args:
            data_path: Path to CAT2000 dataset root directory
            num_samples: Number of images to randomly sample (default: 50)
            image_size: Target image size as (width, height)
            normalize: Whether to apply ImageNet normalization
            seed: Random seed for reproducible sampling
        """
        self.data_path = Path(data_path)
        self.num_samples = num_samples
        self.image_size = image_size
        self.normalize = normalize
        self.seed = seed

        # Setup paths
        # CAT2000 structure: Stimuli/{category}/Output_{category}_{num}.jpg
        # Fixations: FIXATIONMAPS/{category}/fixMap_{category}_{num}.jpg
        self.stimuli_dir = self.data_path / "Stimuli"
        self.fixations_dir = self.data_path / "FIXATIONMAPS"

        # Collect all available images
        self.all_image_files = self._collect_all_images()

        if len(self.all_image_files) == 0:
            raise FileNotFoundError(
                f"No images found in {self.stimuli_dir}. "
                "Expected CAT2000 directory structure with category subdirectories."
            )

        # Randomly sample num_samples images
        self.sampled_image_files = self._sample_images()

        # Setup preprocessing transforms
        self.transform = self._get_transforms()

    def _collect_all_images(self) -> List[Path]:
        """Collect all image files from CAT2000 categories.

        Returns:
            List of Path objects for all found images
        """
        all_images = []

        # CAT2000 has 20 categories
        # Iterate through all subdirectories in Stimuli
        if not self.stimuli_dir.exists():
            return all_images

        for category_dir in sorted(self.stimuli_dir.iterdir()):
            if category_dir.is_dir():
                # Find all Output_*.jpg images in this category
                category_images = list(category_dir.glob("Output_*.jpg"))
                all_images.extend(category_images)

        return all_images

    def _sample_images(self) -> List[Path]:
        """Randomly sample num_samples images from all available images.

        Returns:
            List of sampled image paths
        """
        # Set seed for reproducibility
        rng = random.Random(self.seed)

        # Sample without replacement
        if len(self.all_image_files) < self.num_samples:
            raise ValueError(
                f"Requested {self.num_samples} samples but only "
                f"{len(self.all_image_files)} images available"
            )

        sampled = rng.sample(self.all_image_files, self.num_samples)
        return sorted(sampled)  # Sort for consistency

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

    def _load_fixation_map(self, image_path: Path) -> torch.Tensor:
        """Load corresponding fixation map for an image.

        Args:
            image_path: Path to image file

        Returns:
            Fixation map as torch.Tensor of shape (1, H, W) in range [0, 1]
        """
        # Extract category and image number from path
        # Format: Stimuli/{category}/Output_{category}_{num}.jpg
        category = image_path.parent.name
        filename = image_path.stem  # Output_{category}_{num}

        # Parse to get fixation map name
        # Fixation format: fixMap_{category}_{num}.jpg
        parts = filename.split('_')
        if len(parts) >= 3:
            img_num = parts[-1]
            fix_filename = f"fixMap_{category}_{img_num}.jpg"
        else:
            raise ValueError(f"Unexpected image filename format: {filename}")

        fix_path = self.fixations_dir / category / fix_filename

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
        return len(self.sampled_image_files)

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
        img_path = self.sampled_image_files[idx]

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
            Image ID (category and number)
        """
        img_path = self.sampled_image_files[idx]
        category = img_path.parent.name
        return f"{category}_{img_path.stem}"


def create_cat2000_dataloader(
    data_path: str,
    num_samples: int = 50,
    batch_size: int = 16,
    num_workers: int = 8,
    image_size: Tuple[int, int] = (1024, 768),
    normalize: bool = True,
    seed: int = 42
) -> torch.utils.data.DataLoader:
    """Create dataloader for CAT2000 OOD evaluation.

    Args:
        data_path: Path to CAT2000 dataset
        num_samples: Number of images to sample
        batch_size: Batch size for dataloader
        num_workers: Number of worker processes for data loading
        image_size: Target image size (width, height)
        normalize: Whether to apply ImageNet normalization
        seed: Random seed for reproducibility

    Returns:
        DataLoader for CAT2000 OOD evaluation
    """
    dataset = CAT2000Dataset(
        data_path=data_path,
        num_samples=num_samples,
        image_size=image_size,
        normalize=normalize,
        seed=seed
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for evaluation
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return dataloader
