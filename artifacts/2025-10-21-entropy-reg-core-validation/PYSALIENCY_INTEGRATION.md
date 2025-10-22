# Pysaliency Integration Guide

## Overview

The `pysaliency` library provides automatic download and loading of MIT1003 and CAT2000 datasets. This document explains how to use pysaliency to download datasets and integrate them with our custom data loaders.

---

## Installation

Pysaliency is already installed in the deepgaze environment:

```bash
# Already installed, but if needed:
/mnt/lustre/work/bethge/bkr710/.conda/deepgaze/bin/pip install pysaliency
```

---

## Available Dataset Functions

Pysaliency provides the following functions for our datasets:

### MIT1003
- `pysaliency.external_datasets.get_mit1003()` - Standard MIT1003 (1003 images)
- `pysaliency.external_datasets.get_mit1003_onesize()` - MIT1003 with uniform size
- `pysaliency.external_datasets.get_mit1003_with_initial_fixation()` - With initial fixation data

### CAT2000
- `pysaliency.external_datasets.get_cat2000_train()` - CAT2000 training set (~1600 images)
- `pysaliency.external_datasets.get_cat2000_test()` - CAT2000 test set (~400 images)

---

## Downloading Datasets

### Option 1: Using the Download Script (Recommended)

We've created a convenient script to download both datasets:

```bash
cd scripts
/mnt/lustre/work/bethge/bkr710/.conda/deepgaze/bin/python download_datasets.py \
    --output_dir /path/to/store/datasets
```

**Options:**
- `--output_dir PATH` - Where to store datasets (default: `../data`)
- `--mit1003_only` - Download only MIT1003
- `--cat2000_only` - Download only CAT2000

### Option 2: Direct Python Usage

```python
import pysaliency

# Specify where datasets should be stored
dataset_location = '/path/to/datasets'

# Download MIT1003
mit_stimuli, mit_fixations = pysaliency.external_datasets.get_mit1003(
    location=dataset_location
)

# Download CAT2000 (train set)
cat_stimuli, cat_fixations = pysaliency.external_datasets.get_cat2000_train(
    location=dataset_location
)
```

---

## Pysaliency Data Format

### Stimuli Object
```python
# Access images
image = mit_stimuli.stimuli[0]  # First image as numpy array
n_images = len(mit_stimuli)     # Total number of images

# Image attributes
stimuli.filenames              # List of original filenames
stimuli.sizes                  # Image sizes
```

### Fixations Object
```python
# Access fixation data
fixations.x                    # X coordinates of all fixations
fixations.y                    # Y coordinates of all fixations
fixations.n                    # Image index for each fixation
fixations.subjects             # Subject IDs
fixations.lengths              # Fixation durations (if available)
```

---

## Integrating with Our Custom Data Loaders

Our custom data loaders (`src/data/mit1003_loader.py` and `src/data/cat2000_loader.py`) expect a specific directory structure. Here are two approaches:

### Approach 1: Adapt Our Loaders (Recommended)

Keep our existing loaders but point them to pysaliency's downloaded data:

1. **Download datasets with pysaliency** (stores in cache directory)
2. **Find the cache location**:
   ```python
   import pysaliency
   print(pysaliency.external_datasets.get_mit1003(location='/tmp').__dict__)
   # This will show where files are actually stored
   ```
3. **Update config files** with the actual data path

### Approach 2: Create Pysaliency Wrapper Loader

Create a new loader that wraps pysaliency's data format:

```python
# src/data/pysaliency_loader.py
import pysaliency
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class PysaliencyMIT1003Dataset(Dataset):
    """MIT1003 dataset using pysaliency backend."""

    def __init__(self, location, split='train', image_size=(1024, 768),
                 normalize=True, seed=42):
        # Load using pysaliency
        stimuli, fixations = pysaliency.external_datasets.get_mit1003(location)

        # Create train/val split (902/101)
        import random
        rng = random.Random(seed)
        indices = list(range(len(stimuli)))
        rng.shuffle(indices)

        if split == 'train':
            self.indices = indices[:902]
        else:
            self.indices = indices[902:1003]

        self.stimuli = stimuli
        self.fixations = fixations
        self.transform = self._get_transforms(image_size, normalize)

    def _get_transforms(self, image_size, normalize):
        transforms_list = [
            transforms.ToPILImage(),
            transforms.Resize(image_size[::-1]),
            transforms.ToTensor(),
        ]
        if normalize:
            transforms_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        return transforms.Compose(transforms_list)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_idx = self.indices[idx]

        # Get image
        image = self.stimuli.stimuli[img_idx]
        image = self.transform(image)

        # Get fixation map (need to create from fixation points)
        # This requires converting fixations.x, fixations.y to a map
        # For now, return placeholder
        fixation_map = torch.zeros(1, image.shape[1], image.shape[2])

        return image, fixation_map
```

### Approach 3: Hybrid Approach (Best of Both)

1. **Use pysaliency to download** datasets automatically
2. **Extract to standard directory structure** that our loaders expect
3. **Use our existing loaders** (already tested and working)

This combines pysaliency's convenience with our loaders' tested functionality.

---

## Recommended Workflow

### Step 1: Download Datasets

```bash
cd /mnt/lustre/work/bethge/bkr710/projects/research-os-deepgaze/research-os/artifacts/2025-10-21-entropy-reg-core-validation/scripts

/mnt/lustre/work/bethge/bkr710/.conda/deepgaze/bin/python download_datasets.py \
    --output_dir /mnt/lustre/work/bethge/bkr710/datasets/saliency
```

### Step 2: Find Dataset Location

After download, check where pysaliency stored the files:

```python
import pysaliency
stimuli, _ = pysaliency.external_datasets.get_mit1003(
    location='/mnt/lustre/work/bethge/bkr710/datasets/saliency'
)
# Inspect stimuli object to find actual file paths
```

### Step 3: Update Configuration Files

Update `configs/baseline_config.yaml` and `configs/entropy_reg_config.yaml`:

```yaml
data:
  dataset: "MIT1003"
  data_path: "/mnt/lustre/work/bethge/bkr710/datasets/saliency/MIT1003"  # Update this path
```

### Step 4: Verify with Tests

Run our data loader tests to ensure everything works:

```bash
cd tests
/mnt/lustre/work/bethge/bkr710/.conda/deepgaze/bin/python -m pytest test_mit1003_loader.py -v
```

---

## Dataset Directory Structure

After pysaliency downloads, the expected structure should be:

### MIT1003
```
datasets/saliency/MIT1003/
├── ALLSTIMULI/
│   ├── i1.jpeg
│   ├── i2.jpeg
│   └── ... (1003 images)
└── ALLFIXATIONMAPS/
    ├── i1_fixMap.jpg
    ├── i2_fixMap.jpg
    └── ... (1003 fixation maps)
```

### CAT2000
```
datasets/saliency/CAT2000/
├── Stimuli/
│   ├── Action/
│   │   └── Output_Action_*.jpg
│   ├── Affective/
│   └── ... (20 categories)
└── FIXATIONMAPS/
    ├── Action/
    │   └── fixMap_Action_*.jpg
    └── ... (20 categories)
```

---

## Troubleshooting

### Issue: Pysaliency downloads to different location
**Solution**: Use the `location` parameter to specify exactly where you want the data.

### Issue: Directory structure doesn't match expected format
**Solution**: After pysaliency download, you may need to reorganize files to match our loader's expected structure. Or use Approach 2 (wrapper loader).

### Issue: Fixation maps not generated correctly
**Solution**: Pysaliency provides fixation points (x, y coordinates). If you need actual fixation maps (heatmaps), you'll need to generate them from the fixation coordinates, or check if pysaliency provides a conversion function.

---

## Next Steps

1. **Download datasets** using the provided script
2. **Inspect the downloaded structure** to understand how pysaliency organizes files
3. **Update data loaders if needed** to match pysaliency's structure
4. **Run tests** to verify everything works
5. **Update config files** with correct paths
6. **Begin training!**

---

## References

- Pysaliency GitHub: https://github.com/matthias-k/pysaliency
- Pysaliency PyPI: https://pypi.org/project/pysaliency/
- MIT1003 Dataset: https://people.csail.mit.edu/tjudd/WherePeopleLook/
- CAT2000 Dataset: http://saliency.mit.edu/datasets.html
