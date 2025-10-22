"""
Download MIT1003 and CAT2000 datasets using pysaliency.

This script uses the pysaliency library to automatically download
and prepare the MIT1003 and CAT2000 datasets for training and evaluation.

Usage:
    python download_datasets.py --output_dir /path/to/datasets
"""

import argparse
import pysaliency
from pathlib import Path
import sys


def download_mit1003(output_dir: Path):
    """Download MIT1003 dataset using pysaliency.

    Args:
        output_dir: Directory to store downloaded datasets
    """
    print("=" * 60)
    print("Downloading MIT1003 Dataset")
    print("=" * 60)

    try:
        # Download MIT1003
        mit_stimuli, mit_fixations = pysaliency.external_datasets.get_mit1003(
            location=str(output_dir)
        )

        print(f"✓ MIT1003 downloaded successfully")
        print(f"  - Number of stimuli: {len(mit_stimuli)}")
        print(f"  - Number of fixation sets: {len(mit_fixations)}")
        print(f"  - Location: {output_dir}")

        # Print sample information
        print("\nSample stimulus information:")
        print(f"  - First stimulus shape: {mit_stimuli.stimuli[0].shape}")
        print(f"  - Stimulus type: {type(mit_stimuli.stimuli[0])}")

        print("\nSample fixation information:")
        print(f"  - Total fixations: {len(mit_fixations.x)}")
        print(f"  - Fixation fields: {mit_fixations.__dict__.keys()}")

        return mit_stimuli, mit_fixations

    except Exception as e:
        print(f"✗ Failed to download MIT1003: {e}")
        raise


def download_cat2000(output_dir: Path):
    """Download CAT2000 dataset using pysaliency.

    Args:
        output_dir: Directory to store downloaded datasets
    """
    print("\n" + "=" * 60)
    print("Downloading CAT2000 Dataset")
    print("=" * 60)

    try:
        # Try to get CAT2000 train set
        # Note: CAT2000 might have different functions for train/test
        cat_stimuli, cat_fixations = pysaliency.external_datasets.get_cat2000_train(
            location=str(output_dir)
        )

        print(f"✓ CAT2000 downloaded successfully")
        print(f"  - Number of stimuli: {len(cat_stimuli)}")
        print(f"  - Number of fixation sets: {len(cat_fixations)}")
        print(f"  - Location: {output_dir}")

        # Print sample information
        print("\nSample stimulus information:")
        print(f"  - First stimulus shape: {cat_stimuli.stimuli[0].shape}")
        print(f"  - Categories: {len(set(cat_stimuli.attributes.get('category', [])))}")

        return cat_stimuli, cat_fixations

    except AttributeError:
        # Try alternative method
        try:
            cat_stimuli, cat_fixations = pysaliency.external_datasets.get_cat2000(
                location=str(output_dir)
            )
            print(f"✓ CAT2000 downloaded successfully (using get_cat2000)")
            print(f"  - Number of stimuli: {len(cat_stimuli)}")
            return cat_stimuli, cat_fixations
        except Exception as e:
            print(f"✗ Failed to download CAT2000: {e}")
            print("  Note: CAT2000 might need manual download")
            return None, None
    except Exception as e:
        print(f"✗ Failed to download CAT2000: {e}")
        return None, None


def verify_dataset_structure(output_dir: Path):
    """Verify the downloaded dataset structure.

    Args:
        output_dir: Directory containing downloaded datasets
    """
    print("\n" + "=" * 60)
    print("Verifying Dataset Structure")
    print("=" * 60)

    # Check for MIT1003 directories
    mit_dirs = [
        output_dir / "MIT1003",
        output_dir / "mit1003",
    ]

    for mit_dir in mit_dirs:
        if mit_dir.exists():
            print(f"✓ Found MIT1003 directory: {mit_dir}")
            # List contents
            contents = list(mit_dir.iterdir())
            print(f"  Contents: {[d.name for d in contents[:10]]}")
            break
    else:
        print("⚠ MIT1003 directory structure may differ from expected")

    # Check for CAT2000 directories
    cat_dirs = [
        output_dir / "CAT2000",
        output_dir / "cat2000",
    ]

    for cat_dir in cat_dirs:
        if cat_dir.exists():
            print(f"✓ Found CAT2000 directory: {cat_dir}")
            # List contents
            contents = list(cat_dir.iterdir())
            print(f"  Contents: {[d.name for d in contents[:10]]}")
            break
    else:
        print("⚠ CAT2000 directory structure may differ from expected")


def create_dataset_info_file(output_dir: Path, mit_stimuli, cat_stimuli):
    """Create a dataset info file with paths and metadata.

    Args:
        output_dir: Directory containing datasets
        mit_stimuli: MIT1003 stimuli object from pysaliency
        cat_stimuli: CAT2000 stimuli object from pysaliency (optional)
    """
    info_file = output_dir / "dataset_info.txt"

    with open(info_file, 'w') as f:
        f.write("# Dataset Information\n")
        f.write("=" * 60 + "\n\n")

        f.write("## MIT1003\n")
        if mit_stimuli:
            f.write(f"Number of images: {len(mit_stimuli)}\n")
            f.write(f"Expected split: 902 train / 101 validation\n")
            f.write(f"Image dimensions: Varies (will be resized to 1024x768)\n")
        f.write(f"Dataset location: {output_dir}\n\n")

        f.write("## CAT2000\n")
        if cat_stimuli:
            f.write(f"Number of images: {len(cat_stimuli)}\n")
            f.write(f"Expected samples for OOD: 50 (randomly sampled)\n")
            f.write(f"Categories: 20\n")
        else:
            f.write("Dataset not downloaded or failed\n")
        f.write(f"Dataset location: {output_dir}\n\n")

        f.write("## Usage\n")
        f.write("Update configuration files with the dataset location:\n")
        f.write(f"  data_path: {output_dir}\n")

    print(f"\n✓ Dataset info saved to: {info_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Download MIT1003 and CAT2000 datasets using pysaliency"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data",
        help="Directory to store downloaded datasets (default: ../data)"
    )
    parser.add_argument(
        "--mit1003_only",
        action="store_true",
        help="Download only MIT1003 dataset"
    )
    parser.add_argument(
        "--cat2000_only",
        action="store_true",
        help="Download only CAT2000 dataset"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print(f"Datasets will be downloaded to: {output_dir}\n")

    mit_stimuli = None
    cat_stimuli = None

    # Download MIT1003
    if not args.cat2000_only:
        try:
            mit_stimuli, mit_fixations = download_mit1003(output_dir)
        except Exception as e:
            print(f"Error downloading MIT1003: {e}")
            sys.exit(1)

    # Download CAT2000
    if not args.mit1003_only:
        cat_stimuli, cat_fixations = download_cat2000(output_dir)

    # Verify structure
    verify_dataset_structure(output_dir)

    # Create info file
    create_dataset_info_file(output_dir, mit_stimuli, cat_stimuli)

    print("\n" + "=" * 60)
    print("Dataset Download Complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Update config files with dataset path: {output_dir}")
    print(f"2. Run data loader tests to verify everything works")
    print(f"3. Start training!")


if __name__ == "__main__":
    main()
