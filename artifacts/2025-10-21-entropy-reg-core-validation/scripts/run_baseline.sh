#!/bin/bash
#SBATCH --job-name=deepgaze3_baseline
#SBATCH --output=../outputs/logs/baseline_%j.out
#SBATCH --error=../outputs/logs/baseline_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --partition=a100-galvani

# Exit on any error
set -e

# Print job information
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Start Time: $(date)"
echo ""

export DEEPGAZE_CONDA_ENV="/mnt/lustre/work/bethge/bkr710/.conda/deepgaze"

# Navigate to artifact directory
# Get artifact directory relative to script location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ARTIFACT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ARTIFACT_DIR" || { echo "ERROR: Failed to cd to $ARTIFACT_DIR"; exit 1; }
echo "Working directory: $(pwd)"
echo ""

# Create necessary directories
echo "=========================================="
echo "Setting up directories"
echo "=========================================="
mkdir -p outputs/logs
mkdir -p outputs/checkpoints/baseline
mkdir -p outputs/results
echo "✓ Created output directories"
echo ""

# Activate conda environment
echo "=========================================="
echo "Activating conda environment"
echo "=========================================="

# Check for conda environment in multiple locations
# Priority: 1. Environment variable, 2. User's .conda directory, 3. Colleague's directory
if [ -n "$DEEPGAZE_CONDA_ENV" ]; then
    CONDA_ENV="$DEEPGAZE_CONDA_ENV"
elif [ -d "$HOME/.conda/deepgaze" ]; then
    CONDA_ENV="$HOME/.conda/deepgaze"
elif [ -d "/mnt/lustre/home/bethge/bkr623/.conda/deepgaze" ]; then
    CONDA_ENV="/mnt/lustre/home/bethge/bkr623/.conda/deepgaze"
elif [ -d "/mnt/lustre/work/bethge/bkr710/.conda/deepgaze" ]; then
    CONDA_ENV="/mnt/lustre/work/bethge/bkr710/.conda/deepgaze"
    echo "WARNING: Using colleague's conda environment"
else
    echo "ERROR: Conda environment 'deepgaze' not found"
    echo "       Set DEEPGAZE_CONDA_ENV environment variable or create environment at:"
    echo "       - $HOME/.conda/deepgaze"
    echo "       - /mnt/lustre/home/bethge/bkr623/.conda/deepgaze"
    exit 1
fi

if [ ! -d "$CONDA_ENV" ]; then
    echo "ERROR: Conda environment not found at $CONDA_ENV"
    exit 1
fi

export PATH="$CONDA_ENV/bin:$PATH"
export CONDA_PREFIX="$CONDA_ENV"
echo "✓ Conda environment activated: $CONDA_ENV"
echo ""

# Set random seeds for reproducibility
export PYTHONHASHSEED=0

# Verify Python and PyTorch installation
echo "=========================================="
echo "Verifying environment"
echo "=========================================="
echo "Python: $(which python)"

if ! python -c 'import torch' 2>/dev/null; then
    echo "ERROR: PyTorch not found in environment"
    exit 1
fi

TORCH_VERSION=$(python -c 'import torch; print(torch.__version__)')
CUDA_AVAILABLE=$(python -c 'import torch; print(torch.cuda.is_available())')
CUDA_VERSION=$(python -c 'import torch; print(torch.version.cuda)')
NUM_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')

echo "✓ PyTorch version: $TORCH_VERSION"
echo "✓ CUDA available: $CUDA_AVAILABLE"
echo "✓ CUDA version: $CUDA_VERSION"
echo "✓ Number of GPUs: $NUM_GPUS"

if [ "$CUDA_AVAILABLE" != "True" ]; then
    echo "ERROR: CUDA not available"
    exit 1
fi

if [ "$NUM_GPUS" -lt "4" ]; then
    echo "WARNING: Expected 4 GPUs, but found $NUM_GPUS"
fi
echo ""

# Verify config file exists
echo "=========================================="
echo "Verifying configuration"
echo "=========================================="
CONFIG_FILE="configs/baseline_config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi
echo "✓ Config file found: $CONFIG_FILE"

# Check if dataset path is configured (grep for placeholder)
if grep -q "/path/to/MIT1003" "$CONFIG_FILE"; then
    echo "WARNING: Dataset path contains placeholder '/path/to/MIT1003'"
    echo "         Update config file with actual dataset path before training"
fi
echo ""

# Launch distributed training
echo "=========================================="
echo "Launching distributed training"
echo "=========================================="
echo "Training type: Baseline (no entropy regularization)"
echo "Number of processes: 4"
echo "Config: configs/baseline_config.yaml"
echo "Output directory: outputs/checkpoints/baseline"
echo "Log directory: outputs/logs/baseline"
echo ""

# Run distributed training with PyTorch DDP
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="localhost" \
    --master_port=29500 \
    src/training/train_baseline.py \
    --config configs/baseline_config.yaml \
    --output_dir outputs/checkpoints/baseline \
    --log_dir outputs/logs/baseline

# Check if training completed successfully
TRAIN_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Training Completed"
echo "=========================================="
echo "End Time: $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully"

    # List saved checkpoints
    if [ -d "outputs/checkpoints/baseline" ]; then
        echo ""
        echo "Saved checkpoints:"
        ls -lh outputs/checkpoints/baseline/*.pth 2>/dev/null || echo "  No checkpoints found"
    fi
else
    echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

echo ""
echo "=========================================="
echo "Job Summary"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $SLURM_START_TIME"
echo "End Time: $(date)"
echo "Log file: outputs/logs/baseline_${SLURM_JOB_ID}.out"
echo "=========================================="
