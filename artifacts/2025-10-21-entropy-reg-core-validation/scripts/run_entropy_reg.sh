#!/bin/bash
#SBATCH --job-name=deepgaze3_entropy_reg
#SBATCH --output=../outputs/logs/entropy_reg_%j.out
#SBATCH --error=../outputs/logs/entropy_reg_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=16:00:00
#SBATCH --partition=gpu

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

# Navigate to artifact directory
ARTIFACT_DIR="/mnt/lustre/work/bethge/bkr710/projects/research-os-deepgaze/research-os/artifacts/2025-10-21-entropy-reg-core-validation"
cd "$ARTIFACT_DIR" || { echo "ERROR: Failed to cd to $ARTIFACT_DIR"; exit 1; }
echo "Working directory: $(pwd)"
echo ""

# Create necessary directories
echo "=========================================="
echo "Setting up directories"
echo "=========================================="
mkdir -p outputs/logs
mkdir -p outputs/checkpoints/entropy_reg
mkdir -p outputs/results
echo "✓ Created output directories"
echo ""

# Activate conda environment
echo "=========================================="
echo "Activating conda environment"
echo "=========================================="
CONDA_ENV="/mnt/lustre/work/bethge/bkr710/.conda/deepgaze"

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
CONFIG_FILE="configs/entropy_reg_config.yaml"
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
echo "Training type: Entropy-regularized (with bias entropy maximization)"
echo "Number of processes: 4"
echo "Config: configs/entropy_reg_config.yaml"
echo "Output directory: outputs/checkpoints/entropy_reg"
echo "Log directory: outputs/logs/entropy_reg"
echo ""

# Run distributed training with PyTorch DDP and entropy regularization
# Using torch.distributed.launch for multi-GPU training
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="localhost" \
    --master_port=29501 \
    src/training/train_entropy_reg.py \
    --config configs/entropy_reg_config.yaml \
    --output_dir outputs/checkpoints/entropy_reg \
    --log_dir outputs/logs/entropy_reg

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
    if [ -d "outputs/checkpoints/entropy_reg" ]; then
        echo ""
        echo "Saved checkpoints:"
        ls -lh outputs/checkpoints/entropy_reg/*.pth 2>/dev/null || echo "  No checkpoints found"
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
echo "Log file: outputs/logs/entropy_reg_${SLURM_JOB_ID}.out"
echo "=========================================="
