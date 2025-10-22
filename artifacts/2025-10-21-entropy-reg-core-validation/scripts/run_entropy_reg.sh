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

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Start Time: $(date)"

# Activate conda environment
export PATH=/mnt/lustre/work/bethge/bkr710/.conda/deepgaze/bin:$PATH
export CONDA_PREFIX=/mnt/lustre/work/bethge/bkr710/.conda/deepgaze

# Set random seeds for reproducibility
export PYTHONHASHSEED=0

# Navigate to artifact directory
cd /mnt/lustre/work/bethge/bkr710/projects/research-os-deepgaze/research-os/artifacts/2025-10-21-entropy-reg-core-validation

# Print environment information
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"

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

echo "End Time: $(date)"
echo "Job completed successfully"
