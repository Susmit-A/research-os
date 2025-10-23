# Task 6: Training Execution Guide

This guide provides step-by-step instructions for executing Task 6 (parallel training of baseline and entropy-regularized DeepGaze 3 models).

## Pre-Launch Checklist

### 1. Verify Environment Setup

Check that the conda environment is properly configured:

```bash
cd /mnt/lustre/work/bethge/bkr710/projects/research-os-deepgaze/research-os/artifacts/2025-10-21-entropy-reg-core-validation

# Activate environment
export PATH=/mnt/lustre/work/bethge/bkr710/.conda/deepgaze/bin:$PATH
export CONDA_PREFIX=/mnt/lustre/work/bethge/bkr710/.conda/deepgaze

# Verify PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"
```

Expected output:
- PyTorch version >= 2.0.0
- CUDA available: True
- GPUs: Should detect at least 4 when on compute node

### 2. Verify Dataset Availability

Check that MIT1003 dataset has been downloaded:

```bash
# Run the dataset download script if not already done
python scripts/download_datasets.py --output_dir /path/to/datasets
```

Then update the configuration files with the correct data path:

**Edit `configs/baseline_config.yaml` and `configs/entropy_reg_config.yaml`:**
- Change `data_path: "/path/to/MIT1003"` to your actual dataset location

### 3. Run Smoke Tests

Verify that the training pipelines work before submitting long jobs:

```bash
# Test baseline training (1-2 epochs on small subset)
python scripts/smoke_test_training.py

# Test entropy-regularized training (1-2 epochs)
python scripts/smoke_test_entropy_training.py
```

These should complete without errors in 10-20 minutes.

## Launching Training Jobs

### Task 6.1: Launch Baseline Training

Submit the baseline training job to SLURM:

```bash
cd /mnt/lustre/work/bethge/bkr710/projects/research-os-deepgaze/research-os/artifacts/2025-10-21-entropy-reg-core-validation/scripts

sbatch run_baseline.sh
```

This will:
- Request 1 node with 4x A100 GPUs
- Run for up to 12 hours
- Train DeepGaze 3 baseline model for 25 epochs
- Save checkpoints to `outputs/checkpoints/baseline/`
- Write logs to `outputs/logs/baseline_<JOB_ID>.out`

### Task 6.2: Launch Entropy-Regularized Training

Submit the entropy-regularized training job:

```bash
sbatch run_entropy_reg.sh
```

This will:
- Request 1 node with 4x A100 GPUs
- Run for up to 16 hours (longer due to entropy computation)
- Train DeepGaze 3 with entropy regularization for 25 epochs
- Save checkpoints to `outputs/checkpoints/entropy_reg/`
- Write logs to `outputs/logs/entropy_reg_<JOB_ID>.out`

**Note:** Both jobs can run in parallel if resources are available.

## Task 6.3: Monitoring Training Progress

### Check Job Status

```bash
# View all your SLURM jobs
squeue -u $USER

# View specific job details
scontrol show job <JOB_ID>
```

### Monitor Training Logs

```bash
# Tail the baseline training log
tail -f ../outputs/logs/baseline_<JOB_ID>.out

# Tail the entropy-regularized training log
tail -f ../outputs/logs/entropy_reg_<JOB_ID>.out
```

### What to Look For in Logs

**Startup (first few lines):**
```
Job ID: <JOB_ID>
Job Name: deepgaze3_baseline (or deepgaze3_entropy_reg)
Node: <NODE_NAME>
GPUs: 4
Start Time: <TIMESTAMP>
...
Python: /mnt/lustre/work/bethge/bkr710/.conda/deepgaze/bin/python
PyTorch version: <VERSION>
CUDA available: True
Number of GPUs: 4
```

**Training Progress:**
- Epoch numbers increasing from 0 to 24
- Training loss decreasing over epochs
- Validation loss being reported each epoch
- For entropy-regularized model: Entropy values being logged
- No NaN values in losses
- No CUDA out-of-memory errors

**Expected Training Times:**
- Baseline: ~6-8 hours (estimated 15-20 minutes per epoch)
- Entropy-regularized: ~8-12 hours (estimated 20-30 minutes per epoch due to entropy computation)

### Checkpoints

Checkpoints are saved every 5 epochs plus the final checkpoint:

```bash
# Check saved checkpoints during training
ls -lh ../outputs/checkpoints/baseline/
ls -lh ../outputs/checkpoints/entropy_reg/
```

Expected checkpoint files:
- `checkpoint_epoch_5.pth`
- `checkpoint_epoch_10.pth`
- `checkpoint_epoch_15.pth`
- `checkpoint_epoch_20.pth`
- `checkpoint_epoch_25.pth` (final)
- `best_model.pth` (best validation loss)

## Task 6.4: Verify Training Completion

After both jobs complete, verify the following:

### 1. Check Job Exit Status

```bash
# Check the end of the log files
tail -20 ../outputs/logs/baseline_<JOB_ID>.out
tail -20 ../outputs/logs/entropy_reg_<JOB_ID>.out
```

Look for:
```
Training complete!
Best validation loss: <VALUE>
End Time: <TIMESTAMP>
Job completed successfully
```

### 2. Verify All Checkpoints Exist

```bash
# Baseline checkpoints (should have 6 files: epochs 5,10,15,20,25 + best)
ls ../outputs/checkpoints/baseline/*.pth | wc -l
# Expected output: 6

# Entropy-regularized checkpoints (should have 6 files)
ls ../outputs/checkpoints/entropy_reg/*.pth | wc -l
# Expected output: 6
```

### 3. Check for Training Issues

```bash
# Search for error messages in logs
grep -i "error\|exception\|nan\|failed" ../outputs/logs/baseline_<JOB_ID>.out
grep -i "error\|exception\|nan\|failed" ../outputs/logs/entropy_reg_<JOB_ID>.out
```

No significant errors should appear (some INFO/DEBUG messages are OK).

### 4. Verify Training Metrics

Load and inspect the final checkpoints to verify training occurred:

```bash
python <<EOF
import torch

# Check baseline model
baseline_ckpt = torch.load('../outputs/checkpoints/baseline/checkpoint_epoch_25.pth', map_location='cpu')
print(f"Baseline - Final epoch: {baseline_ckpt['epoch']}")
print(f"Baseline - Final train loss: {baseline_ckpt['train_loss']:.4f}")
print(f"Baseline - Final val loss: {baseline_ckpt['val_loss']:.4f}")

# Check entropy-regularized model
entropy_ckpt = torch.load('../outputs/checkpoints/entropy_reg/checkpoint_epoch_25.pth', map_location='cpu')
print(f"Entropy-reg - Final epoch: {entropy_ckpt['epoch']}")
print(f"Entropy-reg - Final train loss: {entropy_ckpt['train_loss']:.4f}")
print(f"Entropy-reg - Final val loss: {entropy_ckpt['val_loss']:.4f}")
if 'bias_entropy' in entropy_ckpt:
    print(f"Entropy-reg - Final bias entropy: {entropy_ckpt['bias_entropy']:.4f}")
EOF
```

## Task 6.5: Save Final Model Checkpoints with Metadata

The final checkpoints should automatically include optimizer states. Verify this:

```bash
python <<EOF
import torch

checkpoint = torch.load('../outputs/checkpoints/baseline/checkpoint_epoch_25.pth', map_location='cpu')
print("Keys in baseline checkpoint:")
for key in checkpoint.keys():
    print(f"  - {key}")

# Expected keys: 'epoch', 'model_state_dict', 'optimizer_state_dict', 'train_loss', 'val_loss', 'config', etc.
EOF
```

## Troubleshooting

### Job Fails to Start

**Issue:** Job stays in PENDING state
```bash
squeue -u $USER
# Shows PD (pending) status
```

**Solution:** Check SLURM reasons
```bash
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
```

Common reasons:
- `Resources`: Not enough GPUs available (wait for resources)
- `Priority`: Other jobs have higher priority (wait)
- `ReqNodeNotAvail`: Requested resources don't exist (check partition and GPU specs)

### Training Crashes or Shows NaN

**Issue:** Loss becomes NaN or training crashes mid-way

**Possible Causes:**
1. Learning rate too high
2. Numerical instability in entropy computation
3. Data loading issues
4. Out of memory

**Debug Steps:**
```bash
# Check full error in logs
grep -A 10 -B 10 "NaN\|Error\|Traceback" ../outputs/logs/baseline_<JOB_ID>.out

# Try with smaller batch size (edit configs/baseline_config.yaml)
# Change batch_size from 32 to 16 or 8

# Resubmit job
sbatch scripts/run_baseline.sh
```

### Out of Memory Errors

**Issue:** CUDA out of memory error

**Solution:** Reduce batch size in config files
```yaml
# Edit configs/baseline_config.yaml and configs/entropy_reg_config.yaml
training:
  batch_size: 16  # Reduce from 32 to 16
```

For entropy-regularized training, also reduce:
```yaml
training:
  entropy_regularization:
    num_uniform_samples: 8  # Reduce from 16 to 8
```

### Job Takes Longer Than Expected

**Issue:** Training is running but very slow

**Check:**
1. Are all 4 GPUs being used?
```bash
# On the compute node (use srun to access running job):
srun --jobid <JOB_ID> --pty nvidia-smi
# Should show 4 GPUs with processes
```

2. Check if data loading is bottleneck:
```bash
# Look for data loading warnings in logs
grep "data" ../outputs/logs/baseline_<JOB_ID>.out
```

## Success Criteria for Task 6

Task 6 is complete when ALL of the following are true:

- ✅ Baseline training job completed all 25 epochs without errors
- ✅ Entropy-regularized training job completed all 25 epochs without errors
- ✅ Both models have 6 checkpoint files saved (epochs 5,10,15,20,25 + best)
- ✅ No NaN or divergence issues in training logs
- ✅ Training and validation losses show convergence (decreasing trend)
- ✅ Final checkpoints contain model weights AND optimizer states
- ✅ Entropy-regularized model logs show entropy values during training

Once all criteria are met, you can proceed to Task 7 (Implement evaluation metrics).

## Next Steps

After Task 6 completes successfully:

1. ✅ Mark Task 6 as complete in `tasks.md`
2. → Proceed to Task 7: Implement evaluation metrics and analysis
3. → Evaluate both models on MIT1003 validation and CAT2000 OOD sets
4. → Generate go/no-go report with performance comparison

---

**Quick Reference Commands:**

```bash
# Submit jobs
sbatch scripts/run_baseline.sh
sbatch scripts/run_entropy_reg.sh

# Check job status
squeue -u $USER

# Monitor logs
tail -f ../outputs/logs/baseline_<JOB_ID>.out
tail -f ../outputs/logs/entropy_reg_<JOB_ID>.out

# Check checkpoints
ls -lh ../outputs/checkpoints/baseline/
ls -lh ../outputs/checkpoints/entropy_reg/

# Cancel job if needed
scancel <JOB_ID>
```
