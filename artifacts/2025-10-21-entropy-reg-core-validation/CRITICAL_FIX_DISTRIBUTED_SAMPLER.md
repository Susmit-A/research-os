# CRITICAL FIX: DistributedSampler for DDP Training

## Issue Summary

**Severity**: CRITICAL
**Impact**: Training performance and correctness
**Status**: ✅ FIXED

## Problem Description

The original data loader implementation did NOT use `DistributedSampler` for multi-GPU distributed training, causing:

### Critical Issues

1. **Data Duplication**: All 4 GPUs processed identical batches
   - Each GPU saw the same samples in the same order
   - No actual parallelization of data processing

2. **4x Slower Training**:
   - Baseline: Expected 6-8 hours → Would take 24-32 hours
   - Entropy-reg: Expected 8-12 hours → Would take 32-48 hours

3. **Wasted GPU Resources**:
   - 3 out of 4 GPUs doing redundant computation
   - Only benefiting from gradient averaging, not data parallelism

4. **Incorrect Gradient Computation**:
   - While DDP averages gradients, this doesn't compensate for processing the same data
   - Training dynamics would differ from expected behavior

### Root Cause

The `create_mit1003_dataloaders()` functions in both data loader files:
- Did NOT accept `world_size` and `rank` parameters
- Used `shuffle=True` instead of `DistributedSampler`
- Had no conditional logic for distributed vs single-GPU training

**Affected Files**:
- `src/data/mit1003_loader.py`
- `src/data/mit1003_loader_hdf5.py`

## Solution Implemented

### Changes Made

#### 1. Updated Function Signatures

Added distributed training parameters:

```python
def create_mit1003_dataloaders(
    data_path: str,
    batch_size: int = 32,
    num_workers: int = 8,
    image_size: Tuple[int, int] = (1024, 768),
    normalize: bool = True,
    seed: int = 42,
    world_size: int = 1,      # NEW
    rank: int = 0              # NEW
) -> Tuple[DataLoader, DataLoader]:
```

#### 2. Added DistributedSampler Logic

```python
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
        shuffle=False,  # Don't shuffle validation
        seed=seed,
        drop_last=False
    )
```

#### 3. Updated DataLoader Creation

```python
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=(train_sampler is None),  # Only shuffle if single-GPU
    sampler=train_sampler,             # Use DistributedSampler for DDP
    num_workers=num_workers,
    pin_memory=True
)
```

### How It Works Now

**Single-GPU Mode (world_size=1)**:
- `train_sampler = None`
- DataLoader uses `shuffle=True`
- Normal sequential data loading

**Multi-GPU Mode (world_size=4)**:
- Creates `DistributedSampler` with 4 replicas
- Each GPU (rank 0-3) gets different subset of data
- GPU 0: samples [0, 4, 8, 12, ...]
- GPU 1: samples [1, 5, 9, 13, ...]
- GPU 2: samples [2, 6, 10, 14, ...]
- GPU 3: samples [3, 7, 11, 15, ...]

## Verification

### Test Coverage

Added tests in `tests/test_distributed_loading.py`:

1. **test_create_dataloaders_with_distributed_sampler**
   - Verifies DistributedSampler is used when world_size > 1
   - ✅ PASSED

2. **test_create_dataloaders_without_distributed_sampler**
   - Verifies normal shuffle when world_size = 1
   - ✅ PASSED

### Manual Verification

Run existing distributed tests:

```bash
pytest tests/test_distributed_loading.py -v
```

All tests pass ✅

## Expected Performance Impact

### Before Fix
- **Effective GPUs**: 1 (other 3 redundant)
- **Baseline Training**: ~24-32 hours (4x slower)
- **Entropy-reg Training**: ~32-48 hours (4x slower)
- **Total GPU-hours wasted**: ~120 hours

### After Fix
- **Effective GPUs**: 4 (all utilized)
- **Baseline Training**: ~6-8 hours (as expected)
- **Entropy-reg Training**: ~8-12 hours (as expected)
- **GPU-hours saved**: ~120 hours per experiment ✅

## What Changed for Users

### NO CHANGES REQUIRED

The Trainer class already passes `world_size` and `rank` to the data loaders (trainer.py:125-126), so:

- ✅ Training scripts work without modification
- ✅ SLURM job scripts work without modification
- ✅ Configuration files work without modification
- ✅ No changes to user-facing API

### Backward Compatibility

The fix is fully backward compatible:

```python
# Old code (single-GPU) still works
train_loader, val_loader = create_mit1003_dataloaders(
    data_path="/path/to/data",
    batch_size=32
)

# New code (DDP) now works correctly
train_loader, val_loader = create_mit1003_dataloaders(
    data_path="/path/to/data",
    batch_size=32,
    world_size=4,
    rank=0
)
```

## Key Takeaways

### For Distributed Training

Always remember:
1. **Use DistributedSampler** for train and val datasets in DDP
2. **Set shuffle=False** when using DistributedSampler (sampler handles shuffling)
3. **Call `sampler.set_epoch(epoch)`** at start of each epoch for proper shuffling
4. **Test with world_size > 1** to catch these issues early

### Common Pitfalls Avoided

❌ **Wrong**: `DataLoader(dataset, shuffle=True)` in DDP
✅ **Right**: `DataLoader(dataset, sampler=DistributedSampler(...))`

❌ **Wrong**: Same sampler for all ranks
✅ **Right**: Different rank for each GPU's sampler

❌ **Wrong**: Forgetting to set epoch on sampler
✅ **Right**: `sampler.set_epoch(epoch)` before each epoch

## References

- PyTorch DistributedSampler Docs: https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
- PyTorch DDP Tutorial: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

## Timeline

- **Issue Identified**: 2025-10-23 (before training started)
- **Fix Implemented**: 2025-10-23 (same day)
- **Tests Verified**: 2025-10-23 (all passing)
- **Status**: ✅ Ready for training

---

**This fix was critical** - without it, the entire 4-GPU training setup would have been ineffective, wasting significant compute resources and time. The issue was caught before any long training runs were initiated, saving approximately 120 GPU-hours of wasted computation.
