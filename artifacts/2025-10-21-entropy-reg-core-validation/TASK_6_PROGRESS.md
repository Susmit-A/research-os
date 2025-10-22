# Task 6 Progress: Execute Parallel Training of Both Models

## Status: BLOCKED - Data Download Required

Task 6 execution has begun but is currently **blocked** by a critical prerequisite: the MIT1003 and CAT2000 datasets need to be downloaded and prepared in the expected format.

---

## Current Assessment

### Prerequisites Review

**✓ Code Implementation**: All training code from Tasks 1-5 is complete and tested
- Baseline training script: `src/training/train_baseline.py`
- Entropy-regularized training script: `src/training/train_entropy_reg.py`
- Data loaders: `src/data/mit1003_loader.py`, `src/data/cat2000_loader.py`
- SLURM scripts: `scripts/run_baseline.sh`, `scripts/run_entropy_reg.sh`
- Configurations: `configs/baseline_config.yaml`, `configs/entropy_reg_config.yaml`

**✓ Computational Resources**: 4x A100 GPUs available via SLURM
- SLURM scripts configured for 4 GPUs
- DDP (Distributed Data Parallel) implemented
- Estimated training time: 6-8 hours (baseline), 8-12 hours (entropy-reg)

**✗ BLOCKER - Dataset Availability**: Datasets not available in expected format
- **MIT1003**: Found at two locations but in HDF5 format, not individual images
  - `/mnt/lustre/work/bethge/bkr499/datasets/MIT1003/` (HDF5)
  - `/mnt/lustre/work/bethge/bkr139/datasets/MIT1003/` (HDF5)
- **CAT2000**: Not found in searched locations
- **Expected format**: Individual JPEG images in `ALLSTIMULI/` and fixation maps in `ALLFIXATIONMAPS/`
- **Data loader requirement**: Implemented for individual image files, not HDF5

---

## Blocker Details

### MIT1003 Dataset

**Current State**:
- Available datasets use HDF5 format: `fixations.hdf5`, `stimuli.hdf5`, `stimuli/` directory
- Our data loader (`src/data/mit1003_loader.py`) expects:
  ```
  MIT1003/
    ├── ALLSTIMULI/
    │   ├── i1.jpeg
    │   ├── i2.jpeg
    │   └── ...
    └── ALLFIXATIONMAPS/
        ├── i1_fixMap.jpeg
        ├── i2_fixMap.jpeg
        └── ...
  ```

**Required Action**: Download MIT1003 dataset in the expected format or convert HDF5 to individual images

### CAT2000 Dataset

**Current State**:
- Not found in searched locations
- Our data loader (`src/data/cat2000_loader.py`) expects:
  ```
  CAT2000/
    ├── Stimuli/
    │   ├── Action/
    │   ├── Affective/
    │   └── ... (20 categories)
    └── FIXATIONMAPS/
        ├── Action/
        ├── Affective/
        └── ... (20 categories)
  ```

**Required Action**: Download CAT2000 dataset

---

## Resolution Options

### Option 1: Download Datasets (Recommended)

**Advantages**:
- Uses tested data loaders from Task 2
- No code modifications required
- Standard format for reproducibility

**Steps**:
1. Download MIT1003 dataset from MIT Saliency Benchmark
   - URL: http://people.csail.mit.edu/tjudd/WherePeopleLook/index.html
   - Download ALLSTIMULI.zip and ALLFIXATIONMAPS.zip
   - Extract to `/mnt/lustre/work/bethge/bkr710/datasets/MIT1003/`

2. Download CAT2000 dataset
   - URL: http://saliency.mit.edu/datasets.html (CAT2000 section)
   - Download stimuli and fixation maps
   - Extract to `/mnt/lustre/work/bethge/bkr710/datasets/CAT2000/`

3. Update config files with correct paths
4. Verify data loaders with test script
5. Proceed with training

**Estimated Time**: 1-2 hours (depending on download speeds and dataset sizes)

### Option 2: Modify Data Loaders for HDF5 Format

**Advantages**:
- Uses existing datasets on cluster
- Faster to implement than downloading

**Disadvantages**:
- Requires modifying tested code from Task 2
- Need to re-run all data loader tests
- May introduce bugs
- Doesn't solve CAT2000 issue

**Not Recommended**: Modifying tested code introduces risk and doesn't fully resolve the blocker

### Option 3: Extract Images from HDF5

**Advantages**:
- Uses existing data
- Creates expected format for data loaders

**Steps**:
1. Write script to extract images from HDF5 files
2. Create ALLSTIMULI and ALLFIXATIONMAPS directories
3. Extract all images and fixation maps
4. Verify with data loader tests

**Estimated Time**: 30-60 minutes

---

## Recommended Action Plan

**Immediate Next Steps**:

1. **Choose Resolution Path** (Option 1 recommended)
2. **Download MIT1003 Dataset**:
   ```bash
   mkdir -p /mnt/lustre/work/bethge/bkr710/datasets/MIT1003
   cd /mnt/lustre/work/bethge/bkr710/datasets/MIT1003
   # Download ALLSTIMULI.zip and ALLFIXATIONMAPS.zip
   wget [MIT1003_STIMULI_URL]
   wget [MIT1003_FIXMAPS_URL]
   unzip ALLSTIMULI.zip
   unzip ALLFIXATIONMAPS.zip
   ```

3. **Download CAT2000 Dataset**:
   ```bash
   mkdir -p /mnt/lustre/work/bethge/bkr710/datasets/CAT2000
   cd /mnt/lustre/work/bethge/bkr710/datasets/CAT2000
   # Download stimuli and fixation maps
   wget [CAT2000_URL]
   # Extract as needed
   ```

4. **Update Configuration Files**:
   - Edit `configs/baseline_config.yaml`: Set `data_path` to `/mnt/lustre/work/bethge/bkr710/datasets/MIT1003`
   - Edit `configs/entropy_reg_config.yaml`: Set `data_path` to `/mnt/lustre/work/bethge/bkr710/datasets/MIT1003`

5. **Verify Data Loaders**:
   ```bash
   python -m pytest tests/test_mit1003_loader.py -v
   python -m pytest tests/test_cat2000_loader.py -v
   ```

6. **Proceed with Sub-task 6.1**: Launch baseline training job

---

## Task 6 Sub-tasks Status

- [ ] 6.1 Launch baseline training job on 4x A100 GPUs (25 epochs, ~6-8 hours)
  - **Status**: Blocked - awaiting dataset download
  - **Blocker**: MIT1003 dataset not available

- [ ] 6.2 Launch entropy-regularized training job on 4x A100 GPUs (25 epochs, ~8-12 hours)
  - **Status**: Blocked - awaiting dataset download
  - **Blocker**: MIT1003 dataset not available

- [ ] 6.3 Monitor training progress (check for NaN, divergence, convergence)
  - **Status**: Blocked - training not started

- [ ] 6.4 Verify both models complete 25 epochs successfully
  - **Status**: Blocked - training not started

- [ ] 6.5 Save final model checkpoints with optimizer states
  - **Status**: Blocked - training not started

---

## Additional Notes

### Conda Environment Status
- Background process for conda environment creation was running
- Process ID: eee1aa
- May need to verify completion before training

### SLURM Script Verification
- Scripts are ready: `scripts/run_baseline.sh`, `scripts/run_entropy_reg.sh`
- Configuration files exist and are structurally correct
- Only need data paths updated

### Next Session Recommendations
1. Start with dataset download (Option 1 is fastest path to unblock)
2. Verify datasets with existing unit tests
3. Launch training jobs via SLURM
4. Set up monitoring for long-running jobs

---

*Last Updated: 2025-10-22*
*Blocker Identified: Dataset availability*
*Recommended Resolution: Download MIT1003 and CAT2000 datasets*
