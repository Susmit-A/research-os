# Setup Status - Task 1

## Completed Sub-tasks

### ✓ 1.1 Create directory structure
- Created complete directory structure as specified in technical-spec.md:
  - `src/models/`, `src/data/`, `src/training/`, `src/evaluation/`
  - `configs/`
  - `scripts/`
  - `outputs/checkpoints/`, `outputs/logs/`, `outputs/results/`

### ✓ 1.2 Clone and adapt DeepGaze 3 implementation
- Cloned Matthias Kümmerer's DeepGaze repository from https://github.com/matthias-k/DeepGaze
- Copied all necessary DeepGaze files to `src/models/`:
  - `deepgaze3.py` - Main DeepGaze III model
  - `modules.py`, `layers.py` - Model components
  - `data.py`, `training.py` - Training utilities
  - `metrics.py` - Evaluation metrics
  - `features/` - Feature extraction modules
- **Created `entropy_regularizer.py`** with:
  - `UniformImageGenerator` - Generates uniform images for bias extraction
  - `ShannonEntropyComputer` - Computes Shannon entropy H = -Σ(p * log(p))
  - `BiasMapExtractor` - Extracts bias maps from model using uniform images
  - `EntropyRegularizer` - Complete regularization module

### ⏳ 1.3 Create conda environment (IN PROGRESS)
- Created `environment.yml` with all required dependencies:
  - PyTorch >= 2.0.0
  - torchvision >= 0.15.0
  - numpy >= 1.24.0
  - scipy >= 1.10.0
  - opencv >= 4.7.0
  - scikit-image >= 0.20.0
  - pyyaml >= 6.0
  - pytest, pytest-cov for testing
- **Status**: Conda environment creation is running in background (solving dependencies)
- **Next step**: Wait for completion, then activate and verify

### ✓ 1.4 Setup SLURM job scripts
- Created `scripts/run_baseline.sh`:
  - Requests 4x A100 GPUs
  - Sets up distributed training with PyTorch DDP
  - Runs baseline training for ~12 hours
- Created `scripts/run_entropy_reg.sh`:
  - Requests 4x A100 GPUs
  - Sets up distributed training with PyTorch DDP
  - Runs entropy-regularized training for ~16 hours
- Both scripts made executable

### ✓ 1.5 Create configuration YAML files
- Created `configs/baseline_config.yaml`:
  - 25 epochs, batch size 32 per GPU
  - Adam optimizer with lr=0.001585
  - MultiStepLR scheduler (milestones at epochs 12, 18)
  - NLL loss only (no regularization)
- Created `configs/entropy_reg_config.yaml`:
  - Same training hyperparameters as baseline
  - NLL + entropy regularization loss (lambda=1.0)
  - Entropy computed every 50 batches
  - 16 uniform samples for bias extraction

### ⏳ 1.6 Verify environment setup (PENDING)
- Created `scripts/smoke_test.py`:
  - Tests all package imports
  - Verifies CUDA availability
  - Tests model files accessibility
  - Tests entropy regularizer functionality
  - Validates configuration files
- **Status**: Waiting for conda environment to complete
- **Next step**: Run smoke test once environment is ready

## Next Steps

1. **Wait for conda environment creation to complete**
   ```bash
   # Check status
   conda env list | grep deepgaze-entropy-reg
   ```

2. **Activate environment and run smoke test**
   ```bash
   source activate deepgaze-entropy-reg
   cd scripts
   python smoke_test.py
   ```

3. **If smoke test passes, Task 1 is complete**
   - Mark all sub-tasks as [x] in tasks.md
   - Proceed to Task 2: Implement data loading and preprocessing

## Notes

- DeepGaze source code cloned to: `/mnt/lustre/work/bethge/bkr710/projects/research-os-deepgaze/deepgaze_source/`
- All paths in config files are relative to artifact directory
- **TODO**: Update `data_path` in both config files to point to actual MIT1003 dataset location
- Reproducibility seeds set to 42 in both configs

## Files Created

```
research-os/artifacts/2025-10-21-entropy-reg-core-validation/
├── environment.yml
├── configs/
│   ├── baseline_config.yaml
│   └── entropy_reg_config.yaml
├── scripts/
│   ├── run_baseline.sh
│   ├── run_entropy_reg.sh
│   └── smoke_test.py
└── src/models/
    ├── deepgaze3.py
    ├── modules.py
    ├── layers.py
    ├── data.py
    ├── training.py
    ├── metrics.py
    ├── entropy_regularizer.py
    └── features/
        └── [feature extraction modules]
```
