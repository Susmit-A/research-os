# Technical Specification

This is the technical specification for the artifact detailed in research-os/artifacts/2025-10-21-entropy-reg-core-validation/spec.md

## Technical Requirements

### Model Architecture

- **Base Model**: DeepGaze 3 architecture from Matthias Kümmerer's implementation
- **Backbone**: VGG-19 pre-trained on ImageNet (standard torchvision weights)
- **Input Resolution**: 1024×768 pixels (standard DeepGaze 3 resolution)
- **Output**: Dense saliency prediction maps (log-density predictions)
- **Mode**: Saliency-only mode (no fixation history modeling for this triage)

### Training Configuration

- **Dataset**: MIT1003 with 902-101 train-validation split
- **Batch Size**: 16-32 (optimize for 4x A100 GPU memory)
- **Epochs**: 25 epochs (fixed, no early stopping for fair comparison)
- **Learning Rate Schedule**: Multistep LR scheduler
  - Initial LR: 0.001585
  - Final LR: 1.5e-7
  - Milestones: [TBD based on DeepGaze 3 defaults - typically at 50%, 75% of training]
- **Optimizer**: Adam (DeepGaze 3 default)
- **Loss Function (Baseline)**: Negative log-likelihood on fixation data
- **Loss Function (Entropy-Regularized)**: NLL + lambda * (-Shannon_Entropy(bias_map))
  - Lambda: 1.0 (fixed for triage)

### Entropy Regularization Implementation

- **Uniform Image Generation**:
  - Generate images with constant pixel values across all channels
  - Test multiple uniform values (e.g., 0.0, 0.5, 1.0 normalized intensity)
  - Batch size: 8-16 uniform images per entropy computation

- **Shannon Entropy Computation**:
  - Formula: H = -Σ(p * log(p)) where p is normalized saliency map
  - Normalize bias maps to form valid probability distributions (softmax or min-max normalization + smoothing)
  - Compute spatial entropy across predicted saliency maps from uniform images

- **Regularization Frequency**:
  - Extract bias maps every N batches during training (e.g., every 10-50 batches)
  - Balance between regularization effectiveness and computational overhead

- **Loss Integration**:
  - Total Loss = NLL_Loss + lambda * (-Entropy)
  - Maximizing entropy = minimizing negative entropy
  - Lambda = 1.0 (no tuning for triage)

### Data Processing

- **Training Data**: 902 images from MIT1003
- **Validation Data (In-Domain)**: 101 images from MIT1003
- **OOD Evaluation Data**: 50 images randomly sampled from CAT2000
- **Preprocessing**:
  - Resize/crop to 1024×768 (DeepGaze 3 standard)
  - Normalize using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  - Load fixation maps and convert to ground-truth density maps

### Evaluation Metrics

- **Information Gain (IG)**: Primary metric for saliency performance
  - IG = KL(human_fixations || model_prediction) - KL(human_fixations || center_prior)
  - Use Gaussian center prior (standard baseline)
  - Compute per-image, report mean and standard deviation

- **Bias Entropy Measurement**:
  - Extract bias maps from 16 uniform images at end of training
  - Compute Shannon entropy on averaged bias map
  - Report percentage increase: (H_regularized - H_baseline) / H_baseline * 100%

- **Training Stability**:
  - Monitor loss convergence (ensure no NaN, no divergence)
  - Log training/validation loss every epoch
  - Save loss curves for visual inspection

### Computational Resources

- **GPUs**: 4x NVIDIA A100 (40GB or 80GB)
- **Distributed Training**: Use PyTorch DistributedDataParallel (DDP)
- **Mixed Precision**: Optional FP16 for faster training (test for numerical stability)
- **Estimated Training Time**:
  - Baseline: ~6-8 hours for 25 epochs
  - Entropy-Regularized: ~8-12 hours (additional entropy computation overhead)
  - Total: ~20 hours for both models in parallel

### Code Organization

- **Repository Structure**:
  ```
  research-os/artifacts/2025-10-21-entropy-reg-core-validation/
    ├── src/
    │   ├── models/
    │   │   ├── deepgaze3.py (adapted from Kümmerer)
    │   │   └── entropy_regularizer.py
    │   ├── data/
    │   │   ├── mit1003_loader.py
    │   │   └── cat2000_loader.py
    │   ├── training/
    │   │   ├── train_baseline.py
    │   │   └── train_entropy_reg.py
    │   └── evaluation/
    │       ├── compute_ig.py
    │       └── measure_bias_entropy.py
    ├── configs/
    │   ├── baseline_config.yaml
    │   └── entropy_reg_config.yaml
    ├── scripts/
    │   ├── run_baseline.sh (SLURM job script)
    │   └── run_entropy_reg.sh (SLURM job script)
    └── outputs/
        ├── checkpoints/
        ├── logs/
        └── results/
  ```

- **Configuration Management**: Use YAML config files for hyperparameters
- **Logging**: Print to stdout, save to text log files (no W&B for triage)
- **Checkpointing**: Save model weights + optimizer state every 5 epochs, keep final checkpoint

### Testing and Validation

- **Unit Tests**:
  - Test uniform image generation produces correct shapes/values
  - Test Shannon entropy computation on toy examples (verify against scipy.stats)
  - Test bias map extraction from models

- **Integration Tests**:
  - Run 1-2 epoch smoke test on small data subset before full training
  - Verify baseline model matches expected DeepGaze 3 performance from literature

- **Reproducibility**:
  - Set random seeds (torch, numpy, python)
  - Document CUDA version, PyTorch version, GPU model
  - Save git commit hash of code used for training

## External Dependencies

This experiment adapts existing code and uses standard dependencies from the tech stack. The following external resource is required:

- **Matthias Kümmerer's DeepGaze 3 Implementation**
  - Repository: https://github.com/matthias-k/DeepGaze (or similar official repo)
  - Purpose: Base implementation of DeepGaze 3 architecture and training pipeline
  - Justification: Building on established, validated implementation ensures correct architecture and enables focus on entropy regularization novelty rather than reimplementing standard components
  - License: Check license compatibility for academic research use
  - Adaptation Required: Modify training loop to inject entropy regularization, adapt data loaders for MIT1003/CAT2000 datasets

**Standard Python Dependencies** (from tech-stack.md):
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scipy>=1.10.0
opencv-python>=4.7.0
scikit-image>=0.20.0
pyyaml>=6.0
```

No additional external dependencies beyond those specified in the tech stack are required for this triage experiment.
