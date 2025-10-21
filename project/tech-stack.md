# Technical Stack Documentation

## Project Overview

This document specifies the complete technical infrastructure for **Entropy-Regularized Saliency Prediction with Adaptive Bias Modeling (ERSP-ABM)**. The framework combines entropy regularization during training, explicit deconvolutional bias modeling, and few-shot bias adaptation to improve cross-dataset generalization in saliency prediction models.

**Target**: 6.3% average cross-dataset Information Gain improvement with minimal in-domain performance degradation.

---

## 1. Core Frameworks & Libraries

### 1.1 Deep Learning Framework

**Primary Framework**: PyTorch
- **Version**: 2.0.0 or higher (for improved performance and stability)
- **Rationale**: Industry standard for research, excellent VGG-19 support, flexible for custom architectures
- **Required Features**:
  - Automatic differentiation for joint optimization
  - Custom loss functions for entropy maximization
  - Gradient freezing for few-shot adaptation
  - Mixed precision training (FP16) support

### 1.2 Core Dependencies

#### Essential Libraries

```python
# Deep Learning & Computation
torch>=2.0.0                 # Core deep learning framework
torchvision>=0.15.0          # Pre-trained VGG-19 and image transforms
numpy>=1.24.0                # Numerical operations
scipy>=1.10.0                # Statistical functions and entropy calculations

# Computer Vision & Image Processing
Pillow>=9.5.0                # Image loading and preprocessing
opencv-python>=4.7.0         # Advanced image operations
scikit-image>=0.20.0         # Image processing utilities

# Data Handling
h5py>=3.8.0                  # HDF5 dataset storage
pandas>=2.0.0                # Dataset metadata and analysis
tqdm>=4.65.0                 # Progress bars for training loops

# Visualization
matplotlib>=3.7.0            # Plotting and visualization
seaborn>=0.12.0              # Statistical visualizations
```

#### Optional but Recommended

```python
# Experiment Tracking
wandb>=0.15.0                # Weights & Biases for experiment logging
tensorboard>=2.13.0          # TensorBoard for training curves

# Performance & Optimization
einops>=0.6.1                # Tensor operations for deconv network
timm>=0.9.0                  # Additional pre-trained models if needed

# Development Tools
pytest>=7.3.0                # Unit testing
black>=23.3.0                # Code formatting
ruff>=0.0.270                # Fast Python linter
jupyter>=1.0.0               # Notebook analysis
```

### 1.3 Model Architectures

#### DeepGaze 2E (DeepGaze IIE)

**Base Architecture**: VGG-19 (pre-trained on ImageNet)
- **Paper**: Kümmerer et al. (2016) "DeepGaze II: Reading fixations from deep features"
- **Enhancement**: DeepGaze IIE (2021) with improved calibration
- **Key Components**:
  - VGG-19 backbone: Frozen feature extractor
  - Readout layers: Trainable dense layers on top of VGG features
  - Centerbias integration: Multiplicative combination with external prior
  - Log-density output: Final layer produces log-density map

**Expected Performance**:
- In-domain (MIT1003): 92-93% Information Gain
- Baseline OOD: 81-82% average across 5 datasets

**Implementation Notes**:
- No fine-tuning of VGG-19 backbone (transfer learning approach)
- Use features from multiple VGG layers (multi-scale integration)
- Spatial dimensions preserved through readout layers

#### DeepGaze 3

**Base Architecture**: Enhanced version of DeepGaze II
- **Paper**: Scanpath model with fixation history
- **Our Usage**: Saliency predictions only (ignore scanpath capabilities)
- **Key Differences from DeepGaze 2E**:
  - More powerful readout architecture
  - Better feature integration across scales
  - Higher baseline performance (83.7% vs 81.3% OOD)

**Expected Performance**:
- Baseline OOD: 83.7% average
- With ERSP-ABM: 89.4% (+5.7% improvement)

#### VGG-19 Backbone Details

**Architecture**: 19-layer convolutional network
- **Pre-training**: ImageNet classification (1000 classes)
- **Feature Layers Used**:
  - conv1_1, conv1_2 (64 filters)
  - conv2_1, conv2_2 (128 filters)
  - conv3_1, conv3_2, conv3_3, conv3_4 (256 filters)
  - conv4_1, conv4_2, conv4_3, conv4_4 (512 filters)
  - conv5_1, conv5_2, conv5_3, conv5_4 (512 filters)
- **Weights**: Load from torchvision.models.vgg19(pretrained=True)

#### Deconvolutional Bias Model (Novel Component)

**Architecture Design**:
```
Input: Learned tensor [C=64, H_latent=8, W_latent=8]
  ↓
DeconvBlock1: 64 → 128 channels, 2x upsampling
  ↓
DeconvBlock2: 128 → 64 channels, 2x upsampling
  ↓
DeconvBlock3: 64 → 32 channels, 2x upsampling
  ↓
DeconvBlock4: 32 → 1 channel, output size matching image resolution
  ↓
Softmax: Convert to valid probability distribution
  ↓
Output: Spatial bias map [1, H_image, W_image]
```

**DeconvBlock Components**:
- ConvTranspose2d (deconvolution layer)
- BatchNorm2d (normalization)
- ReLU activation
- Optional: Skip connections for spatial preservation

**Training Strategy**:
- Joint optimization with saliency model
- Input tensor initialized randomly (Gaussian noise, σ=0.01)
- Freezing: All deconv parameters frozen during adaptation, only input tensor optimized

**Parameter Count**:
- Deconv network: ~2-5M parameters (depending on architecture depth)
- Input tensor: 4,096 parameters (64×8×8)
- Adaptation: Only 4,096 parameters updated (0.001% of total)

---

## 2. Datasets

### 2.1 Dataset Overview

The research uses **five standard saliency prediction datasets** in a leave-one-out cross-validation protocol. Each dataset provides eye fixation data from human observers viewing images.

| Dataset | Images | Purpose | Task Type | Bias Characteristics |
|---------|--------|---------|-----------|---------------------|
| MIT1003 | 1,003 | Training/OOD eval | Free-viewing | Strong center bias (~5.9 nats entropy) |
| CAT2000 | 2,000 | Training/OOD eval | Category-specific | Diverse scenes, moderate bias |
| COCO Freeview | ~5,000 | Training/OOD eval | Free-viewing on COCO | Natural images, strong center bias |
| Daemons | ~1,000 | Training/OOD eval | Specialized task | Unique spatial patterns |
| Figrim | ~800 | Training/OOD eval | Figure-ground | Different viewing patterns |

### 2.2 MIT1003

**Name**: MIT1003 (MIT Saliency Benchmark)

**Source**:
- MIT Computational Vision and Learning Lab
- Part of MIT300 benchmark ecosystem
- Download: http://saliency.mit.edu/datasets.html

**Statistics**:
- **Total Images**: 1,003 natural images
- **Resolution**: Variable (typically 1024×768 or similar)
- **Observers**: 15 observers per image
- **Fixations**: ~15 fixations per observer (3 seconds viewing)
- **Categories**: Indoor, outdoor, urban, natural scenes

**Purpose**:
- Primary training dataset for DeepGaze II/IIE
- Source of external centerbias prior used in training
- One of 5 datasets in leave-one-out evaluation

**Centerbias Characteristics**:
- Strong center bias (observers tend to look at image center)
- Extracted bias entropy: ~5.8-6.2 nats
- Gaussian-like spatial distribution
- Used as external prior during training

**Preprocessing Requirements**:
- Resize to consistent resolution (e.g., 512×384)
- Convert fixations to continuous fixation density maps (Gaussian blurring)
- Normalize density maps to sum to 1 (valid probability distribution)
- Extract empirical centerbias by averaging all fixation maps

**Data Splits**:
- Training: 80% (~800 images)
- Validation: 20% (~200 images)
- For leave-one-out: use all images when MIT1003 is in training set

**File Format**:
- Images: JPEG or PNG
- Fixations: MAT files or CSV (x, y coordinates)
- Fixation maps: PNG or NPY (continuous density)

**Access Instructions**:
```bash
# Download from MIT Saliency Benchmark
wget http://saliency.mit.edu/datasets/MIT1003.zip
unzip MIT1003.zip

# Expected structure:
MIT1003/
├── ALLSTIMULI/          # Images
├── ALLFIXATIONMAPS/     # Pre-computed fixation maps
└── DATA/                # Raw fixation coordinates
```

### 2.3 CAT2000

**Name**: CAT2000 (Category 2000)

**Source**:
- Developed for category-specific saliency analysis
- Download: https://github.com/cvzoya/saliency/tree/master/code_forMetrics

**Statistics**:
- **Total Images**: 2,000 images
- **Categories**: 20 categories × 100 images each
  - Action, Affective, Art, Black&White, Cartoon, Fractal, Indoor, Inverted, Jumbled, Line Drawing, Low Resolution, Noisy, Object, Outdoor, Pattern, Random, Satellite, Sketch, Social, Texture
- **Observers**: 24 observers per image
- **Resolution**: Variable

**Purpose**:
- Diverse scene categories for cross-dataset evaluation
- Tests generalization across different content types
- Challenges spatial bias assumptions (some categories have non-center biases)

**Centerbias Characteristics**:
- Varies by category (outdoor: center, line drawing: distributed)
- Overall entropy: moderate (6.0-6.5 nats)
- More diverse spatial patterns than MIT1003

**Preprocessing Requirements**:
- Same as MIT1003
- Consider per-category analysis if needed
- Balance training samples across categories

**Expected Performance**:
- Baseline OOD IG: 82.1%
- ERSP-ABM target: 87.4% (+5.3%)

### 2.4 COCO Freeview

**Name**: COCO Freeview (Free-viewing on COCO Images)

**Source**:
- Built on COCO dataset images
- Eye tracking data collected in free-viewing paradigm
- Download: (Specify actual source when available)

**Statistics**:
- **Total Images**: ~5,000 images (subset of COCO)
- **Content**: Natural images with objects, scenes, people
- **Observers**: Variable (typically 15-30 per image)
- **Resolution**: COCO standard (variable, commonly 640×480)

**Purpose**:
- Tests generalization to natural images from COCO distribution
- Free-viewing task without specific instruction
- Large dataset for robust training

**Centerbias Characteristics**:
- Strong center bias (free-viewing paradigm)
- Similar to MIT1003 in spatial distribution
- Entropy: ~6.0-6.3 nats

**Preprocessing Requirements**:
- Resize to consistent resolution
- Convert COCO annotations if needed
- Process fixation data to density maps

**Expected Performance**:
- Baseline OOD IG: 85.6%
- ERSP-ABM target: 90.1% (+4.5%)

### 2.5 Daemons

**Name**: Daemons Dataset

**Source**: (Likely MIT300 testset or similar benchmark)

**Statistics**:
- **Total Images**: ~1,000 images
- **Task**: Specialized viewing task
- **Observers**: Variable
- **Resolution**: Variable

**Purpose**:
- One of challenging OOD datasets
- Tests adaptation to unique spatial patterns
- Baseline performance lowest among 5 datasets

**Centerbias Characteristics**:
- Unique spatial distribution (task-specific)
- Lower entropy than free-viewing datasets
- Challenges standard center bias assumptions

**Preprocessing Requirements**:
- Same standardization as other datasets
- Careful analysis of task-specific viewing patterns

**Expected Performance**:
- Baseline OOD IG: 78.1% (worst baseline)
- ERSP-ABM target: 86.0% (+7.9% largest gain)

### 2.6 Figrim

**Name**: Figrim (Figure-Ground Dataset)

**Source**: (Specify actual source)

**Statistics**:
- **Total Images**: ~800 images
- **Task**: Figure-ground segmentation or related task
- **Content**: Images with clear figure-ground structure

**Purpose**:
- Tests generalization to task-specific viewing
- Figure-ground distinction affects viewing patterns
- Another challenging OOD scenario

**Centerbias Characteristics**:
- Task-driven spatial patterns
- Lower entropy due to focused viewing on figures
- Different from free-viewing distributions

**Expected Performance**:
- Baseline OOD IG: 76.4% (second worst)
- ERSP-ABM target: 84.7% (+8.3% second largest gain)

### 2.7 Dataset Preparation Pipeline

**Step 1: Download**
```bash
# Create dataset directory
mkdir -p data/saliency_datasets
cd data/saliency_datasets

# Download each dataset (adapt URLs as needed)
wget <MIT1003_URL> && unzip MIT1003.zip
wget <CAT2000_URL> && unzip CAT2000.zip
# ... repeat for other datasets
```

**Step 2: Preprocessing**
```python
# For each dataset:
1. Load images and resize to standard resolution (e.g., 512×384)
2. Load fixation coordinates or maps
3. Convert fixations to continuous density maps:
   - Create binary fixation map
   - Apply Gaussian blur (σ=3-5 pixels)
   - Normalize to sum to 1
4. Extract empirical centerbias:
   - Average all fixation maps in dataset
   - Normalize to probability distribution
5. Compute centerbias entropy for analysis
```

**Step 3: Train/Val Splits**
```python
# For each dataset, create 80/20 split
import numpy as np
from sklearn.model_selection import train_test_split

images, fixations = load_dataset(dataset_name)
train_idx, val_idx = train_test_split(
    range(len(images)),
    test_size=0.2,
    random_state=42
)

# Save splits for reproducibility
save_splits(dataset_name, train_idx, val_idx)
```

**Step 4: Data Loaders**
```python
# PyTorch DataLoader for training
class SaliencyDataset(torch.utils.data.Dataset):
    def __init__(self, images, fixation_maps, centerbias, transform=None):
        self.images = images
        self.fixation_maps = fixation_maps
        self.centerbias = centerbias
        self.transform = transform

    def __getitem__(self, idx):
        image = self.images[idx]
        fixation = self.fixation_maps[idx]

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'fixation_map': fixation,
            'centerbias': self.centerbias  # Same for all in dataset
        }
```

**Step 5: Validation**
```python
# Verify preprocessing
1. Check all images load correctly
2. Verify fixation maps are valid probability distributions (sum to 1)
3. Compute and compare centerbias entropy across datasets
4. Visualize samples from each dataset
5. Compute KL divergence matrix between dataset centerbiases
```

---

## 3. Pre-trained Models

### 3.1 VGG-19 (ImageNet Pre-trained)

**Source**: torchvision.models

**Download**:
```python
import torchvision.models as models

# Automatically downloads pre-trained weights
vgg19 = models.vgg19(pretrained=True)
```

**Weights Location**:
- Downloaded to: `~/.cache/torch/hub/checkpoints/`
- File: `vgg19-dcbb9e9d.pth`
- Size: ~548 MB

**Usage**:
- Feature extraction only (no fine-tuning)
- Remove classification head
- Extract multi-scale features from conv layers

**Verification**:
```python
# Verify pre-trained weights load correctly
vgg19.eval()
with torch.no_grad():
    test_input = torch.randn(1, 3, 224, 224)
    features = vgg19.features(test_input)
    print(f"VGG-19 features shape: {features.shape}")
```

### 3.2 DeepGaze 2E (DeepGaze IIE)

**Source**:
- Original DeepGaze II: https://github.com/matthias-k/DeepGaze
- Matthias Kümmerer's repository

**Checkpoints Needed**:
- Pre-trained DeepGaze IIE weights (if available)
- MIT1003 centerbias map (included in official release)

**Training from Scratch**:
- For this research, we train from scratch (per research plan)
- Pre-trained checkpoints used only for baseline comparison/validation
- Expected to reproduce 92-93% IG on MIT1003

**Model Components**:
```python
class DeepGaze2E(nn.Module):
    def __init__(self):
        super().__init__()
        # Load frozen VGG-19
        self.vgg19 = models.vgg19(pretrained=True).features
        for param in self.vgg19.parameters():
            param.requires_grad = False

        # Readout layers (trainable)
        self.readout = nn.Sequential(
            nn.Conv2d(512, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )

        # Centerbias (external, not learned)
        self.register_buffer('centerbias', torch.zeros(1, 1, H, W))
```

### 3.3 DeepGaze 3

**Source**:
- Research paper on scanpath prediction
- Implementation: (Official repo if available)

**Architecture Differences**:
- More sophisticated readout architecture
- Better multi-scale feature integration
- Originally designed for scanpath prediction (we use saliency only)

**Training from Scratch**:
- Same as DeepGaze 2E: train from scratch for experiments
- Higher baseline performance expected (83.7% OOD vs 81.3%)

**Implementation Notes**:
- Adapt scanpath architecture to produce single saliency map
- Ignore temporal/sequential components
- Focus on spatial prediction only

### 3.4 UNISAL (Deferred)

**Note**: UNISAL integration is **deferred to later work** (not part of main experiments)

**Source**:
- Paper: "Unified Image and Video Saliency Modeling" (2020)
- Domain-Adaptive Priors approach

**Potential Use**:
- Future comparison: UNISAL's domain-adaptive priors vs our implicit bias adaptation
- Image saliency only (not video)

---

## 4. Evaluation Metrics

### 4.1 Information Gain (IG) - Primary Metric

**Definition**:
Information Gain measures how much a saliency model's predictions improve over a baseline (typically uniform distribution or centerbias) in predicting human fixations.

**Formula**:
```
IG = KL(P_fixations || P_baseline) - KL(P_fixations || P_model)

where:
- P_fixations: Empirical fixation distribution (ground truth)
- P_baseline: Baseline prior (uniform or centerbias)
- P_model: Model's predicted saliency distribution
- KL: Kullback-Leibler divergence
```

**Interpretation**:
- IG = 0: Model performs no better than baseline
- IG = 1: Model perfectly predicts fixations (captures all explainable information)
- IG = 0.87 (87%): Model captures 87% of explainable information
- Higher IG = Better saliency prediction

**Percentage Form**:
- IG is typically reported as percentage: IG × 100%
- Example: 87% IG, 93% IG

**Baseline Choice**:
- Standard: Uniform distribution (all pixels equally likely)
- Alternative: Centerbias (dataset's average fixation pattern)
- Our experiments: Use uniform as baseline for consistency

**Implementation**:
```python
def information_gain(fixation_map, prediction, baseline='uniform'):
    """
    Compute Information Gain metric.

    Args:
        fixation_map: Ground truth fixation density [H, W]
        prediction: Model prediction [H, W]
        baseline: 'uniform' or centerbias array [H, W]

    Returns:
        ig: Information gain (0-1, higher is better)
    """
    # Normalize to probability distributions
    fixation_map = fixation_map / fixation_map.sum()
    prediction = prediction / prediction.sum()

    if baseline == 'uniform':
        baseline = np.ones_like(fixation_map) / fixation_map.size
    else:
        baseline = baseline / baseline.sum()

    # Compute KL divergences
    kl_baseline = kl_divergence(fixation_map, baseline)
    kl_model = kl_divergence(fixation_map, prediction)

    # Information Gain
    ig = (kl_baseline - kl_model) / kl_baseline
    return ig

def kl_divergence(p, q, epsilon=1e-10):
    """KL(P || Q) = sum(P * log(P / Q))"""
    p = p + epsilon  # Avoid log(0)
    q = q + epsilon
    return np.sum(p * np.log(p / q))
```

**Usage in Experiments**:
- Primary success metric for OOD performance
- Target: ≥5% absolute improvement over baseline
- Reported for each fold in leave-one-out evaluation

**Aggregation**:
- Compute IG per image
- Average across all images in validation set
- Report mean and standard deviation across random seeds

### 4.2 Entropy (Bias Analysis)

**Definition**:
Shannon entropy measures the uniformity of a probability distribution. For spatial bias maps, higher entropy indicates more uniform (less biased) distributions.

**Formula**:
```
H(P) = -∑_i P(i) * log(P(i))

where:
- P(i): Probability at spatial location i
- Higher H = More uniform distribution
- Lower H = More concentrated (biased) distribution
```

**Theoretical Maximum**:
```
H_max = log(N)

where N = number of spatial locations
Example: For 512×384 image (196,608 pixels)
H_max = log(196,608) ≈ 12.19 nats (natural log)
       = 7.12 nats if using normalized units
```

**Usage in Research**:
- **Primary Analysis**: Measure entropy of extracted implicit bias maps
- **Hypothesis**: Higher entropy (closer to uniform) = less implicit bias = better OOD generalization
- **Target**: Increase bias entropy from 5.92 nats (baseline) to 6.84 nats (regularized), ~15.5% increase

**Implementation**:
```python
def entropy(probability_map):
    """
    Compute Shannon entropy of probability distribution.

    Args:
        probability_map: 2D array [H, W], sums to 1

    Returns:
        H: Entropy in nats (natural logarithm)
    """
    # Flatten to 1D
    p = probability_map.flatten()

    # Remove zeros (log(0) undefined)
    p = p[p > 0]

    # Shannon entropy
    H = -np.sum(p * np.log(p))
    return H

def normalized_entropy(probability_map):
    """Entropy normalized by theoretical maximum."""
    H = entropy(probability_map)
    N = probability_map.size
    H_max = np.log(N)
    return H / H_max  # 0 to 1
```

**Entropy Calculation for Bias Maps**:
```python
# Extract implicit bias using uniform images
def extract_implicit_bias(model, device='cuda'):
    """Extract implicit bias by averaging predictions on uniform images."""
    model.eval()

    # 8 uniform color intensities
    intensities = [0, 36, 73, 109, 146, 182, 219, 255]

    bias_maps = []
    for intensity in intensities:
        # Create uniform image
        uniform_img = torch.full((1, 3, H, W), intensity/255.0).to(device)

        # Get prediction
        with torch.no_grad():
            pred = model(uniform_img)

        bias_maps.append(pred.cpu().numpy())

    # Average over all uniform colors
    implicit_bias = np.mean(bias_maps, axis=0)

    # Normalize to probability distribution
    implicit_bias = implicit_bias / implicit_bias.sum()

    return implicit_bias

# Compute entropy
bias_map = extract_implicit_bias(model)
bias_entropy = entropy(bias_map)
print(f"Implicit bias entropy: {bias_entropy:.3f} nats")
```

**Expected Results**:
- **Baseline models**: 5.92 ± 0.34 nats
- **ERSP-ABM models**: 6.84 ± 0.19 nats
- **Improvement**: +15.5% increase in entropy
- **Correlation with OOD IG**: Pearson r = 0.83 (p < 0.001)

### 4.3 KL Divergence (Bias Comparison)

**Definition**:
Kullback-Leibler divergence measures the difference between two probability distributions. Used to quantify how different extracted biases are from uniform distribution.

**Formula**:
```
KL(P || Q) = ∑_i P(i) * log(P(i) / Q(i))

where:
- P: Extracted bias distribution
- Q: Reference distribution (uniform)
- KL = 0: P and Q are identical
- Higher KL: Greater difference from reference
```

**Usage**:
- Measure how far extracted bias deviates from uniform
- Lower KL from uniform = less implicit bias
- Complement to entropy analysis

**Implementation**:
```python
def kl_from_uniform(bias_map):
    """
    Compute KL divergence of bias map from uniform distribution.

    Args:
        bias_map: Extracted implicit bias [H, W]

    Returns:
        kl: KL divergence in nats
    """
    # Normalize bias map
    p = bias_map.flatten()
    p = p / p.sum()

    # Uniform distribution
    q = np.ones_like(p) / len(p)

    # KL divergence with epsilon for numerical stability
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon

    kl = np.sum(p * np.log(p / q))
    return kl
```

**Expected Results**:
- Baseline: KL ≈ 0.8-1.2 nats from uniform
- Regularized: KL ≈ 0.3-0.5 nats from uniform (closer to uniform)

### 4.4 Statistical Significance Tests

**Purpose**: Validate that observed improvements are statistically significant, not due to random variation.

#### Paired t-test

**Usage**: Compare baseline vs ERSP-ABM performance across images

```python
from scipy.stats import ttest_rel

# Compute IG for each image
baseline_igs = [compute_ig(img, baseline_pred) for img in test_set]
ersp_abm_igs = [compute_ig(img, ersp_pred) for img in test_set]

# Paired t-test (same images compared)
t_statistic, p_value = ttest_rel(ersp_abm_igs, baseline_igs)

print(f"t-statistic: {t_statistic:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Significant at p<0.01: {p_value < 0.01}")
```

**Success Criterion**: p < 0.01 for OOD improvements

#### Correlation Analysis

**Usage**: Validate entropy-IG correlation hypothesis

```python
from scipy.stats import pearsonr, spearmanr

# Collect data across all models
entropies = []  # Bias entropy for each model
ood_igs = []    # OOD performance for each model

# Pearson correlation (linear relationship)
r_pearson, p_pearson = pearsonr(entropies, ood_igs)

# Spearman correlation (monotonic relationship)
r_spearman, p_spearman = spearmanr(entropies, ood_igs)

print(f"Pearson r: {r_pearson:.3f} (p={p_pearson:.4f})")
print(f"Spearman ρ: {r_spearman:.3f} (p={p_spearman:.4f})")
```

**Target**: Pearson r ≥ 0.75 (target: 0.83) with p < 0.001

#### Multi-seed Validation

**Usage**: Verify reproducibility across random seeds

```python
import numpy as np

# Run experiment with 5 random seeds
seeds = [42, 123, 456, 789, 1011]
results = []

for seed in seeds:
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Train and evaluate
    model = train_model(seed=seed)
    ig = evaluate_model(model)
    results.append(ig)

# Compute statistics
mean_ig = np.mean(results)
std_ig = np.std(results)
ci_95 = 1.96 * std_ig / np.sqrt(len(seeds))

print(f"Mean IG: {mean_ig:.2f}% ± {std_ig:.2f}%")
print(f"95% CI: [{mean_ig - ci_95:.2f}%, {mean_ig + ci_95:.2f}%]")
```

**Success Criterion**: Standard deviation ≤ 2% across seeds

### 4.5 Additional Metrics (For Analysis)

#### Normalized Scanpath Saliency (NSS)

**Definition**: Measures average saliency value at fixation locations, normalized by saliency map statistics.

```python
def nss(fixation_map, saliency_map):
    """
    Compute Normalized Scanpath Saliency.

    Args:
        fixation_map: Binary fixation locations [H, W]
        saliency_map: Model prediction [H, W]

    Returns:
        nss: NSS score (higher is better)
    """
    # Normalize saliency map
    saliency_map = (saliency_map - saliency_map.mean()) / saliency_map.std()

    # Get fixation locations
    fixation_locs = fixation_map > 0

    # Average normalized saliency at fixations
    nss = saliency_map[fixation_locs].mean()
    return nss
```

**Usage**: Supplementary metric for paper

#### Area Under ROC Curve (AUC)

**Definition**: Measures ability to discriminate fixated from non-fixated locations.

```python
from sklearn.metrics import roc_auc_score

def auc_saliency(fixation_map, saliency_map):
    """
    Compute AUC for saliency prediction.

    Args:
        fixation_map: Binary fixation map [H, W]
        saliency_map: Predicted saliency [H, W]

    Returns:
        auc: AUC score (0.5-1.0, higher is better)
    """
    y_true = fixation_map.flatten()
    y_score = saliency_map.flatten()

    auc = roc_auc_score(y_true, y_score)
    return auc
```

**Usage**: Optional secondary metric

---

## 5. Hardware Requirements

### 5.1 Recommended Setup

**GPU Requirements**:
- **Model**: NVIDIA A100 (40GB) or V100 (32GB)
- **Count**: 2-4 GPUs for parallel experiments
- **Memory**: 32-40GB VRAM per GPU
- **Rationale**:
  - VGG-19 features are memory-intensive
  - Batch sizes of 16-32 require substantial memory
  - Multiple GPUs enable parallel fold training

**System Requirements**:
- **CPU**: 16+ cores (for data loading parallelism)
- **RAM**: 64GB system memory
  - Dataset loading and preprocessing
  - Multiple worker processes for data loaders
- **Storage**: 500GB available space
  - Datasets: ~100GB total
  - Model checkpoints: ~50GB (multiple versions)
  - Experiment logs and outputs: ~50GB
  - Working space: ~300GB buffer

**Network**:
- High-bandwidth connection for dataset download
- Access to compute cluster (if using shared resources)

### 5.2 Minimum Viable Setup

**GPU Requirements**:
- **Model**: NVIDIA RTX 3090 (24GB) or RTX 4090 (24GB)
- **Count**: 1 GPU (extends timeline by ~2x)
- **Memory**: 24GB VRAM minimum
- **Limitations**:
  - Smaller batch sizes (8-16)
  - Sequential training (no parallel folds)
  - Longer training times

**System Requirements**:
- **CPU**: 8+ cores
- **RAM**: 32GB system memory
- **Storage**: 200GB available space

**Timeline Impact**:
- With 1 GPU: 18-20 weeks (vs 11 weeks with 4 GPUs)
- With 2 GPUs: 13-14 weeks

### 5.3 Computational Budget

**Training Time Estimates**:

| Experiment | GPU Hours | Wall-Clock (4 GPUs) | Wall-Clock (1 GPU) |
|------------|-----------|---------------------|-------------------|
| Triage (0.1) | 8-12 | 8-12 hours | 8-12 hours |
| Baseline (1.2-1.3) | 220-330 | 2-3 days | 9-14 days |
| Entropy Reg (2.1-2.2) | 140-210 | 2-3 days | 6-9 days |
| Deconv Model (3.1-3.2) | 150-220 | 2-3 days | 6-9 days |
| Adaptation (4.1-4.3) | 60-90 | 1-2 days | 3-4 days |
| Analysis (5.1-5.6) | 290-390 | 5-7 days | 12-16 days |
| Validation (6.1-6.3) | 230-330 | 4-6 days | 10-14 days |
| **Total** | **1,100-1,600** | **11-13 weeks** | **18-20 weeks** |

**Cost Estimates** (Cloud Computing):

**AWS p4d.24xlarge (8× A100 40GB)**:
- Price: ~$32/hour
- Total cost: 1,200 GPU-hours ÷ 8 GPUs = 150 hours × $32 = **$4,800**

**AWS p3.2xlarge (1× V100 16GB)**:
- Price: ~$3/hour
- Total cost: 1,500 GPU-hours × $3 = **$4,500**
- Requires smaller batch sizes

**GCP a2-highgpu-1g (1× A100 40GB)**:
- Price: ~$3.67/hour
- Total cost: 1,200 GPU-hours × $3.67 = **$4,400**

**Recommendation**:
- Academic cluster access preferred (free GPU hours)
- Cloud compute viable for ~$4,000-5,000 budget
- Use spot/preemptible instances for 60-70% cost savings

### 5.4 Memory Profiling

**Typical Memory Usage**:

```
Training (per GPU):
├─ Model parameters (VGG-19 + readout): ~150 MB
├─ Deconv bias model: ~12 MB
├─ Optimizer states (Adam): ~300 MB (2× parameters)
├─ Batch data (16 images, 512×384): ~4 GB
├─ Forward activations: ~6 GB
├─ Gradients: ~150 MB
└─ Working memory: ~2 GB
────────────────────────────────────────────
Total: ~12-14 GB per GPU (with batch size 16)
```

**Memory Optimization Strategies**:
- **Gradient checkpointing**: Trade compute for memory
- **Mixed precision (FP16)**: 50% memory reduction
- **Smaller batches**: Reduce from 16 to 8 (fit on 12GB GPU)
- **Accumulate gradients**: Simulate larger batches with small memory

```python
# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    with autocast():  # FP16 forward pass
        outputs = model(batch)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()  # FP32 backward
    scaler.step(optimizer)
    scaler.update()
```

### 5.5 Storage Requirements

**Dataset Storage**:
```
Datasets:
├─ MIT1003: ~15 GB (images + fixations)
├─ CAT2000: ~30 GB
├─ COCO Freeview: ~40 GB
├─ Daemons: ~10 GB
└─ Figrim: ~5 GB
────────────────────────────────────────
Total: ~100 GB
```

**Model Checkpoints**:
```
Checkpoints:
├─ Baseline models (5 datasets): 5 × 200 MB = 1 GB
├─ ERSP-ABM models (5 folds × 3 seeds): 15 × 210 MB = 3.2 GB
├─ DeepGaze 3 models: ~2 GB
├─ Intermediate checkpoints: ~5 GB
└─ VGG-19 pretrained: 0.5 GB
────────────────────────────────────────────
Total: ~12 GB
```

**Experiment Outputs**:
```
Outputs:
├─ Training logs (tensorboard/wandb): ~10 GB
├─ Extracted bias maps: ~2 GB
├─ Evaluation results (JSON/CSV): ~1 GB
├─ Visualizations (PNG): ~5 GB
└─ Analysis notebooks: ~2 GB
────────────────────────────────────────
Total: ~20 GB
```

**Working Space**: 300 GB buffer for temporary files

**Total Storage**: **~450-500 GB recommended**

---

## 6. Software Environment

### 6.1 Operating System

**Recommended**: Linux (Ubuntu 20.04 LTS or 22.04 LTS)
- Best CUDA support and driver compatibility
- Standard for deep learning research
- Easy package management

**Alternative**:
- Windows 10/11 with WSL2 (Windows Subsystem for Linux)
- macOS (limited GPU support, CPU-only training)

### 6.2 Python Environment

**Python Version**: 3.9 or 3.10
- PyTorch 2.0+ requires Python ≥3.8
- 3.9/3.10 recommended for stability and compatibility
- Avoid 3.11+ (some packages may not be compatible)

**Environment Management**: Conda (recommended) or virtualenv

```bash
# Create conda environment
conda create -n ersp-abm python=3.10
conda activate ersp-abm

# Or with virtualenv
python3.10 -m venv ersp-abm-env
source ersp-abm-env/bin/activate
```

### 6.3 CUDA and GPU Drivers

**CUDA Version**: 11.7 or 11.8
- Compatible with PyTorch 2.0+
- Stable and widely supported
- CUDA 12.x also supported (newer, less tested)

**cuDNN Version**: 8.5+
- Deep learning primitives for NVIDIA GPUs
- Installed automatically with PyTorch in conda

**NVIDIA Driver**:
- **For CUDA 11.7**: Driver version ≥515.43.04
- **For CUDA 11.8**: Driver version ≥520.61.05
- Check: `nvidia-smi`

**Installation**:
```bash
# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Verify GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

### 6.4 Environment Specification Files

#### requirements.txt

```txt
# Core Deep Learning
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0

# Scientific Computing
numpy>=1.24.0
scipy>=1.10.0

# Computer Vision
Pillow>=9.5.0
opencv-python>=4.7.0
scikit-image>=0.20.0

# Data Handling
h5py>=3.8.0
pandas>=2.0.0
tqdm>=4.65.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Experiment Tracking
wandb>=0.15.0
tensorboard>=2.13.0

# Utilities
einops>=0.6.1
pyyaml>=6.0

# Development
pytest>=7.3.0
black>=23.3.0
ruff>=0.0.270
jupyter>=1.0.0
ipython>=8.12.0

# Machine Learning Utilities
scikit-learn>=1.2.2
```

#### environment.yml (Conda)

```yaml
name: ersp-abm
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults

dependencies:
  - python=3.10
  - pytorch>=2.0.0
  - torchvision>=0.15.0
  - torchaudio>=2.0.0
  - pytorch-cuda=11.8
  - numpy>=1.24.0
  - scipy>=1.10.0
  - matplotlib>=3.7.0
  - seaborn>=0.12.0
  - pandas>=2.0.0
  - h5py>=3.8.0
  - pillow>=9.5.0
  - scikit-image>=0.20.0
  - scikit-learn>=1.2.2
  - jupyter>=1.0.0
  - pytest>=7.3.0
  - pip
  - pip:
    - opencv-python>=4.7.0
    - wandb>=0.15.0
    - tensorboard>=2.13.0
    - einops>=0.6.1
    - black>=23.3.0
    - ruff>=0.0.270
```

**Usage**:
```bash
# Create environment from environment.yml
conda env create -f environment.yml
conda activate ersp-abm

# Or install from requirements.txt
pip install -r requirements.txt
```

### 6.5 Docker Configuration (Optional but Recommended)

**Dockerfile**:

```dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Expose ports for Jupyter and TensorBoard
EXPOSE 8888 6006

CMD ["/bin/bash"]
```

**docker-compose.yml**:

```yaml
version: '3.8'

services:
  ersp-abm:
    build: .
    image: ersp-abm:latest
    container_name: ersp-abm-research

    runtime: nvidia

    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0,1,2,3

    volumes:
      - ./data:/workspace/data
      - ./experiments:/workspace/experiments
      - ./checkpoints:/workspace/checkpoints

    ports:
      - "8888:8888"  # Jupyter
      - "6006:6006"  # TensorBoard

    shm_size: '16gb'  # Shared memory for data loaders

    command: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

**Usage**:
```bash
# Build Docker image
docker-compose build

# Run container
docker-compose up -d

# Access Jupyter Lab
# Open browser: http://localhost:8888

# Run training inside container
docker exec -it ersp-abm-research python train.py
```

---

## 7. Development Tools

### 7.1 Version Control

**Git Configuration**:

```bash
# Initialize repository
git init
git remote add origin <repository-url>

# Create .gitignore
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Environments
ersp-abm-env/
venv/
ENV/

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Data and Checkpoints
data/
checkpoints/
experiments/
*.pth
*.pt

# Logs
logs/
wandb/
tensorboard/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
EOF
```

**Branching Strategy**:
```
main (stable releases)
├── develop (active development)
    ├── feature/entropy-regularization
    ├── feature/deconv-bias-model
    └── feature/few-shot-adaptation
```

### 7.2 Experiment Tracking

#### Weights & Biases (wandb)

**Setup**:
```bash
# Install
pip install wandb

# Login
wandb login

# Initialize in code
import wandb

wandb.init(
    project="ersp-abm",
    name="baseline-mit1003-seed42",
    config={
        "learning_rate": 1e-4,
        "batch_size": 16,
        "epochs": 100,
        "dataset": "MIT1003",
        "model": "DeepGaze2E"
    }
)
```

**Logging During Training**:
```python
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader)
    val_ig = evaluate(model, val_loader)

    # Log to wandb
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_ig": val_ig,
        "learning_rate": optimizer.param_groups[0]['lr']
    })

# Log model checkpoint
wandb.save("checkpoints/model_best.pth")
```

**Features Used**:
- Real-time training curves
- Hyperparameter comparison across runs
- Model checkpoint storage
- Collaborative sharing of results

#### TensorBoard (Alternative)

**Setup**:
```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='runs/baseline-mit1003')

# Log scalars
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('IG/validation', val_ig, epoch)

# Log images
writer.add_images('Bias Maps', bias_maps, epoch)

writer.close()
```

**Launch TensorBoard**:
```bash
tensorboard --logdir=runs --port=6006
# Open browser: http://localhost:6006
```

### 7.3 Code Quality Tools

#### Black (Code Formatter)

```bash
# Format all Python files
black src/ tests/

# Check formatting
black --check src/

# Configuration in pyproject.toml
[tool.black]
line-length = 100
target-version = ['py310']
```

#### Ruff (Linter)

```bash
# Lint code
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/

# Configuration in pyproject.toml
[tool.ruff]
line-length = 100
select = ["E", "F", "W", "I"]
ignore = ["E501"]  # Line too long (handled by black)
```

#### pytest (Testing)

**Test Structure**:
```
tests/
├── test_models.py        # Model architecture tests
├── test_data.py          # Dataset and data loader tests
├── test_metrics.py       # Evaluation metrics tests
└── test_training.py      # Training loop tests
```

**Example Test**:
```python
# tests/test_metrics.py
import pytest
import numpy as np
from src.metrics import information_gain, entropy

def test_information_gain_perfect_prediction():
    """Test IG = 1 for perfect prediction."""
    fixations = np.random.rand(100, 100)
    fixations = fixations / fixations.sum()

    # Perfect prediction = fixations themselves
    ig = information_gain(fixations, fixations, baseline='uniform')

    assert np.isclose(ig, 1.0, atol=0.01)

def test_entropy_uniform():
    """Test entropy of uniform distribution."""
    uniform = np.ones((100, 100)) / 10000
    H = entropy(uniform)

    # Should equal log(10000)
    assert np.isclose(H, np.log(10000), atol=0.01)
```

**Run Tests**:
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Generate coverage report
pytest --cov=src --cov-report=html tests/
# Open htmlcov/index.html
```

### 7.4 Jupyter Notebooks

**Analysis Notebooks**:
```
notebooks/
├── 01_data_exploration.ipynb        # Dataset statistics and visualization
├── 02_bias_extraction.ipynb         # Implicit bias extraction analysis
├── 03_entropy_analysis.ipynb        # Entropy-IG correlation study
├── 04_results_visualization.ipynb   # Main results plots
└── 05_failure_analysis.ipynb        # Error case investigation
```

**Best Practices**:
- Use notebooks for exploration and visualization only
- Move production code to .py modules
- Clear outputs before committing (nbstripout)
- Include markdown explanations

**nbstripout** (Strip outputs from notebooks):
```bash
# Install
pip install nbstripout

# Setup for git
nbstripout --install

# Now notebooks automatically stripped on commit
```

### 7.5 Documentation Tools

#### Sphinx (API Documentation)

**Setup**:
```bash
# Install
pip install sphinx sphinx-rtd-theme

# Initialize
cd docs/
sphinx-quickstart

# Configuration in docs/conf.py
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

html_theme = 'sphinx_rtd_theme'
```

**Docstring Format** (NumPy style):
```python
def information_gain(fixation_map, prediction, baseline='uniform'):
    """
    Compute Information Gain metric for saliency prediction.

    Parameters
    ----------
    fixation_map : np.ndarray
        Ground truth fixation density map, shape (H, W)
    prediction : np.ndarray
        Model's predicted saliency map, shape (H, W)
    baseline : str or np.ndarray, default='uniform'
        Baseline distribution, either 'uniform' or array of shape (H, W)

    Returns
    -------
    ig : float
        Information gain value between 0 and 1, where higher is better

    Examples
    --------
    >>> fixations = np.random.rand(100, 100)
    >>> prediction = model.predict(image)
    >>> ig = information_gain(fixations, prediction)
    >>> print(f"IG: {ig:.2%}")
    """
    pass
```

**Build Docs**:
```bash
cd docs/
make html
# Open _build/html/index.html
```

---

## 8. Data Processing Pipeline

### 8.1 Image Preprocessing

**Standard Pipeline**:

```python
import torch
from torchvision import transforms
from PIL import Image

# Training transforms
train_transform = transforms.Compose([
    transforms.Resize((384, 512)),           # Standardize resolution
    transforms.ToTensor(),                    # Convert to tensor [0, 1]
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],          # ImageNet statistics
        std=[0.229, 0.224, 0.225]
    )
])

# Validation transforms (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((384, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Apply
image = Image.open('image.jpg').convert('RGB')
image_tensor = train_transform(image)  # [3, 384, 512]
```

**Optional Data Augmentation** (for training):
```python
# Augmentation for improved generalization
augmentation_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
        hue=0.05
    ),
    transforms.Resize((384, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])
```

**Note**: Use augmentation carefully - fixation maps must be transformed consistently with images (horizontal flip applies to both).

### 8.2 Fixation Map Processing

**From Raw Fixations to Density Maps**:

```python
import numpy as np
from scipy.ndimage import gaussian_filter

def fixations_to_density_map(fixations, image_shape, sigma=3):
    """
    Convert discrete fixation coordinates to continuous density map.

    Parameters
    ----------
    fixations : list of tuples
        [(x1, y1), (x2, y2), ...] fixation coordinates
    image_shape : tuple
        (H, W) output density map shape
    sigma : float
        Gaussian blur standard deviation (pixels)

    Returns
    -------
    density_map : np.ndarray
        Continuous density map, shape (H, W), sums to 1
    """
    H, W = image_shape

    # Create binary fixation map
    fixation_map = np.zeros((H, W), dtype=np.float32)

    for x, y in fixations:
        # Ensure coordinates are within bounds
        x = int(np.clip(x, 0, W-1))
        y = int(np.clip(y, 0, H-1))
        fixation_map[y, x] += 1

    # Apply Gaussian blur
    density_map = gaussian_filter(fixation_map, sigma=sigma)

    # Normalize to probability distribution
    if density_map.sum() > 0:
        density_map = density_map / density_map.sum()
    else:
        # No fixations: return uniform
        density_map = np.ones((H, W)) / (H * W)

    return density_map
```

**Multiple Observers**:
```python
def aggregate_fixations(observer_fixations, image_shape, sigma=3):
    """
    Aggregate fixations from multiple observers.

    Parameters
    ----------
    observer_fixations : list of lists
        [[(x1, y1), ...], [(x1, y1), ...], ...]
        Outer list: observers, inner list: fixations per observer
    image_shape : tuple
        (H, W) output shape
    sigma : float
        Gaussian blur sigma

    Returns
    -------
    aggregated_map : np.ndarray
        Aggregated density map, shape (H, W)
    """
    # Create density map for each observer
    observer_maps = []
    for fixations in observer_fixations:
        density = fixations_to_density_map(fixations, image_shape, sigma)
        observer_maps.append(density)

    # Average across observers
    aggregated_map = np.mean(observer_maps, axis=0)

    # Renormalize
    aggregated_map = aggregated_map / aggregated_map.sum()

    return aggregated_map
```

### 8.3 Centerbias Extraction

**Extract Empirical Centerbias from Dataset**:

```python
def extract_centerbias(fixation_maps):
    """
    Extract empirical centerbias from collection of fixation maps.

    Parameters
    ----------
    fixation_maps : list of np.ndarray
        List of fixation density maps, each shape (H, W)

    Returns
    -------
    centerbias : np.ndarray
        Average fixation pattern, shape (H, W), normalized
    """
    # Stack and average
    centerbias = np.mean(np.stack(fixation_maps), axis=0)

    # Normalize
    centerbias = centerbias / centerbias.sum()

    return centerbias

# Usage
all_fixation_maps = [load_fixation_map(img_id) for img_id in dataset]
mit1003_centerbias = extract_centerbias(all_fixation_maps)

# Save for later use
np.save('data/centerbiases/mit1003_centerbias.npy', mit1003_centerbias)
```

**Visualize Centerbias**:
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
datasets = ['MIT1003', 'CAT2000', 'COCO Freeview', 'Daemons', 'Figrim']

for ax, dataset_name in zip(axes, datasets):
    centerbias = np.load(f'data/centerbiases/{dataset_name}_centerbias.npy')

    im = ax.imshow(centerbias, cmap='hot')
    ax.set_title(f'{dataset_name}\nEntropy: {entropy(centerbias):.3f} nats')
    ax.axis('off')

    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('figures/dataset_centerbiases.png', dpi=300)
```

### 8.4 Uniform Image Generation

**Generate Uniform-Colored Images for Bias Extraction**:

```python
def generate_uniform_images(image_shape, intensities=None):
    """
    Generate uniform-colored images for implicit bias extraction.

    Parameters
    ----------
    image_shape : tuple
        (C, H, W) image tensor shape
    intensities : list of int, optional
        Gray intensities to use (0-255). Default: [0, 36, 73, 109, 146, 182, 219, 255]

    Returns
    -------
    uniform_images : torch.Tensor
        Batch of uniform images, shape (N, C, H, W)
    """
    if intensities is None:
        intensities = [0, 36, 73, 109, 146, 182, 219, 255]

    C, H, W = image_shape
    uniform_images = []

    for intensity in intensities:
        # Create uniform image (all pixels same value)
        img = torch.full((C, H, W), intensity / 255.0, dtype=torch.float32)
        uniform_images.append(img)

    # Stack into batch
    uniform_batch = torch.stack(uniform_images)  # [N, C, H, W]

    return uniform_batch

# Usage
uniform_batch = generate_uniform_images((3, 384, 512))
print(f"Generated {len(uniform_batch)} uniform images")

# Extract bias by averaging predictions
model.eval()
with torch.no_grad():
    predictions = model(uniform_batch.to(device))
    implicit_bias = predictions.mean(dim=0)  # Average across colors
```

### 8.5 Batch Construction

**Training Batch with Entropy Regularization**:

```python
class SaliencyDataLoader:
    """
    Custom data loader that appends uniform images to batches
    for entropy regularization.
    """

    def __init__(self, dataset, batch_size, add_uniform=True, num_workers=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.add_uniform = add_uniform
        self.num_workers = num_workers

        # Standard PyTorch data loader
        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

    def __iter__(self):
        for batch in self.loader:
            images = batch['image']          # [B, 3, H, W]
            fixations = batch['fixation']    # [B, H, W]
            centerbias = batch['centerbias'] # [H, W]

            if self.add_uniform:
                # Generate one uniform image
                B, C, H, W = images.shape
                uniform_intensity = np.random.randint(0, 256)
                uniform_img = torch.full((1, C, H, W), uniform_intensity / 255.0)

                # Uniform centerbias (all pixels equal)
                uniform_centerbias = torch.ones((1, H, W)) / (H * W)

                # Append to batch
                images = torch.cat([images, uniform_img], dim=0)
                fixations = torch.cat([fixations, uniform_centerbias], dim=0)

            yield {
                'image': images,
                'fixation': fixations,
                'centerbias': centerbias,
                'is_uniform': torch.cat([
                    torch.zeros(self.batch_size, dtype=torch.bool),
                    torch.ones(1, dtype=torch.bool)
                ]) if self.add_uniform else torch.zeros(self.batch_size, dtype=torch.bool)
            }

    def __len__(self):
        return len(self.loader)
```

**Usage in Training Loop**:
```python
train_loader = SaliencyDataLoader(
    train_dataset,
    batch_size=16,
    add_uniform=True  # Enable entropy regularization
)

for batch in train_loader:
    images = batch['image']          # [17, 3, H, W] (16 + 1 uniform)
    fixations = batch['fixation']    # [17, H, W]
    is_uniform = batch['is_uniform'] # [17] boolean mask

    # Forward pass
    predictions = model(images)

    # Separate losses
    task_loss = criterion(predictions[~is_uniform], fixations[~is_uniform])
    entropy_loss = -entropy(predictions[is_uniform])  # Maximize entropy

    # Combined loss
    total_loss = task_loss + lambda_reg * entropy_loss
```

---

## 9. Model Training Infrastructure

### 9.1 Training Configuration

**Hyperparameters** (from research plan and literature):

```python
# Training configuration
config = {
    # Optimization
    'optimizer': 'Adam',
    'learning_rate': 1e-4,           # Standard for DeepGaze II
    'weight_decay': 1e-5,             # L2 regularization
    'betas': (0.9, 0.999),            # Adam parameters

    # Training schedule
    'epochs': 100,                    # Train until convergence
    'batch_size': 16,                 # Adjust based on GPU memory
    'gradient_clip': 1.0,             # Prevent gradient explosion

    # Entropy regularization
    'lambda_entropy': 0.1,            # Regularization weight (tune in Phase 2)
    'add_uniform_images': True,       # Enable entropy regularization

    # Learning rate schedule
    'lr_scheduler': 'ReduceLROnPlateau',
    'lr_patience': 10,                # Epochs without improvement
    'lr_factor': 0.5,                 # Reduce LR by half

    # Early stopping
    'early_stopping': True,
    'patience': 20,                   # Epochs without improvement

    # Checkpointing
    'save_best': True,                # Save best validation model
    'save_interval': 10,              # Save checkpoint every N epochs

    # Mixed precision
    'use_amp': True,                  # Automatic mixed precision (FP16)

    # Data loading
    'num_workers': 4,                 # Parallel data loading
    'pin_memory': True,               # Faster CPU-GPU transfer
}
```

### 9.2 Loss Functions

**Primary Task Loss** (Saliency Prediction):

```python
def kl_divergence_loss(prediction, target, epsilon=1e-10):
    """
    KL divergence loss for saliency prediction.

    KL(target || prediction) = sum(target * log(target / prediction))

    Parameters
    ----------
    prediction : torch.Tensor
        Predicted saliency map, shape (B, H, W)
    target : torch.Tensor
        Ground truth fixation map, shape (B, H, W)
    epsilon : float
        Small constant for numerical stability

    Returns
    -------
    loss : torch.Tensor
        Scalar KL divergence loss
    """
    # Normalize to probability distributions
    prediction = prediction / (prediction.sum(dim=(-2, -1), keepdim=True) + epsilon)
    target = target / (target.sum(dim=(-2, -1), keepdim=True) + epsilon)

    # Add epsilon for numerical stability
    prediction = prediction + epsilon
    target = target + epsilon

    # KL divergence
    kl = target * torch.log(target / prediction)
    loss = kl.sum(dim=(-2, -1)).mean()  # Average over batch

    return loss
```

**Entropy Regularization Loss**:

```python
def entropy_loss(prediction, epsilon=1e-10):
    """
    Negative entropy loss (minimizing = maximizing entropy).

    H(P) = -sum(P * log(P))

    Parameters
    ----------
    prediction : torch.Tensor
        Predicted saliency map, shape (B, H, W)
    epsilon : float
        Numerical stability constant

    Returns
    -------
    neg_entropy : torch.Tensor
        Negative entropy (minimize to maximize entropy)
    """
    # Normalize to probability distribution
    p = prediction / (prediction.sum(dim=(-2, -1), keepdim=True) + epsilon)
    p = p + epsilon

    # Shannon entropy
    H = -(p * torch.log(p)).sum(dim=(-2, -1))

    # Return negative (minimizing negative = maximizing positive)
    return -H.mean()
```

**Combined Loss**:

```python
def compute_loss(model, batch, lambda_entropy=0.1):
    """
    Compute combined task loss and entropy regularization loss.

    Parameters
    ----------
    model : nn.Module
        Saliency prediction model
    batch : dict
        Batch data with images, fixations, and uniform mask
    lambda_entropy : float
        Entropy regularization weight

    Returns
    -------
    total_loss : torch.Tensor
        Combined loss
    losses_dict : dict
        Individual loss components for logging
    """
    images = batch['image']
    fixations = batch['fixation']
    is_uniform = batch['is_uniform']

    # Forward pass
    predictions = model(images)

    # Task loss (non-uniform images)
    if (~is_uniform).any():
        task_loss = kl_divergence_loss(
            predictions[~is_uniform],
            fixations[~is_uniform]
        )
    else:
        task_loss = torch.tensor(0.0, device=images.device)

    # Entropy regularization loss (uniform images)
    if is_uniform.any():
        entropy_reg_loss = entropy_loss(predictions[is_uniform])
    else:
        entropy_reg_loss = torch.tensor(0.0, device=images.device)

    # Combined loss
    total_loss = task_loss + lambda_entropy * entropy_reg_loss

    return total_loss, {
        'total_loss': total_loss.item(),
        'task_loss': task_loss.item(),
        'entropy_reg_loss': entropy_reg_loss.item()
    }
```

### 9.3 Training Loop

**Main Training Function**:

```python
def train_model(model, train_loader, val_loader, config, device='cuda'):
    """
    Main training loop with validation and checkpointing.

    Parameters
    ----------
    model : nn.Module
        Saliency prediction model
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    config : dict
        Training configuration
    device : str
        Device to train on

    Returns
    -------
    model : nn.Module
        Trained model
    history : dict
        Training history (losses, metrics)
    """
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=config['betas']
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Maximize validation IG
        factor=config['lr_factor'],
        patience=config['lr_patience'],
        verbose=True
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config['use_amp'] else None

    # Training history
    history = {
        'train_loss': [],
        'val_ig': [],
        'learning_rate': []
    }

    # Early stopping
    best_val_ig = 0.0
    patience_counter = 0

    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            optimizer.zero_grad()

            # Forward pass with mixed precision
            if config['use_amp']:
                with torch.cuda.amp.autocast():
                    loss, loss_dict = compute_loss(
                        model, batch, lambda_entropy=config['lambda_entropy']
                    )
                scaler.scale(loss).backward()

                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config['gradient_clip']
                )

                scaler.step(optimizer)
                scaler.update()
            else:
                loss, loss_dict = compute_loss(
                    model, batch, lambda_entropy=config['lambda_entropy']
                )
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config['gradient_clip']
                )

                optimizer.step()

            train_losses.append(loss_dict['total_loss'])

        # Validation phase
        model.eval()
        val_ig = evaluate_information_gain(model, val_loader, device)

        # Update history
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        history['val_ig'].append(val_ig)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val IG = {val_ig:.4f}")

        # Learning rate scheduling
        scheduler.step(val_ig)

        # Checkpointing
        if val_ig > best_val_ig:
            best_val_ig = val_ig
            patience_counter = 0

            if config['save_best']:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_ig': val_ig,
                    'config': config
                }, 'checkpoints/model_best.pth')
        else:
            patience_counter += 1

        # Early stopping
        if config['early_stopping'] and patience_counter >= config['patience']:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

        # Periodic checkpoint
        if (epoch + 1) % config['save_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ig': val_ig,
                'config': config
            }, f'checkpoints/model_epoch{epoch+1}.pth')

    # Load best model
    if config['save_best']:
        checkpoint = torch.load('checkpoints/model_best.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']} (Val IG: {checkpoint['val_ig']:.4f})")

    return model, history
```

### 9.4 Few-Shot Adaptation

**Adaptation Function**:

```python
def adapt_bias_model(model, adaptation_loader, num_iterations=50, lr=1e-3, device='cuda'):
    """
    Adapt implicit bias model using few-shot samples.

    Freezes all parameters except the deconv input tensor.

    Parameters
    ----------
    model : nn.Module
        Trained model with deconv bias component
    adaptation_loader : DataLoader
        Few-shot adaptation data (100 samples)
    num_iterations : int
        Number of adaptation epochs
    lr : float
        Learning rate for adaptation
    device : str
        Device

    Returns
    -------
    model : nn.Module
        Adapted model
    """
    model = model.to(device)
    model.train()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only deconv input tensor
    model.deconv_bias_model.input_tensor.requires_grad = True

    # Optimizer for input tensor only
    optimizer = torch.optim.Adam(
        [model.deconv_bias_model.input_tensor],
        lr=lr
    )

    print(f"Adapting with {len(adaptation_loader.dataset)} samples for {num_iterations} iterations")

    for iteration in range(num_iterations):
        total_loss = 0.0

        for batch in adaptation_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            optimizer.zero_grad()

            # Forward pass
            predictions = model(batch['image'])

            # Task loss only (no regularization during adaptation)
            loss = kl_divergence_loss(predictions, batch['fixation'])

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(adaptation_loader)

        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration+1}/{num_iterations}: Loss = {avg_loss:.4f}")

    # Refreeze for evaluation
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model
```

### 9.5 Distributed Training (Optional)

**Multi-GPU Setup** (if using 2-4 GPUs):

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed(rank, world_size):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def train_distributed(rank, world_size, config):
    """Training function for distributed setup."""
    setup_distributed(rank, world_size)

    # Create model and move to GPU
    model = create_model(config).to(rank)
    model = DDP(model, device_ids=[rank])

    # Create distributed sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=config['num_workers']
    )

    # Train
    train_model(model, train_loader, val_loader, config, device=rank)

    dist.destroy_process_group()

# Launch distributed training
if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        train_distributed,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )
```

---

## 10. Reproducibility Requirements

### 10.1 Random Seed Management

**Seed Everything Function**:

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

# Usage at the start of every script
set_seed(42)
```

**Seed Management in Experiments**:
```python
# Different seeds for multi-seed validation
SEEDS = [42, 123, 456, 789, 1011]

for seed in SEEDS:
    set_seed(seed)

    # Train model
    model = create_model()
    trained_model, history = train_model(model, train_loader, val_loader, config)

    # Evaluate
    results = evaluate(trained_model, test_loader)

    # Save with seed identifier
    save_results(results, f'results/seed{seed}_results.json')
```

### 10.2 Deterministic Operations

**PyTorch Determinism**:

```python
# Enable deterministic algorithms
torch.use_deterministic_algorithms(True)

# Handle operations that don't have deterministic implementation
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Warning: Some operations may be slower in deterministic mode
# Trade-off between reproducibility and speed
```

**Data Loader Determinism**:
```python
def worker_init_fn(worker_id):
    """Initialize workers with different seeds."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    worker_init_fn=worker_init_fn,  # Ensure deterministic workers
    generator=torch.Generator().manual_seed(42)  # Deterministic shuffling
)
```

### 10.3 Environment Specification

**Complete environment.yml**:

```yaml
name: ersp-abm
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults

dependencies:
  - python=3.10.11
  - pytorch=2.0.1
  - torchvision=0.15.2
  - torchaudio=2.0.2
  - pytorch-cuda=11.8
  - numpy=1.24.3
  - scipy=1.10.1
  - matplotlib=3.7.1
  - seaborn=0.12.2
  - pandas=2.0.2
  - h5py=3.8.0
  - pillow=9.5.0
  - scikit-image=0.20.0
  - scikit-learn=1.2.2
  - jupyter=1.0.0
  - pytest=7.3.1
  - pip=23.1.2
  - pip:
    - opencv-python==4.7.0.72
    - wandb==0.15.3
    - tensorboard==2.13.0
    - einops==0.6.1
    - black==23.3.0
    - ruff==0.0.270
    - tqdm==4.65.0
```

**Lock file** (for exact reproduction):
```bash
# Generate lock file with exact versions
conda env export --no-builds > environment_lock.yml

# Create environment from lock file
conda env create -f environment_lock.yml
```

### 10.4 Code Version Control

**Git Practices**:

```bash
# Tag releases for experiments
git tag -a v1.0-baseline "Baseline experiments"
git tag -a v2.0-entropy-reg "Entropy regularization"
git tag -a v3.0-full-framework "Full ERSP-ABM framework"

# Push tags
git push origin --tags

# Checkout specific version
git checkout v2.0-entropy-reg
```

**Track Experiment Commits**:
```python
import subprocess

def get_git_commit():
    """Get current git commit hash."""
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).decode('ascii').strip()
        return commit
    except:
        return 'unknown'

# Log commit with experiment
config['git_commit'] = get_git_commit()
```

### 10.5 Experiment Tracking

**Complete Experiment Metadata**:

```python
experiment_metadata = {
    # Code version
    'git_commit': get_git_commit(),
    'git_branch': 'main',

    # Environment
    'python_version': sys.version,
    'torch_version': torch.__version__,
    'cuda_version': torch.version.cuda,
    'cudnn_version': torch.backends.cudnn.version(),

    # Hardware
    'gpu_name': torch.cuda.get_device_name(0),
    'gpu_count': torch.cuda.device_count(),

    # Experiment
    'experiment_name': 'baseline-mit1003',
    'dataset': 'MIT1003',
    'model': 'DeepGaze2E',
    'seed': 42,
    'timestamp': datetime.datetime.now().isoformat(),

    # Hyperparameters
    'config': config,

    # Results
    'final_val_ig': 0.923,
    'best_epoch': 87,
    'training_time_hours': 12.5,
}

# Save metadata
with open('experiments/baseline-mit1003/metadata.json', 'w') as f:
    json.dump(experiment_metadata, f, indent=2)
```

### 10.6 Reproducibility Checklist

**Before Running Experiments**:
- [ ] Set random seed at beginning of script
- [ ] Enable deterministic algorithms
- [ ] Pin dependency versions in environment.yml
- [ ] Document hardware specifications
- [ ] Tag git commit for experiment
- [ ] Initialize experiment tracking (wandb/tensorboard)

**During Experiments**:
- [ ] Log all hyperparameters
- [ ] Save checkpoints with metadata
- [ ] Record training time and resource usage
- [ ] Track git commit in experiment logs

**After Experiments**:
- [ ] Save final model checkpoint
- [ ] Export training curves and metrics
- [ ] Document any manual interventions
- [ ] Generate reproducibility report
- [ ] Archive code, data, and results

**For Publication**:
- [ ] Create release tag in git
- [ ] Prepare code release repository
- [ ] Upload model checkpoints
- [ ] Write detailed README with reproduction steps
- [ ] Include requirements.txt and environment.yml
- [ ] Provide example scripts
- [ ] Document expected results and tolerances

---

## 11. Summary

This technical stack documentation provides a complete specification of the infrastructure required for the **ERSP-ABM** research project. Key highlights:

**Core Technology**:
- PyTorch 2.0+ with VGG-19 backbone
- Custom deconvolutional bias modeling architecture
- Entropy regularization training framework
- Few-shot adaptation with frozen networks

**Datasets**:
- 5 standard saliency benchmarks (MIT1003, CAT2000, COCO Freeview, Daemons, Figrim)
- Leave-one-out cross-validation protocol
- ~100GB total storage requirement

**Evaluation**:
- Information Gain (IG) as primary metric
- Entropy analysis for bias quantification
- Statistical significance testing (p < 0.01 target)
- Correlation analysis (r ≥ 0.83 target)

**Computational Resources**:
- 2-4 NVIDIA A100/V100 GPUs (recommended)
- 1,200 GPU-hours estimated
- 11-week timeline with 4 GPUs
- ~$4,000-5,000 cloud compute budget

**Reproducibility**:
- Complete environment specifications (conda/docker)
- Deterministic training with seed management
- Comprehensive experiment tracking
- Version control and metadata logging

**Development Tools**:
- Weights & Biases / TensorBoard for tracking
- pytest for testing, black/ruff for code quality
- Jupyter notebooks for analysis
- Sphinx for documentation

This stack enables reproducible implementation of the research with clear success criteria: **6.3% average OOD IG improvement** with **0.3% in-domain degradation** and **r=0.83 entropy-IG correlation**.

---

**Document Version**: 1.0
**Last Updated**: 2025-01-XX
**Next Review**: After Phase 0 (Triage Experiment)
