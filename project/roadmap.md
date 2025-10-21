# Research Experiment Roadmap

## Overview

This roadmap outlines the experimental plan for validating the **Entropy-Regularized Saliency Prediction with Adaptive Bias Modeling (ERSP-ABM)** framework. The experiments are organized by dependencies, with each phase building on validated results from previous phases. Our goal is to demonstrate that entropy regularization during training, combined with explicit deconvolutional bias modeling and few-shot adaptation, achieves superior cross-dataset generalization (target: +5% absolute Information Gain improvement) while maintaining competitive in-domain performance (≤1% degradation).

**Target Outcome**: Average cross-dataset IG of 87.6% (up from 81.3% baseline), with successful validation of the hypothesis that reduced implicit bias (measured by increased entropy) correlates with improved OOD performance.

---

## Phase 0: Minimum Triage Experiment (Days 1-3)

**CRITICAL: This experiment determines go/no-go for the entire research project**

### Experiment 0.1: Core Hypothesis Validation - Entropy Regularization Effect

**Objective**: Quickly validate that entropy regularization on uniform images reduces implicit bias and shows promise for improved generalization.

**Duration**: 2-3 days maximum

**Approach**:
1. Train **minimal DeepGaze 2E model** on single dataset (MIT1003, subset of 500 images)
2. Two configurations:
   - **Baseline**: Standard training with MIT1003 centerbias
   - **+Entropy Reg**: Append 1 uniform-colored image per batch, maximize output entropy
3. Extract implicit bias using uniform color images (average over 8 colors: 0, 36, 73, 109, 146, 182, 219, 255)
4. Test on small validation split + small sample from one OOD dataset (CAT2000, 50 images)

**Required Resources**:
- Dataset: 500 training images from MIT1003, 50 val images, 50 OOD images from CAT2000
- Compute: Single GPU, ~8-12 hours total (2 models × 4-6 hours)
- Implementation: Basic DeepGaze 2E architecture, simple entropy loss

**Baseline Comparison**:
- Naive baseline: Uniform centerbias (entropy = max)
- Standard training: DeepGaze 2E without regularization
- Check if entropy regularization increases extracted bias entropy by ≥5%

**Success Criteria**:
- [ ] Entropy regularization increases extracted bias entropy by ≥5% (target: 6.0 vs 5.7 nats)
- [ ] Training remains stable (loss converges, no NaN/divergence)
- [ ] In-domain validation IG degradation ≤2% (acceptable for triage)
- [ ] OOD performance shows any improvement trend (even if not statistically significant)
- [ ] Uniform image extraction method works as expected (non-zero variance in bias maps)

**Metrics to Track**:
- Extracted bias entropy (primary go/no-go signal)
- In-domain IG on MIT1003 val
- OOD IG on CAT2000 subset
- Training time overhead from uniform images

**Decision Gate**:
- **GO** (Proceed to Phase 1): If entropy increases ≥5% AND training is stable AND in-domain degradation ≤2%
- **PIVOT**: If entropy increases but in-domain degradation >2%, investigate regularization weight tuning
- **NO-GO**: If entropy does not increase OR training becomes unstable OR fundamental implementation issues

**Fallback Plan**: If uniform image approach fails, try alternative regularization: maximize prediction entropy on randomly selected training images (less principled but tests entropy mechanism).

---

## Phase 1: Foundation & Baselines (Weeks 1-2)

### Experiment 1.1: Dataset Preparation & Analysis

**Depends on**: Experiment 0.1 success (GO decision)

**Objective**: Prepare all five datasets with proper train/val splits and analyze their spatial statistics to understand bias characteristics.

**Duration**: 3-4 days

**Datasets to Prepare**:
1. **MIT1003**: Standard benchmark, primary training dataset
2. **CAT2000**: Diverse scene categories
3. **COCO Freeview**: Free-viewing on natural images
4. **Daemons**: Specialized dataset
5. **Figrim**: Additional diversity

**Tasks**:
- Download and preprocess all datasets (resize, normalize, format fixation maps)
- Extract empirical centerbiases from each dataset
- Create standardized train/val splits (80/20) for each dataset
- Implement data loaders compatible with leave-one-out protocol
- Analyze spatial statistics:
  - Compute entropy of empirical centerbiases
  - Visualize spatial distributions
  - Calculate KL divergences between dataset centerbiases

**Deliverables**:
- [ ] Five preprocessed datasets in standardized format
- [ ] Empirical centerbias maps for each dataset
- [ ] Data analysis notebook with:
  - Entropy measurements per dataset
  - KL divergence matrix between datasets
  - Visualization of centerbias differences
- [ ] Documented data loading pipeline with validation

**Success Criteria**:
- [ ] All datasets load correctly with consistent dimensions
- [ ] Empirical centerbiases match expected patterns (center-weighted for free-viewing)
- [ ] Splits are balanced and representative
- [ ] Data statistics are documented and reproducible

**Expected Results**:
- Dataset centerbiases should show entropy variation (lower entropy = stronger bias)
- KL divergences between datasets should be significant (justifying cross-dataset evaluation)
- MIT1003 centerbias entropy: ~5.8-6.2 nats (based on notebook findings)

---

### Experiment 1.2: Baseline Reproduction - DeepGaze 2E

**Depends on**: Experiment 1.1 completion

**Objective**: Reproduce baseline DeepGaze 2E performance on all five datasets to establish reliable baselines for comparison.

**Duration**: 4-5 days

**Implementation**:
- DeepGaze 2E architecture (VGG-19 features + readout layers)
- Train from scratch with standard protocol:
  - Optimizer: Adam
  - Learning rate: 1e-4 (based on DeepGaze II paper)
  - Batch size: 16-32
  - Training epochs: 50-100 (until convergence)
- External centerbias: Use MIT1003 empirical centerbias
- No regularization (pure baseline)

**Training Protocol**:
1. Train one model per dataset (5 models total)
2. Track convergence on validation splits
3. Extract implicit bias using uniform images:
   - Feed 8 uniform-colored images (0-255 intensity range)
   - Average predictions to get implicit bias map
   - Compute entropy of extracted bias

**Evaluation**:
- In-domain IG on validation split of training dataset
- Information Gain metric calculation
- Bias extraction and entropy measurement

**Expected Results** (from literature and notebook):
- **MIT1003**: In-domain IG ~92-93% (matches DeepGaze IIE performance)
- **Extracted bias entropy**: ~5.9-6.0 nats (lower than uniform ~7.1 nats)
- **KL divergence from uniform**: Moderate (0.8-1.2 nats)

**Success Criteria**:
- [ ] Models converge stably on all datasets
- [ ] In-domain IG ≥90% on MIT1003 (within range of published results)
- [ ] Extracted bias entropy is measurably below uniform (confirming implicit bias exists)
- [ ] Training is reproducible across random seeds (variance <2%)

**Fallback**: If can't achieve 90%+ IG, validate against published DeepGaze II results (87%) and proceed with our achieved baseline as reference.

---

### Experiment 1.3: Cross-Dataset Baseline Evaluation

**Depends on**: Experiment 1.2 completion

**Objective**: Establish baseline cross-dataset performance for leave-one-out evaluation.

**Duration**: 2-3 days

**Protocol**:
1. For each of 5 folds (leave one dataset out):
   - Train DeepGaze 2E on 4 datasets (mixed training set)
   - Evaluate on validation set of left-out dataset
   - Measure Information Gain

**Example Fold 1** (MIT1003 left out):
- Train on: CAT2000 + COCO Freeview + Daemons + Figrim
- Test on: MIT1003 validation set
- Use: Mixed centerbias (average of 4 training datasets)

**Metrics**:
- Cross-dataset IG for each fold
- Average OOD IG across all 5 folds
- Comparison with in-domain performance (performance drop)

**Expected Results** (from mission):
- **Average baseline OOD IG**: ~81-82% (based on hypothetical results)
- **Performance drop**: ~10-12% below in-domain (91-93% → 81-82%)
- **Variation across datasets**: Figrim and Daemons should show worst performance (76-78%)

**Success Criteria**:
- [ ] All five folds complete successfully
- [ ] Average OOD IG is 75-85% (reasonable range)
- [ ] Performance drop from in-domain is consistent (8-15% range)
- [ ] Results are reproducible across random seeds

**Deliverables**:
- Baseline OOD performance table for all 5 folds
- Statistical analysis (mean, std across seeds)
- Baseline bias entropy measurements

---

## Phase 2: Entropy Regularization Development (Weeks 3-4)

### Experiment 2.1: Implement Entropy Regularization

**Depends on**: Validated baselines from Experiment 1.3

**Objective**: Implement and validate the entropy regularization mechanism during training.

**Duration**: 4-5 days

**Implementation Details**:
1. **Uniform Image Generation**:
   - For each batch, append one uniform-colored image
   - Color: Random intensity value (0-255)
   - Shape: Same as training images
   - Centerbias: Uniform distribution (all pixels = 1/numel)

2. **Entropy Loss**:
   - For uniform image prediction: `L_entropy = -entropy(p)` where `p` is softmax of model output
   - Entropy calculation: `H = -sum(p * log(p))` over spatial dimensions
   - Maximize entropy ↔ minimize negative entropy

3. **Combined Loss**:
   ```
   L_total = L_task + λ * L_entropy
   ```
   - `L_task`: Standard saliency loss (e.g., KL divergence, NSS)
   - `λ`: Regularization weight (hyperparameter to tune)

**Hyperparameter Search**:
- Test `λ ∈ {0.01, 0.05, 0.1, 0.5, 1.0}`
- Criterion: Maximize bias entropy while maintaining in-domain IG ≥90%
- Use MIT1003 for initial tuning

**Training**:
- Train DeepGaze 2E with entropy regularization on MIT1003
- Compare extracted bias entropy with baseline
- Measure in-domain validation IG

**Success Criteria**:
- [ ] Entropy regularization increases extracted bias entropy by ≥10% (target: 6.5+ nats vs 5.9 baseline)
- [ ] In-domain IG degradation ≤1% (maintain ≥92% on MIT1003)
- [ ] Training remains stable (no divergence)
- [ ] Training time overhead ≤15%

**Deliverables**:
- [ ] Implemented entropy regularization module
- [ ] Hyperparameter sweep results
- [ ] Comparison: baseline vs regularized bias maps (visualizations)
- [ ] Entropy measurements across different λ values

**Expected Results**:
- Optimal `λ` likely in range 0.1-0.5 (balances entropy and task performance)
- Extracted bias entropy: ~6.5-6.8 nats (vs baseline 5.9)
- In-domain IG: ~92-93% (minimal degradation)

---

### Experiment 2.2: Entropy Regularization - Full Dataset Evaluation

**Depends on**: Experiment 2.1 success (found optimal λ)

**Objective**: Validate entropy regularization across all five datasets and measure initial OOD improvements.

**Duration**: 3-4 days

**Protocol**:
1. Train DeepGaze 2E with optimal `λ` on each dataset (5 models)
2. Extract implicit biases and measure entropy
3. Perform leave-one-out cross-dataset evaluation (5 folds)
4. Compare with Experiment 1.3 baseline

**Evaluation**:
- In-domain IG for each dataset (with regularization)
- Cross-dataset OOD IG for each fold
- Bias entropy for each model
- Correlation analysis: bias entropy vs OOD performance

**Expected Results** (from mission ablation table):
- **Average OOD IG**: ~84.5% (vs 81.3% baseline, +3.2% improvement)
- **Bias entropy**: 6.7-6.9 nats (vs 5.9 baseline, +13-17% increase)
- **In-domain IG**: ~91.0-91.5% (vs 91.3% baseline, -0.3% degradation)
- **Correlation**: Pearson r ≥ 0.7 between entropy and OOD IG (p < 0.01)

**Success Criteria**:
- [ ] Average OOD IG improvement ≥2.5% (target 3.2%)
- [ ] Bias entropy increase ≥10% across all datasets
- [ ] In-domain degradation ≤1%
- [ ] Statistically significant improvements (t-test, p < 0.05)

**Deliverables**:
- Results table: baseline vs +entropy regularization
- Bias map visualizations showing increased entropy
- Correlation plot: entropy vs OOD IG
- Statistical significance tests

---

## Phase 3: Deconvolutional Bias Modeling (Weeks 5-6)

### Experiment 3.1: Implement Deconvolutional Bias Model

**Depends on**: Entropy regularization validation from 2.2

**Objective**: Implement explicit deconvolutional network to model implicit biases jointly with saliency training.

**Duration**: 5-6 days

**Architecture Design**:

1. **Deconvolution Network**:
   - Input: Learned tensor of shape [C, H_latent, W_latent] (e.g., C=64, H=8, W=8)
   - Architecture: 3-4 deconvolutional layers with upsampling
   - Output: Spatial bias map matching image resolution
   - Activation: Softmax to ensure valid probability distribution

2. **Training Strategy**:
   - Initialize input tensor randomly (Gaussian noise)
   - Joint optimization: train saliency model + deconv network + input tensor
   - Deconv network learns general bias structure
   - Input tensor captures dataset-specific bias

3. **Integration with Saliency Model**:
   - Saliency prediction: `p_saliency = saliency_model(image)`
   - Bias prediction: `p_bias = deconv_model(latent_tensor)`
   - Combined: `p_final = combine(p_saliency, p_bias)` (multiplicative or additive)
   - Loss on `p_final` vs ground truth

**Implementation Milestones**:
- [ ] Deconv architecture implemented and tested (forward pass works)
- [ ] Learnable input tensor initialized
- [ ] Integration with DeepGaze 2E complete
- [ ] Joint training pipeline functional
- [ ] Bias extraction validated (deconv output matches expected patterns)

**Validation**:
- Train on MIT1003 with entropy regularization + explicit bias model
- Extract implicit bias: (1) from uniform images, (2) from deconv network directly
- Compare extracted biases (should be similar)
- Measure in-domain IG

**Success Criteria**:
- [ ] Deconv model produces valid probability distributions (sums to 1)
- [ ] Extracted bias from deconv matches uniform-image extraction (correlation ≥0.8)
- [ ] Training is stable (both components converge)
- [ ] In-domain IG maintained ≥91%

**Expected Challenges**:
- **Challenge**: Deconv network might collapse (produce uniform output)
  - **Mitigation**: Add small L2 regularization on input tensor, monitor entropy
- **Challenge**: Training instability with two components
  - **Mitigation**: Pre-train saliency model, then add deconv model

**Deliverables**:
- Implemented deconv bias model
- Validation results showing bias extraction accuracy
- Training curves demonstrating stability

---

### Experiment 3.2: Explicit Bias Model - Cross-Dataset Evaluation

**Depends on**: Experiment 3.1 successful implementation

**Objective**: Evaluate whether explicit bias modeling further improves cross-dataset performance beyond entropy regularization alone.

**Duration**: 3-4 days

**Protocol**:
1. Train DeepGaze 2E with **entropy regularization + explicit bias model** on all datasets
2. Leave-one-out evaluation (5 folds)
3. Compare against:
   - Baseline (Exp 1.3)
   - Entropy regularization only (Exp 2.2)

**Evaluation**:
- Cross-dataset OOD IG
- In-domain IG
- Bias entropy (from deconv model output)

**Expected Results** (from mission ablation):
- **Average OOD IG**: ~85.8% (vs 84.5% with entropy only, +1.3% gain)
- **Component contribution**: Explicit modeling adds +1.3% over entropy regularization
- **In-domain IG**: ~91.0% (similar to entropy-only)

**Success Criteria**:
- [ ] OOD IG improvement ≥1% over entropy-only (target +1.3%)
- [ ] Improvement is consistent across datasets (≥3 out of 5)
- [ ] In-domain performance maintained
- [ ] Bias maps from deconv are interpretable (show spatial structure)

**Deliverables**:
- Results table: baseline → +entropy → +explicit bias
- Visualization of learned bias maps from deconv network
- Ablation analysis showing contribution of explicit modeling

---

## Phase 4: Few-Shot Bias Adaptation (Weeks 7-8)

### Experiment 4.1: Implement Few-Shot Adaptation Mechanism

**Depends on**: Experiment 3.2 validation of explicit bias model

**Objective**: Implement few-shot adaptation protocol that freezes deconv network and optimizes only input tensor using 100 OOD samples.

**Duration**: 4-5 days

**Implementation**:

1. **Adaptation Protocol**:
   - Take trained model from cross-dataset evaluation (trained on 4 datasets)
   - Freeze: All saliency model parameters + all deconv network parameters
   - Unfreeze: Only input tensor (4,096 dimensions for 64×8×8)
   - Dataset: 100 training samples from left-out dataset
   - Optimization: Adam, low learning rate (1e-3 to 1e-4)
   - Iterations: 50-100 epochs

2. **Adaptation Objective**:
   - Minimize task loss (saliency prediction) on 100 samples
   - Bias model adapts by updating input tensor only
   - No regularization needed (frozen network prevents overfitting)

3. **Evaluation**:
   - After adaptation, evaluate on left-out dataset's validation set
   - Compare: (1) no adaptation, (2) after adaptation
   - Measure: IG improvement, time to adapt

**Implementation Milestones**:
- [ ] Freezing mechanism implemented (verify gradients)
- [ ] Adaptation loop functional
- [ ] Convergence tracking works
- [ ] Evaluation pipeline integrated

**Validation on Single Fold**:
- Fold 1: Train on 4 datasets (leave MIT1003 out)
- Adapt: Use 100 MIT1003 training samples
- Evaluate: MIT1003 validation set
- Measure: IG before vs after adaptation

**Success Criteria**:
- [ ] Adaptation completes in <5 minutes on single GPU
- [ ] IG improvement ≥1.5% after adaptation (on validation set)
- [ ] Only input tensor updates (verify via gradient inspection)
- [ ] No overfitting to 100 samples (check training vs validation IG)

**Expected Results**:
- Pre-adaptation OOD IG: ~85.8% (from Exp 3.2)
- Post-adaptation OOD IG: ~87.6% (+1.8% gain from adaptation)
- Adaptation time: 2-3 minutes per dataset

**Deliverables**:
- Implemented adaptation pipeline
- Single-fold validation results
- Timing benchmarks
- Gradient verification showing only tensor updates

---

### Experiment 4.2: Few-Shot Adaptation - Full Evaluation

**Depends on**: Experiment 4.1 successful single-fold validation

**Objective**: Complete few-shot adaptation across all five folds and validate full framework performance.

**Duration**: 3-4 days

**Protocol**:
For each of 5 folds:
1. Train model with entropy regularization + explicit bias on 4 datasets
2. Evaluate on left-out validation set (no adaptation)
3. Adapt using 100 training samples from left-out dataset
4. Re-evaluate on left-out validation set (with adaptation)
5. Measure improvement

**Full Evaluation Matrix**:
- 5 folds × 2 conditions (pre-adaptation, post-adaptation)
- 3 random seeds per fold (for statistical validity)
- Total: 30 evaluation runs

**Expected Results** (from mission):

| Left-Out Dataset | Baseline IG | +Entropy | +Explicit Bias | +Adaptation (Final) | Total Gain |
|------------------|-------------|----------|----------------|---------------------|------------|
| MIT1003          | 84.2%       | 87.4%    | 88.5%          | 89.7%               | +5.5%      |
| CAT2000          | 82.1%       | 85.1%    | 86.2%          | 87.4%               | +5.3%      |
| COCO Freeview    | 85.6%       | 88.3%    | 89.1%          | 90.1%               | +4.5%      |
| Daemons          | 78.1%       | 81.8%    | 83.7%          | 86.0%               | +7.9%      |
| Figrim           | 76.4%       | 80.2%    | 82.4%          | 84.7%               | +8.3%      |
| **Average**      | **81.3%**   | **84.5%**| **85.8%**      | **87.6%**           | **+6.3%**  |

**Success Criteria**:
- [ ] Average OOD IG ≥87% (target 87.6%)
- [ ] Total improvement over baseline ≥5% (target 6.3%)
- [ ] Adaptation contributes ≥1.5% average gain (target 1.8%)
- [ ] Improvements statistically significant (p < 0.01 across 3 seeds)
- [ ] All 5 folds show improvement from adaptation

**Deliverables**:
- Complete results table with all ablations
- Statistical analysis (mean, std, significance tests)
- Adaptation contribution breakdown per fold
- Computational cost analysis (time, memory)

---

### Experiment 4.3: Sample Efficiency Analysis

**Depends on**: Experiment 4.2 completion

**Objective**: Analyze adaptation performance as a function of sample count to validate 100-sample choice.

**Duration**: 2-3 days

**Protocol**:
- Use one representative fold (e.g., MIT1003 left out)
- Vary adaptation sample count: {10, 25, 50, 100, 200, 500}
- Measure OOD IG after adaptation
- Plot: sample count vs performance

**Expected Results** (from mission):

| Sample Count | Average OOD IG | vs. No Adaptation | vs. 200 Samples |
|--------------|----------------|-------------------|-----------------|
| 0            | 85.8%          | baseline          | -2.0%           |
| 25           | 86.7%          | +0.9%             | -1.1%           |
| 50           | 87.3%          | +1.5%             | -0.5%           |
| 100          | 87.6%          | +1.8%             | -0.2%           |
| 200          | 87.8%          | +2.0%             | baseline        |

**Success Criteria**:
- [ ] 100 samples achieves ≥90% of 200-sample performance (target: 96.9%)
- [ ] Diminishing returns visible beyond 100 samples
- [ ] Results justify 100-sample choice for efficiency

**Deliverables**:
- Sample efficiency curve
- Statistical comparison of different sample counts
- Recommendation for optimal sample count

---

## Phase 5: Comprehensive Analysis & Validation (Weeks 9-10)

### Experiment 5.1: Bias Entropy Correlation Analysis

**Depends on**: Complete results from Experiment 4.2

**Objective**: Validate core hypothesis that increased bias entropy correlates with improved OOD generalization.

**Duration**: 2-3 days

**Analysis**:
1. **Extract Bias Maps**:
   - For all trained models (baseline, +entropy, +explicit, +adaptation)
   - Use uniform-colored image method
   - Measure entropy of extracted biases

2. **Correlation Study**:
   - X-axis: Extracted bias entropy (5.9-6.9 nats)
   - Y-axis: Cross-dataset OOD IG (76-90%)
   - Compute: Pearson correlation, Spearman correlation
   - Fit: Linear regression with confidence intervals

3. **Visualization**:
   - Scatter plot with trend line
   - Color-code by dataset
   - Show baseline vs regularized models

**Expected Results** (from mission):
- **Baseline entropy**: 5.92 ± 0.34 nats
- **ERSP-ABM entropy**: 6.84 ± 0.19 nats
- **Entropy increase**: +15.5%
- **Correlation**: Pearson r = 0.83 (p < 0.001)

**Success Criteria**:
- [ ] Correlation r ≥ 0.75 (target 0.83)
- [ ] Statistical significance p < 0.01
- [ ] Regularized models show consistently higher entropy
- [ ] Trend is consistent across all datasets

**Deliverables**:
- Correlation plot with statistical annotations
- Regression analysis report
- Entropy measurements table for all models
- Interpretation of correlation strength

---

### Experiment 5.2: In-Domain Performance Trade-off Analysis

**Depends on**: Experiment 4.2 complete results

**Objective**: Quantify the trade-off between in-domain performance and OOD generalization.

**Duration**: 2 days

**Analysis**:
Compare in-domain validation IG across all configurations:
- Baseline (no regularization)
- +Entropy regularization
- +Explicit bias model
- +Few-shot adaptation (adapted back to in-domain if applicable)

**Expected Results** (from mission):

| Dataset        | Baseline | ERSP-ABM | Difference |
|----------------|----------|----------|------------|
| MIT1003        | 93.1%    | 92.8%    | -0.3%      |
| CAT2000        | 91.4%    | 91.0%    | -0.4%      |
| COCO Freeview  | 92.7%    | 92.5%    | -0.2%      |
| Daemons        | 90.2%    | 89.9%    | -0.3%      |
| Figrim         | 88.9%    | 88.6%    | -0.3%      |
| **Average**    | **91.3%**| **91.0%**| **-0.3%**  |

**Success Criteria**:
- [ ] Average in-domain degradation ≤1% (target: 0.3%)
- [ ] Trade-off ratio: OOD gain / in-domain loss ≥ 10:1 (6.3% / 0.3% = 21:1)
- [ ] No catastrophic failures (no dataset degradation >2%)

**Deliverables**:
- In-domain performance table
- Trade-off ratio analysis
- Discussion of acceptable performance cost

---

### Experiment 5.3: Architecture Generalization - DeepGaze 3

**Depends on**: Successful DeepGaze 2E results (Exp 4.2)

**Objective**: Validate that ERSP-ABM framework generalizes to different saliency architectures.

**Duration**: 5-6 days

**Protocol**:
1. Implement ERSP-ABM framework for **DeepGaze 3** architecture
2. Train on same 5-fold leave-one-out protocol
3. Apply entropy regularization + explicit bias + adaptation
4. Compare with DeepGaze 3 baseline (no regularization)

**Training**:
- Use DeepGaze 3 architecture (different from 2E: includes fixation history modeling)
- For our purposes: saliency prediction only (ignore scanpath)
- Apply same regularization protocol
- Same datasets and splits

**Expected Results** (from mission):

| Architecture | Baseline Avg IG | ERSP-ABM Avg IG | Improvement |
|--------------|----------------|-----------------|-------------|
| DeepGaze 2E  | 81.3%          | 87.6%           | +6.3%       |
| DeepGaze 3   | 83.7%          | 89.4%           | +5.7%       |

**Success Criteria**:
- [ ] DeepGaze 3 shows consistent improvements (≥4%)
- [ ] Relative improvement within 20% of DeepGaze 2E (5.7% vs 6.3% ✓)
- [ ] Framework transfers without major modifications
- [ ] Training remains stable

**Deliverables**:
- DeepGaze 3 implementation with ERSP-ABM
- Cross-architecture comparison table
- Analysis of framework applicability

---

### Experiment 5.4: Ablation Studies - Component Contributions

**Depends on**: All main experiments complete

**Objective**: Systematically isolate the contribution of each framework component through comprehensive ablations.

**Duration**: 3-4 days

**Ablation Matrix**:

| Configuration | Components Included | Expected Avg OOD IG |
|---------------|-------------------|---------------------|
| Baseline | None | 81.3% |
| A1 | Entropy regularization only | 84.5% (+3.2%) |
| A2 | Explicit bias model only (no entropy reg) | 83.1% (+1.8%) |
| A3 | Entropy + Explicit bias | 85.8% (+4.5%) |
| A4 | Entropy + Adaptation (no explicit bias) | 86.2% (+4.9%) |
| A5 | Full framework (all three) | 87.6% (+6.3%) |

**Additional Ablations**:
1. **Regularization weight λ**: Vary {0.01, 0.05, 0.1, 0.5, 1.0}
2. **Adaptation samples**: {25, 50, 100, 200} (already in Exp 4.3)
3. **Uniform image count**: {0, 1, 2, 4} per batch
4. **Deconv architecture depth**: {2, 3, 4, 5} layers

**Success Criteria**:
- [ ] Each component contributes ≥1% (entropy: 3.2%, explicit: 1.3%, adaptation: 1.8%)
- [ ] Full framework > sum of individual components (synergistic effect)
- [ ] Ablations are statistically distinguishable

**Deliverables**:
- Comprehensive ablation table
- Component contribution breakdown
- Hyperparameter sensitivity analysis
- Recommendations for optimal configuration

---

### Experiment 5.5: Failure Analysis & Error Cases

**Depends on**: Complete OOD evaluation from Exp 4.2

**Objective**: Understand when and why the framework fails to improve performance or underperforms.

**Duration**: 2-3 days

**Analysis**:

1. **Identify Failure Cases**:
   - Images where OOD IG degrades with regularization
   - Datasets/categories with minimal improvement
   - Scenarios where baseline outperforms ERSP-ABM

2. **Categorize Failures**:
   - Content-driven failures: Images where spatial bias is informative (e.g., reading)
   - Dataset-specific: Unusual spatial patterns not captured by adaptation
   - Model limitations: Cases where saliency model itself is insufficient

3. **Quantitative Analysis**:
   - Failure rate: % of images with IG degradation
   - Average failure magnitude
   - Correlation with image properties (contrast, complexity, category)

4. **Qualitative Analysis**:
   - Visualize failure examples
   - Compare predictions: baseline vs ERSP-ABM
   - Analyze bias maps for failure cases

**Success Criteria**:
- [ ] Failure rate ≤20% of test images
- [ ] Failure cases have identifiable patterns
- [ ] Insights inform future improvements

**Deliverables**:
- Failure analysis report with examples
- Categorization of error types
- Recommendations for addressing limitations
- Discussion section content for paper

---

### Experiment 5.6: Computational Efficiency Benchmarking

**Depends on**: Final framework implementation

**Objective**: Quantify computational costs for practical deployment assessment.

**Duration**: 1-2 days

**Measurements**:

1. **Training Overhead**:
   - Baseline training time (no regularization)
   - Training time with entropy regularization
   - Memory usage (GPU)
   - Overhead percentage

2. **Adaptation Efficiency**:
   - Time to adapt on 100 samples
   - GPU memory during adaptation
   - Number of iterations to convergence

3. **Model Size**:
   - Baseline model parameters
   - Additional parameters from deconv network
   - Total model size (MB)

4. **Inference Cost**:
   - Inference time: baseline vs ERSP-ABM
   - Note: Should be identical (bias model not used at inference)

**Expected Results** (from mission):
- **Training overhead**: +8% (from one uniform image per batch)
- **Adaptation time**: 2.3 minutes per dataset (100 samples, single GPU)
- **Memory overhead**: +12 MB (deconv parameters)
- **Inference overhead**: 0% (no additional cost)

**Success Criteria**:
- [ ] Training overhead ≤15% (target: 8%)
- [ ] Adaptation time ≤5 minutes (target: 2.3 min)
- [ ] Memory overhead ≤50 MB (target: 12 MB)
- [ ] No inference overhead confirmed

**Deliverables**:
- Computational cost table
- Timing benchmarks with hardware specifications
- Memory profiling results
- Practical deployment recommendations

---

## Phase 6: Final Validation & Publication Prep (Week 11)

### Experiment 6.1: Multi-Seed Reproducibility Validation

**Depends on**: All experiments complete

**Objective**: Ensure results are reproducible across random seeds and generate final publication-ready results.

**Duration**: 3-4 days

**Protocol**:
1. Re-run key experiments with 5 different random seeds:
   - Baseline (5 seeds × 5 folds = 25 runs)
   - Full ERSP-ABM (5 seeds × 5 folds = 25 runs)

2. Generate statistics:
   - Mean and standard deviation per fold
   - Confidence intervals (95%)
   - Statistical significance tests (paired t-test)

3. Create publication-quality figures:
   - Main results table with error bars
   - Correlation plot with confidence intervals
   - Ablation bar charts with significance markers
   - Bias map visualizations (before/after)

**Success Criteria**:
- [ ] Standard deviation across seeds ≤2% for all folds
- [ ] Statistical significance maintained (p < 0.01) with multiple seeds
- [ ] Results align with hypothetical expectations (within error margins)

**Deliverables**:
- Final results with error bars and significance tests
- Complete figure set for paper
- Statistical analysis report
- Reproducibility documentation

---

### Experiment 6.2: Reproducibility Package Preparation

**Depends on**: Experiment 6.1 completion

**Objective**: Create comprehensive reproducibility package for publication and community release.

**Duration**: 2-3 days

**Package Contents**:

1. **Code Repository**:
   - Clean, documented codebase
   - Requirements.txt with dependencies
   - README with setup instructions
   - Training scripts with configuration files
   - Evaluation scripts
   - Visualization notebooks

2. **Model Checkpoints**:
   - Trained baseline models (5 datasets)
   - Trained ERSP-ABM models (5 folds)
   - Adapted models (5 folds × 100 samples)

3. **Data Preprocessing**:
   - Scripts to download and preprocess datasets
   - Empirical centerbias maps
   - Train/val split definitions

4. **Evaluation Tools**:
   - Information Gain calculation implementation
   - Bias extraction script (uniform images)
   - Entropy measurement tools

5. **Documentation**:
   - API documentation
   - Tutorial notebooks
   - Hyperparameter settings
   - Hardware requirements

**Success Criteria**:
- [ ] Fresh environment setup reproduces key results (within 2%)
- [ ] All dependencies clearly specified
- [ ] Code passes basic linting (ruff, black)
- [ ] Documentation complete and clear

**Deliverables**:
- Public GitHub repository (prepared for release)
- Pre-trained model checkpoints
- Complete documentation
- Tutorial notebooks demonstrating usage

---

### Experiment 6.3: Extended Analysis for Paper

**Depends on**: All experiments and analyses complete

**Objective**: Generate additional analyses and visualizations for comprehensive paper.

**Duration**: 2-3 days

**Additional Analyses**:

1. **Per-Category Performance**:
   - Break down OOD IG by image category (if available in datasets)
   - Identify which categories benefit most from regularization
   - Example: outdoor vs indoor, natural vs man-made

2. **Spatial Pattern Analysis**:
   - Cluster learned bias maps across datasets
   - Visualize common spatial patterns
   - Show how adaptation modifies patterns

3. **Convergence Analysis**:
   - Training curves showing entropy loss evolution
   - Adaptation convergence speed (iterations to plateau)
   - Stability analysis across seeds

4. **Qualitative Examples**:
   - Select representative success cases
   - Select representative failure cases
   - Side-by-side comparisons: baseline vs ERSP-ABM predictions

**Deliverables**:
- Extended results appendix
- Supplementary figures
- Qualitative analysis with examples
- Additional statistical tests

---

## Risk Mitigation & Contingency Plans

### High-Risk Elements & Mitigation Strategies

#### Risk 1: Triage Experiment Fails (Phase 0)
**Description**: Entropy regularization doesn't reduce implicit bias or destabilizes training.

**Likelihood**: Low-Medium (based on related work success)

**Impact**: High (blocks entire project)

**Mitigation**:
- Start with very low regularization weight (λ = 0.01)
- Try alternative entropy formulations (spatial vs global)
- Validate uniform image extraction independently first

**Fallback**:
- **Option A**: Try alternative regularization (e.g., KL divergence from uniform instead of entropy)
- **Option B**: Focus only on explicit bias modeling + adaptation (skip entropy regularization)
- **Option C**: Pivot to post-hoc bias correction approach (not ideal but salvages some contributions)

**Decision Point**: End of Week 1 (Day 3)

---

#### Risk 2: Baseline Reproduction Differs from Literature
**Description**: Can't achieve 90%+ IG on MIT1003 (published results show 93%).

**Likelihood**: Medium (implementation differences common)

**Impact**: Medium (affects comparison credibility)

**Mitigation**:
- Use official implementations if available
- Match hyperparameters exactly from papers
- Validate on multiple datasets (not just MIT1003)

**Fallback**:
- Document differences and proceed with achieved baseline
- Emphasize relative improvements rather than absolute numbers
- Contact original authors for implementation details

**Decision Point**: End of Week 2

---

#### Risk 3: Deconvolutional Bias Model Collapses
**Description**: Deconv network produces uniform output (fails to capture spatial structure).

**Likelihood**: Medium (common issue in generative models)

**Impact**: Medium (affects explicit modeling component)

**Mitigation**:
- Add diversity regularization to deconv output
- Use skip connections to preserve spatial information
- Initialize with empirical centerbias instead of random

**Fallback**:
- **Option A**: Simplify architecture (use fewer layers, larger latent space)
- **Option B**: Use alternative bias modeling (e.g., parametric centerbias)
- **Option C**: Skip explicit modeling, focus on entropy regularization + adaptation only (still has 2/3 components)

**Decision Point**: End of Week 5

---

#### Risk 4: Few-Shot Adaptation Overfits
**Description**: Adapting on 100 samples causes overfitting, validation IG decreases.

**Likelihood**: Low (frozen network prevents most overfitting)

**Impact**: Medium (affects adaptation component)

**Mitigation**:
- Reduce adaptation iterations (try 25, 50 instead of 100)
- Add early stopping based on validation performance
- Use stronger L2 regularization on input tensor

**Fallback**:
- **Option A**: Use more samples (200-500) if overfitting persists with 100
- **Option B**: Adapt deconv network instead of input tensor (more parameters = less overfitting risk)
- **Option C**: Use ensemble of adapted models (different initializations)

**Decision Point**: End of Week 7

---

#### Risk 5: Cross-Dataset Improvement Below Target
**Description**: OOD IG improvement <5% (below mission target).

**Likelihood**: Low-Medium

**Impact**: High (affects publication viability)

**Mitigation**:
- Tune regularization weight λ more carefully
- Try different combination strategies (multiplicative vs additive bias)
- Increase adaptation samples or iterations

**Fallback**:
- **Option A**: Focus on specific hard datasets (Figrim, Daemons) where improvements might be larger
- **Option B**: Emphasize bias entropy increase as contribution (methodological rather than empirical)
- **Option C**: Target different venue (workshop or domain-specific conference)
- **Option D**: Combine with architectural improvements (e.g., apply to SalNAS)

**Decision Point**: End of Week 8

---

#### Risk 6: Computational Resources Insufficient
**Description**: Training takes too long or requires more GPUs than available.

**Likelihood**: Medium

**Impact**: Medium (delays timeline)

**Mitigation**:
- Use smaller subset of datasets initially (3 instead of 5)
- Reduce number of random seeds for exploration (1-2 instead of 3-5)
- Parallelize across available GPUs

**Fallback**:
- **Option A**: Request additional compute allocation
- **Option B**: Use smaller models (DeepGaze 2E only, skip DeepGaze 3)
- **Option C**: Extend timeline by 2-4 weeks

**Decision Point**: Ongoing monitoring throughout project

---

### Timeline Buffer & Contingency

**Core Timeline**: 11 weeks (as outlined in phases)

**Buffer Periods**:
- **Week 12**: Buffer for delays, re-runs, unexpected issues
- **Week 13**: Final paper writing, figure polishing

**Critical Path**:
```
Phase 0 (Triage, Week 1) → Phase 1 (Baselines, Weeks 1-2) →
Phase 2 (Entropy Reg, Weeks 3-4) → Phase 3 (Deconv Model, Weeks 5-6) →
Phase 4 (Adaptation, Weeks 7-8) → Phase 5 (Analysis, Weeks 9-10) →
Phase 6 (Final Validation, Week 11)
```

**Parallel Opportunities**:
- Experiment 1.1 (data prep) can partially overlap with 0.1 (triage)
- Experiment 5.3 (DeepGaze 3) can run parallel to 5.1-5.2 (analysis)
- Experiment 6.2 (reproducibility) can start during 6.1 (validation)

---

## Dependencies Summary

### Visual Dependency Graph

```
Experiment 0.1 (Triage: Entropy Regularization Validation)
    ├─ GO Decision ─────────────────────┐
    │                                   │
    ↓                                   ↓
Experiment 1.1 (Data Preparation) → Experiment 1.2 (Baseline Reproduction)
                                        ↓
                                    Experiment 1.3 (Cross-Dataset Baseline)
                                        ↓
                                    Experiment 2.1 (Implement Entropy Reg)
                                        ↓
                                    Experiment 2.2 (Entropy Reg Full Eval)
                                        ↓
                                    Experiment 3.1 (Implement Deconv Bias)
                                        ↓
                                    Experiment 3.2 (Deconv Full Eval)
                                        ↓
                                    Experiment 4.1 (Implement Adaptation)
                                        ↓
                                    Experiment 4.2 (Adaptation Full Eval)
                                        ├───────────────┬──────────────┬────────────┐
                                        ↓               ↓              ↓            ↓
                                    Exp 4.3         Exp 5.1        Exp 5.2      Exp 5.3
                                    (Sample         (Entropy       (In-Domain   (DeepGaze 3)
                                     Efficiency)     Correlation)   Trade-off)
                                        ↓               ↓              ↓            ↓
                                        └───────────────┴──────────────┴────────────┘
                                                        ↓
                                                    Exp 5.4 (Ablations)
                                                        ↓
                                                    Exp 5.5 (Failure Analysis)
                                                        ↓
                                                    Exp 5.6 (Efficiency Benchmarks)
                                                        ↓
                                                    Exp 6.1 (Multi-Seed Validation)
                                                        ↓
                                                    Exp 6.2 (Reproducibility Package)
                                                        ↓
                                                    Exp 6.3 (Extended Analysis)
```

### Critical Path Experiments

**Must succeed for project viability**:
1. ✓ **Experiment 0.1**: Triage (validates core hypothesis)
2. ✓ **Experiment 1.3**: Baseline OOD performance (reference point)
3. ✓ **Experiment 2.2**: Entropy regularization improvement (≥2.5%)
4. ✓ **Experiment 4.2**: Full framework evaluation (≥5% total improvement)

**Important but non-critical**:
- Experiment 3.2 (explicit bias): Adds 1.3%, but framework viable without it
- Experiment 5.3 (DeepGaze 3): Validates generalization, but not required for core contribution
- Experiment 4.3, 5.1-5.6: Analysis and ablations (strengthen paper but not make-or-break)

---

## Success Metrics & Go/No-Go Criteria

### Overall Project Success

The project is successful if **ALL** of the following are achieved:

#### Primary Criteria (Mandatory)
1. ✓ **Triage Success**: Entropy regularization increases bias entropy ≥5% without destabilizing training
2. ✓ **Baseline Reproduction**: Achieve in-domain IG ≥88% on MIT1003 (within reasonable margin of published 93%)
3. ✓ **OOD Improvement**: Average cross-dataset IG improvement ≥5% absolute (target: 6.3%)
4. ✓ **In-Domain Trade-off**: In-domain IG degradation ≤1% (target: 0.3%)
5. ✓ **Statistical Significance**: Improvements are significant at p < 0.01 across multiple seeds

#### Secondary Criteria (Highly Desirable)
6. ✓ **Entropy Correlation**: Pearson r ≥ 0.70 between bias entropy and OOD IG (target: 0.83)
7. ✓ **Component Contributions**: Each component contributes ≥1% (entropy: 3.2%, explicit: 1.3%, adaptation: 1.8%)
8. ✓ **Computational Efficiency**: Training overhead ≤15%, adaptation time ≤5 minutes
9. ✓ **Architecture Generalization**: Framework works on DeepGaze 3 with ≥4% improvement

#### Tertiary Criteria (Nice to Have)
10. Sample efficiency: 100 samples achieves ≥90% of 200-sample performance
11. Reproducibility: Results within 2% across different random seeds
12. Failure rate: ≤20% of test images show degradation

### Publication Viability

**Top-tier venue (CVPR)**: Requires all primary + most secondary criteria

**Strong venue (ECCV, ICCV)**: Requires all primary + some secondary criteria

**Domain conference**: Requires primary criteria 1-4

---

## Computational Budget Estimation

### Hardware Requirements

**Recommended Setup**:
- **GPU**: NVIDIA A100 (40GB) or V100 (32GB)
- **Count**: 2-4 GPUs for parallel experiments
- **RAM**: 64GB system memory
- **Storage**: 500GB for datasets and checkpoints

**Minimum Setup**:
- **GPU**: NVIDIA RTX 3090 (24GB) or similar
- **Count**: 1 GPU (will extend timeline by ~2x)
- **RAM**: 32GB system memory
- **Storage**: 200GB

### Time Estimates per Experiment

| Experiment | GPU Hours | Wall-Clock Time | Parallelizable? |
|------------|-----------|-----------------|-----------------|
| 0.1 Triage | 8-12 | 8-12 hours | No (sequential) |
| 1.1 Data Prep | 4-8 | 4-8 hours | Partially |
| 1.2 Baseline (5 datasets) | 100-150 | 2-3 days | Yes (5-way) |
| 1.3 Cross-Dataset (5 folds) | 120-180 | 2-3 days | Yes (5-way) |
| 2.1 Entropy Reg Implementation | 20-30 | 1 day | No |
| 2.2 Entropy Reg Full (5 folds) | 120-180 | 2-3 days | Yes (5-way) |
| 3.1 Deconv Implementation | 30-40 | 1-2 days | No |
| 3.2 Deconv Full (5 folds) | 120-180 | 2-3 days | Yes (5-way) |
| 4.1 Adaptation Implementation | 10-15 | 1 day | No |
| 4.2 Adaptation Full (5 folds × 3 seeds) | 40-60 | 1-2 days | Yes (15-way) |
| 4.3 Sample Efficiency | 20-30 | 1 day | Yes (6-way) |
| 5.1 Entropy Correlation | 5-10 | <1 day | No (analysis) |
| 5.2 In-Domain Analysis | 5-10 | <1 day | No (analysis) |
| 5.3 DeepGaze 3 (5 folds) | 150-200 | 3-4 days | Yes (5-way) |
| 5.4 Ablations | 80-120 | 2-3 days | Yes (6-way) |
| 5.5 Failure Analysis | 10-20 | 1 day | No (analysis) |
| 5.6 Efficiency Benchmarks | 5-10 | <1 day | No |
| 6.1 Multi-Seed (5 seeds × 5 folds) | 200-300 | 4-5 days | Yes (25-way) |
| 6.2 Reproducibility | 20-30 | 1 day | No |
| 6.3 Extended Analysis | 10-20 | 1 day | No (analysis) |
| **Total** | **1,000-1,500** | **11-13 weeks** | With parallelization |

### Total Resource Requirements

**With 4 GPUs (Optimal)**:
- **Total GPU hours**: ~1,200 hours
- **Per GPU**: ~300 hours (12.5 days of continuous use)
- **Wall-clock time**: 11 weeks (as planned)
- **Cost estimate** (cloud): $3,000-$5,000 (@$2.50/GPU-hour for A100)

**With 2 GPUs (Feasible)**:
- **Total GPU hours**: ~1,200 hours
- **Per GPU**: ~600 hours (25 days of continuous use)
- **Wall-clock time**: 13-14 weeks (extended timeline)
- **Cost estimate** (cloud): $3,000-$5,000

**With 1 GPU (Minimum)**:
- **Total GPU hours**: ~1,500 hours (longer runs due to less efficient parallelization)
- **Wall-clock time**: 18-20 weeks (significantly extended)
- **Cost estimate** (cloud): $3,750-$5,000

### Resource Allocation Strategy

**Priority 1 (Critical Path)**: Allocate GPUs first
- Experiment 0.1, 1.3, 2.2, 4.2, 6.1

**Priority 2 (Important)**: Run when GPUs available
- Experiments 3.2, 5.3, 5.4

**Priority 3 (Analysis)**: CPU-intensive, can run in parallel on CPU
- Experiments 5.1, 5.2, 5.5, 6.3

**Optimization Strategy**:
- Run 5-fold experiments in parallel (requires 5 GPUs or sequential)
- Use smaller learning rates for overnight runs (more stable)
- Checkpoint frequently to recover from failures
- Use mixed precision training (FP16) to reduce memory and time

---

## Deliverables & Outputs

### Per-Phase Outputs

**Phase 0**: Triage validation report, go/no-go decision

**Phase 1**: Baseline performance tables, dataset statistics, preprocessed data

**Phase 2**: Entropy regularization implementation, bias entropy measurements, initial OOD improvements

**Phase 3**: Deconvolutional bias model implementation, explicit bias visualizations

**Phase 4**: Few-shot adaptation implementation, full framework results, sample efficiency analysis

**Phase 5**: Comprehensive ablations, correlation analysis, failure analysis, efficiency benchmarks, architecture generalization

**Phase 6**: Final results with statistical validation, reproducibility package, extended analyses

### Final Deliverables for Publication

1. **Main Paper Figures**:
   - Figure 1: Framework overview diagram
   - Figure 2: Main results table (5-fold OOD IG)
   - Figure 3: Bias entropy vs OOD IG correlation plot
   - Figure 4: Ablation study bar chart
   - Figure 5: Qualitative examples (bias maps, predictions)

2. **Supplementary Material**:
   - Extended results tables (all datasets, all configurations)
   - Additional ablations and analyses
   - Failure case visualizations
   - Architecture details
   - Hyperparameter settings

3. **Code Release**:
   - Clean, documented codebase
   - Pre-trained model checkpoints
   - Evaluation scripts
   - Tutorial notebooks
   - Requirements and setup instructions

4. **Data**:
   - Preprocessed datasets (or download scripts)
   - Empirical centerbias maps
   - Train/val splits
   - Extracted bias maps

---

## Conclusion

This roadmap provides a comprehensive, dependency-based experimental plan to validate the **ERSP-ABM framework** for reducing implicit spatial biases in saliency prediction. The phased approach ensures:

1. **Early validation** via minimum triage experiment (go/no-go at Day 3)
2. **Solid foundations** through baseline reproduction and cross-dataset evaluation
3. **Incremental development** of entropy regularization, explicit bias modeling, and few-shot adaptation
4. **Comprehensive validation** through ablations, correlations, and multi-seed experiments
5. **Publication readiness** with reproducibility packages and extended analyses

**Key Success Factors**:
- Triage experiment validates core hypothesis early (≥5% entropy increase)
- Each phase builds on validated results from previous phases
- Multiple fallback options for high-risk elements
- Realistic timeline (11 weeks) with buffer (2-3 weeks)
- Computational budget aligned with available resources

**Expected Outcome**: Demonstration that entropy regularization + explicit bias modeling + few-shot adaptation achieves **6.3% average cross-dataset IG improvement** with only **0.3% in-domain degradation**, validating the hypothesis that reduced implicit bias (measured by entropy) correlates strongly with improved OOD generalization (r=0.83, p<0.001).

**Next Steps**: Begin Phase 0 (Triage Experiment) immediately to validate core hypothesis and make go/no-go decision within 3 days.
