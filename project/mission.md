# Research Mission: Entropy-Regularized Saliency Prediction with Adaptive Bias Modeling

## Project Title
**Reducing Implicit Spatial Biases in Saliency Prediction Through Entropy Regularization and Few-Shot Bias Adaptation**

## Core Mission

This research addresses the critical problem of implicit spatial biases in deep learning-based saliency prediction models. While state-of-the-art models like DeepGaze II (Kümmerer et al., 2016) and recent approaches such as SalNAS (Termritthikun et al., 2024) achieve strong in-domain performance (87-93% information gain), they suffer from poor cross-dataset generalization due to dataset-specific spatial biases learned during training. We develop a novel three-component framework combining entropy regularization during training, explicit deconvolutional bias modeling, and few-shot bias adaptation to enable robust out-of-domain generalization without architectural modifications.

## Target Venues

**Primary Target**:
- **CVPR 2026** (Computer Vision and Pattern Recognition) - Main track or Datasets & Benchmarks

**Secondary Targets**:
- **ICCV 2025** (International Conference on Computer Vision)
- **ECCV 2026** (European Conference on Computer Vision)
- **NeurIPS 2025** (if emphasizing the regularization and learning theory aspects)

**Rationale**: CVPR is ideal for this work as it combines novel regularization methodology with comprehensive empirical evaluation on standard saliency benchmarks. The explicit bias modeling and cross-dataset evaluation align well with CVPR's emphasis on practical improvements to core computer vision tasks.

## Mission Statement

**Opening - Problem Context**

This research addresses the persistent challenge of implicit spatial biases in saliency prediction models, which severely limit cross-dataset generalization. While prior work such as DeepGaze II (Kümmerer et al., 2016) achieves 87% information gain on MIT300 benchmarks and recent approaches like SalNAS (Termritthikun et al., 2024) and CaRDiff (Tang et al., 2024) improve cross-dataset performance through architectural innovations, significant limitations remain in explicitly addressing dataset-specific spatial biases that models inadvertently learn during training. Even when models are trained with explicit centerbias priors, they still acquire residual implicit biases that overfit to the spatial distribution patterns of training datasets, degrading performance when deployed on out-of-domain (OOD) data.

**Research Objective and Hypothesis**

Our primary objective is to enable robust cross-dataset generalization in saliency prediction by developing a principled framework for reducing implicit spatial biases learned during training. We hypothesize that entropy regularization on uninformative inputs during training, combined with explicit bias modeling and few-shot adaptation, will significantly reduce dataset-specific overfitting while maintaining competitive in-domain performance. This approach extends beyond current methods like SalNAS (architectural search) and CaRDiff (zero-shot MLLM integration) by explicitly modeling and regularizing the residual spatial biases that remain even when using external centerbias priors. This approach specifically targets the gap between in-domain and OOD performance that has not been adequately addressed by existing solutions focusing solely on architectural improvements or domain-adaptive priors (UNISAL, 2020).

**Methodology**

We propose **Entropy-Regularized Saliency Prediction with Adaptive Bias Modeling (ERSP-ABM)**, a three-component framework that operates during training, explicitly models residual biases, and enables rapid cross-dataset adaptation. Unlike existing approaches such as SalNAS that rely on architecture search or CaRDiff that requires multimodal large language models, our method introduces a simple yet effective training-time regularizer: we append uniform-colored images with uniform centerbias to each training batch and maximize the prediction entropy on these uninformative inputs, forcing the model to produce spatially uniform outputs when image content provides no information. The approach builds upon the established DeepGaze II/IIE architecture (Kümmerer et al., 2016) while introducing two novel components: (1) a deconvolutional implicit bias model with a learned input tensor that explicitly captures dataset-specific spatial priors, and (2) a few-shot adaptation mechanism that freezes the deconvolution network and optimizes only the input tensor using 100 samples from the target dataset. We evaluate our method on five standard saliency datasets (MIT1003, CAT2000, COCO Freeview, Daemons, Figrim) using Information Gain as the primary metric, following established benchmarking protocols in the saliency prediction community.

<hypothetical>
Our experiments on five-fold leave-one-out cross-validation across standard saliency benchmarks demonstrate substantial improvements in out-of-domain generalization while maintaining competitive in-domain performance. The proposed ERSP-ABM approach achieves an average cross-dataset Information Gain of 88.2% when adapted with 100 samples, surpassing the previous best baseline of 82.7% from unregularized DeepGaze IIE by 5.5 percentage points (6.6% relative improvement). On the most challenging cross-dataset scenarios (MIT1003→Figrim and CAT2000→Daemons), our method shows improvements of 8.3% and 7.9% absolute IG respectively, compared to baseline performance of 76.4% and 78.1%.

Analysis of the extracted implicit biases reveals that our entropy regularization successfully reduces dataset-specific spatial overfitting. The entropy of residual bias maps extracted from regularized models averages 6.84 nats, approaching the theoretical maximum of 7.12 nats for uniform distributions of the same dimensionality, compared to 5.92 nats for unregularized models. This 15.5% increase in entropy strongly correlates (Pearson r=0.83, p<0.001) with improved OOD performance, validating our hypothesis that reduced implicit bias enables better generalization.

Our ablation studies reveal that the entropy regularization component contributes 3.2 percentage points to the overall performance gain, while the few-shot bias adaptation adds 2.3 percentage points, demonstrating that both components are essential. The explicit deconvolutional bias modeling enables rapid adaptation: after fine-tuning only the input tensor (0.001% of total parameters) on 100 OOD samples for 50 iterations, the bias model adapts to the target dataset's spatial distribution with minimal overfitting risk.

In terms of computational efficiency, the training-time regularization adds only 8% overhead compared to standard training (from appending one uniform image per batch), while few-shot adaptation requires just 2.3 minutes on a single GPU for 100 samples. The approach also demonstrates robust performance across different model architectures: when applied to DeepGaze 3, we observe consistent improvements ranging from 4.8% to 6.9% across the five datasets, with an average gain of 5.7% in cross-dataset IG.

Importantly, our framework maintains competitive in-domain performance, achieving 92.8% IG on MIT1003 validation (compared to 93.1% for unregularized baseline), representing only a 0.3 percentage point reduction in exchange for substantial OOD improvements. This demonstrates that our regularization prevents overfitting to spatial biases without sacrificing the model's ability to learn content-driven saliency patterns.
</hypothetical>

**Contributions and Impact**

This work makes four primary contributions to computer vision and visual attention modeling: (1) We introduce the first entropy-based regularization approach specifically designed for reducing implicit spatial biases in dense prediction tasks, demonstrating that maximizing output entropy on uninformative inputs effectively prevents dataset-specific overfitting; (2) We demonstrate that explicit modeling of residual implicit biases using deconvolutional networks with learned input tensors enables principled separation of content-driven saliency from dataset-specific spatial priors; (3) We develop a parameter-efficient few-shot adaptation framework that calibrates bias models to new datasets using only 100 samples while freezing 99.999% of parameters, preventing catastrophic overfitting; and (4) We provide comprehensive empirical evidence through five-fold leave-one-out evaluation across standard benchmarks, establishing that our approach achieves superior cross-dataset generalization compared to architectural innovations alone.

**Broader Impact**

The implications of this research extend beyond saliency prediction to enable more reliable deployment of dense prediction models across diverse visual domains. By addressing the fundamental limitation of implicit spatial bias learning, our approach opens new possibilities for training computer vision models that better disentangle content-driven predictions from dataset-specific artifacts. This work represents a significant step toward domain-robust visual prediction systems, with potential applications in autonomous driving (handling varied camera positions), medical imaging (generalizing across different imaging protocols), and augmented reality (adapting to different viewing behaviors). The principle of entropy regularization on uninformative inputs is broadly applicable to other dense prediction tasks such as depth estimation, semantic segmentation, and optical flow, where spatial biases similarly degrade cross-dataset performance. Our explicit bias modeling framework also provides a diagnostic tool for analyzing what spatial priors models implicitly learn, enabling better understanding and debugging of deployed vision systems.

## Key Contributions

### Novel Methodological Contributions

1. **Entropy Regularization for Spatial Bias Reduction**
   - First application of output entropy maximization to reduce implicit spatial biases in dense prediction
   - Training-time regularizer that appends uniform-colored images to batches
   - Forces models to produce high-entropy (spatially uniform) predictions on uninformative inputs
   - Prevents learning of dataset-specific positional priors

2. **Deconvolutional Implicit Bias Modeling**
   - Explicit architectural component that separates spatial biases from content-driven predictions
   - Deconvolutional network with learned input tensor architecture
   - Jointly trained with saliency model to capture residual dataset-specific patterns
   - Enables post-training analysis and visualization of learned biases

3. **Parameter-Efficient Few-Shot Bias Adaptation**
   - Novel adaptation protocol using only 100 samples from target dataset
   - Freezes all deconvolution parameters, optimizes only input tensor (0.001% of parameters)
   - Enables rapid cross-dataset calibration without catastrophic overfitting
   - Applicable to resource-constrained deployment scenarios

4. **Comprehensive Cross-Dataset Evaluation Framework**
   - Five-fold leave-one-out protocol across standard saliency benchmarks
   - Systematic evaluation of bias entropy vs. OOD generalization correlation
   - Ablation studies isolating contribution of each framework component
   - Establishes new benchmark for cross-dataset saliency evaluation

### Technical Innovations

- **Uniform Image Extraction Method**: Novel technique for isolating and quantifying implicit spatial biases using uniform-colored images with uniform priors
- **Entropy-IG Correlation Analysis**: First systematic study linking bias map entropy to cross-dataset Information Gain
- **Minimal-Parameter Adaptation**: Demonstrates that adapting <0.01% of parameters is sufficient for effective bias calibration
- **Architecture-Agnostic Framework**: Applicable to DeepGaze 2E, DeepGaze 3, and extensible to other saliency architectures

### Empirical Contributions

- **Cross-Dataset Benchmark**: Comprehensive evaluation across MIT1003, CAT2000, COCO Freeview, Daemons, and Figrim
- **Baseline Comparisons**: Direct comparison against state-of-the-art including DeepGaze IIE, with and without regularization
- **Ablation Studies**: Systematic isolation of regularization, explicit modeling, and adaptation components
- **Computational Analysis**: Runtime and memory overhead quantification for practical deployment

## Hypothetical Results

<hypothetical>

### Cross-Dataset Generalization Performance

Our comprehensive five-fold leave-one-out evaluation demonstrates substantial improvements in out-of-domain (OOD) performance:

| Left-Out Dataset | Baseline IG | ERSP-ABM IG | Absolute Gain | Relative Gain |
|------------------|-------------|-------------|---------------|---------------|
| MIT1003          | 84.2%       | 89.7%       | +5.5%         | +6.5%         |
| CAT2000          | 82.1%       | 87.4%       | +5.3%         | +6.5%         |
| COCO Freeview    | 85.6%       | 90.1%       | +4.5%         | +5.3%         |
| Daemons          | 78.1%       | 86.0%       | +7.9%         | +10.1%        |
| Figrim           | 76.4%       | 84.7%       | +8.3%         | +10.9%        |
| **Average**      | **81.3%**   | **87.6%**   | **+6.3%**     | **+7.8%**     |

**Key Finding**: The proposed method achieves 87.6% average cross-dataset IG, representing a 6.3 percentage point absolute improvement over the 81.3% baseline, demonstrating robust generalization across diverse saliency datasets.

### Bias Entropy Analysis

Quantitative analysis of extracted implicit biases reveals strong correlation between entropy and OOD performance:

- **Baseline Model Entropy**: 5.92 ± 0.34 nats (across 5 datasets)
- **ERSP-ABM Entropy**: 6.84 ± 0.19 nats (across 5 datasets)
- **Theoretical Maximum**: 7.12 nats (uniform distribution)
- **Improvement**: +15.5% increase in entropy
- **Correlation**: Pearson r=0.83 (p<0.001) between bias entropy and OOD IG

**Interpretation**: Regularized models learn biases that are 15.5% closer to uniform distributions, validating our hypothesis that reduced implicit bias improves generalization.

### Ablation Study Results

Systematic ablation isolates the contribution of each component:

| Configuration | Average OOD IG | vs. Baseline | Component Contribution |
|---------------|----------------|--------------|------------------------|
| Baseline (no regularization) | 81.3% | - | - |
| + Entropy Regularization | 84.5% | +3.2% | 3.2% (regularization) |
| + Explicit Bias Model | 85.8% | +4.5% | 1.3% (explicit modeling) |
| + Few-Shot Adaptation (100 samples) | 87.6% | +6.3% | 1.8% (adaptation) |

**Key Insights**:
- Entropy regularization provides the largest single contribution (+3.2%)
- Explicit bias modeling adds +1.3% by separating spatial priors
- Few-shot adaptation contributes additional +1.8% through dataset-specific calibration
- All three components are essential for optimal performance

### In-Domain Performance Analysis

Our approach maintains competitive in-domain performance while improving OOD generalization:

| Dataset | Baseline In-Domain IG | ERSP-ABM In-Domain IG | Difference |
|---------|----------------------|----------------------|------------|
| MIT1003 | 93.1% | 92.8% | -0.3% |
| CAT2000 | 91.4% | 91.0% | -0.4% |
| COCO Freeview | 92.7% | 92.5% | -0.2% |
| Daemons | 90.2% | 89.9% | -0.3% |
| Figrim | 88.9% | 88.6% | -0.3% |
| **Average** | **91.3%** | **91.0%** | **-0.3%** |

**Trade-off Analysis**: Only 0.3 percentage points average reduction in in-domain performance in exchange for 6.3 percentage points improvement in cross-dataset scenarios—a highly favorable trade-off for practical deployment.

### Computational Efficiency

The framework adds minimal computational overhead:

- **Training Time Overhead**: +8% (from appending one uniform image per batch)
- **Few-Shot Adaptation Time**: 2.3 minutes per dataset (100 samples, 50 iterations, single GPU)
- **Memory Overhead**: +12 MB (deconvolutional bias model parameters)
- **Inference Time**: No overhead (bias model only used during training/adaptation)
- **Adaptation Parameters**: 0.001% of total model parameters (4,096-dimensional input tensor)

**Practical Viability**: The minimal computational cost makes this approach suitable for both research and production deployments.

### Architecture Generalization

When applied to DeepGaze 3 architecture (in addition to DeepGaze 2E):

| Architecture | Baseline Avg IG | ERSP-ABM Avg IG | Improvement |
|--------------|----------------|-----------------|-------------|
| DeepGaze 2E | 81.3% | 87.6% | +6.3% |
| DeepGaze 3 | 83.7% | 89.4% | +5.7% |

**Key Result**: Consistent improvements across different architectures (5.7-6.3%) demonstrate that our regularization framework is architecture-agnostic and broadly applicable.

### Few-Shot Sample Efficiency

Analysis of adaptation performance vs. number of samples:

| Sample Count | Average OOD IG | vs. Baseline |
|--------------|----------------|--------------|
| 0 (no adaptation) | 84.5% | +3.2% |
| 25 samples | 85.9% | +4.6% |
| 50 samples | 86.8% | +5.5% |
| 100 samples | 87.6% | +6.3% |
| 200 samples | 87.8% | +6.5% |

**Efficiency Finding**: Using 100 samples captures 96.9% of the performance gain achievable with 200 samples, demonstrating excellent sample efficiency.

</hypothetical>

## Significance to the Field

### Advancing Saliency Prediction Research

This work addresses a fundamental limitation in current saliency prediction models: their inability to generalize across datasets with different spatial viewing patterns. While recent advances like SalNAS and CaRDiff have improved cross-dataset performance through architectural innovations, they do not explicitly address the underlying cause of poor generalization—implicit spatial bias learning. Our framework provides the first principled approach to directly reducing these biases during training, complementing rather than replacing architectural improvements.

### Bridging Theory and Practice

The entropy regularization framework connects theoretical principles from maximum entropy learning (Asadi & Abbe, 2020) with practical saliency prediction challenges. By demonstrating that output entropy on uninformative inputs serves as a reliable proxy for generalization capability, we provide both a training objective and a diagnostic tool for assessing model robustness.

### Enabling Real-World Deployment

Cross-dataset generalization is critical for practical applications where training and deployment distributions differ. Our few-shot adaptation mechanism enables models trained on standard benchmarks (MIT1003, CAT2000) to be rapidly calibrated for specific applications (e.g., medical imaging, autonomous driving, AR/VR) using minimal domain-specific data, addressing a key barrier to real-world adoption of saliency prediction technology.

### Methodological Impact Beyond Saliency

The core insight—that regularizing model outputs on uninformative inputs prevents learning of dataset-specific artifacts—is broadly applicable to other dense prediction tasks. Similar implicit spatial biases likely affect depth estimation (camera-specific depth ranges), semantic segmentation (dataset-specific object layouts), and optical flow (camera motion patterns). Our framework provides a template for addressing these challenges across computer vision.

## Positioning Against State-of-the-Art

### Comparison with Architectural Approaches

**SalNAS (Termritthikun et al., 2024)** and **CaRDiff (Tang et al., 2024)** achieve cross-dataset improvements through neural architecture search and multimodal integration respectively. Our approach is **complementary**: entropy regularization can be combined with these architectures for potentially additive benefits. Unlike architecture search (which requires extensive computational resources) or MLLM integration (which adds significant model complexity), our regularization adds minimal overhead and can be applied to existing architectures.

### Comparison with Domain-Adaptive Priors

**UNISAL (2020)** uses domain-adaptive priors for unified image/video saliency. While UNISAL adapts priors based on input characteristics, it does not explicitly regularize against learning implicit biases during training. Our approach **prevents bias learning** at the source (during training) rather than compensating for it during inference, resulting in models that are fundamentally more robust to domain shift.

### Comparison with Entropy Regularization Literature

**CPR (Cha et al., 2020)** uses entropy maximization for continual learning, while **Chapman et al. (2024)** apply it to remove pretrained bias in fine-grained recognition. Our work **extends** these principles to dense spatial prediction with a novel insight: regularizing on uninformative inputs (uniform images) rather than training data itself. This targeted approach ensures entropy maximization specifically addresses spatial biases without interfering with content-driven learning.

### Comparison with Few-Shot Adaptation Methods

**MIV-head (Xu et al., 2025)**, **MMRL++ (Guo & Gu, 2025)**, and other few-shot methods adapt task models (classifiers, action recognizers) to new domains. Our approach is **fundamentally different**: we adapt a separate bias model while keeping the task model frozen. This architectural separation enables more principled bias calibration and prevents catastrophic forgetting of content-driven features.

### Novel Integration

The **unique contribution** is not any single component but their integration: combining training-time entropy regularization (prevents initial bias learning) + explicit deconvolutional modeling (separates biases from task predictions) + few-shot adaptation (calibrates to new distributions). This three-stage approach provides redundant protection against spatial bias while maintaining model performance, advancing beyond state-of-the-art in both methodology and empirical results.

---

## Success Metrics

### Primary Success Criteria

1. **Cross-Dataset IG Improvement**: Average OOD Information Gain ≥ +5% absolute improvement over baseline
2. **In-Domain Performance**: ≤ 1% degradation in in-domain IG compared to baseline
3. **Bias Entropy Increase**: ≥ 10% increase in extracted bias map entropy
4. **Statistical Significance**: p < 0.01 for OOD improvements across five-fold evaluation

### Secondary Success Criteria

1. **Sample Efficiency**: Achieve ≥90% of maximum benefit with ≤100 adaptation samples
2. **Computational Efficiency**: Training overhead ≤15%, adaptation time ≤5 minutes per dataset
3. **Architecture Generalization**: Consistent improvements (≥4%) across DeepGaze 2E and DeepGaze 3
4. **Component Validity**: Each component (regularization, bias model, adaptation) contributes ≥1% improvement

### Publication Readiness

- Comprehensive ablation studies isolating each component's contribution
- Statistical analysis with error bars and significance tests across five datasets
- Qualitative visualizations of extracted bias maps showing entropy differences
- Reproducibility materials including code, trained models, and evaluation protocols

---

**Status**: Ready for implementation and experimentation phase. All methodological components are well-defined with clear evaluation protocols and success criteria aligned with top-tier venue expectations.
