# Related Work

## Core Saliency Prediction Models

### DeepGaze II (Kümmerer et al., 2016)
- **Paper**: "DeepGaze II: Reading fixations from deep features trained on object recognition"
- **arXiv**: 1610.01563
- **Year**: 2016
- **Key Approach**: Uses VGG-19 features pre-trained on ImageNet with readout layers for saliency prediction
- **Key Results**: 87% of explainable information gain on fixation patterns
- **Datasets**: MIT300 benchmark
- **Training Strategy**: No fine-tuning of base network, strong test of transfer learning
- **Relation to Project**: Baseline model for our experiments; we extend with entropy regularization

### DeepGaze IIE (DeepGaze 2E)
- **Year**: 2021
- **Key Improvement**: Enhanced calibration over DeepGaze II
- **Key Results**: 93% performance on MIT1003
- **Relation to Project**: One of three models we will train with entropy regularization and implicit bias modeling

### DeepGaze III
- **Key Feature**: Scanpath model with fixation history
- **Our Usage**: Saliency predictions only (not scanpath predictions)
- **Relation to Project**: Second model for testing our regularization approach

### UNISAL (2020)
- **Paper**: "Unified Image and Video Saliency Modeling"
- **Key Feature**: Domain-Adaptive Priors for unified image/video modeling
- **Our Usage**: Image saliency only (not video)
- **Relation to Project**: Third model for evaluation; UNISAL integration deferred to later work

---

## Cross-Dataset Saliency Prediction

### CaRDiff (Tang et al., 2024)
- **Paper**: "Video Salient Object Ranking Chain of Thought Reasoning for Saliency Prediction with Diffusion"
- **Venue**: AAAI 2025
- **Authors**: Yolo Yunlong Tang, Gen Zhan, Li Yang, Yiting Liao, Chenliang Xu
- **Approach**: Integrates multimodal large language model (MLLM), grounding module, and diffusion model
- **Key Feature**: Zero-shot evaluation capabilities across datasets
- **Datasets**: MVS and DHF1k
- **Relation to Project**: Achieves cross-dataset generalization through architectural improvements; our work adds explicit bias reduction mechanism

### SalNAS (Termritthikun et al., 2024)
- **Paper**: "Efficient Saliency-prediction Neural Architecture Search with self-knowledge distillation"
- **Authors**: Chakkrit Termritthikun, Ayaz Umer, Suwichaya Suwanwimolkul, Feng Xia, Ivan Lee
- **Year**: 2024
- **Approach**: Neural architecture search with self-knowledge distillation
- **Key Results**: Outperforms SOTA on most metrics across seven benchmark datasets
- **Relation to Project**: Demonstrates importance of cross-dataset evaluation; uses architectural search while we use regularization

### SalTR (Djilali et al., 2023)
- **Paper**: "Learning Saliency From Fixations"
- **Authors**: Yasser Abdelaziz Dahou Djilali, Kevin McGuiness, Noel O'Connor
- **Year**: 2023
- **Approach**: Transformer-based saliency model
- **Key Results**: Performance on par with SOTA on Salicon and MIT300 benchmarks
- **Relation to Project**: Strong baseline using transformer architecture; we focus on bias regularization applicable to any architecture

---

## Entropy Maximization & Regularization

### Deep OOD Uncertainty via Weight Entropy Maximization (de Mathelin et al., 2023)
- **Authors**: Antoine de Mathelin, François Deheeger, Mathilde Mougeot, Nicolas Vayatis
- **Year**: 2023
- **Approach**: Maximum entropy principle for weight distribution
- **Key Contribution**: Improves OOD detection through increased weight diversity
- **Relation to Project**: Similar motivation (OOD generalization) but applies entropy to weights; we apply it to output predictions

### CPR: Classifier-Projection Regularization (Cha et al., 2020)
- **Authors**: Sungmin Cha, Hsiang Hsu, Taebaek Hwang, Flavio P. Calmon, Taesup Moon
- **Year**: 2020
- **Approach**: Maximizes entropy of classifier output probability
- **Application**: Continual learning, reducing catastrophic forgetting
- **Relation to Project**: Directly relevant - uses output entropy maximization like our approach, but for continual learning rather than bias reduction

### Maximum Multiscale Entropy and Neural Network Regularization (Asadi & Abbe, 2020)
- **Authors**: Amir R. Asadi, Emmanuel Abbe
- **Year**: 2020
- **Approach**: Extends maximum entropy to multiscale settings
- **Key Result**: Demonstrates reduced excess risk compared to single-scale entropy
- **Relation to Project**: Theoretical foundation for entropy regularization benefits

### Feature Magnitude Regularization (Chapman et al., 2024)
- **Paper**: "Enhancing Fine-Grained Visual Recognition in the Low-Data Regime Through Feature Magnitude Regularization"
- **Authors**: Avraham Chapman, Haiming Xu, Lingqiao Liu
- **Year**: 2024
- **Approach**: Entropy maximization to ensure even distribution of feature magnitudes
- **Application**: Remove bias from pretrained models in low-data regime
- **Relation to Project**: Uses entropy to remove bias from pretrained models; we use it to prevent bias learning during training

---

## Few-Shot Adaptation with Frozen Networks

### MIV-head (Xu et al., 2025)
- **Paper**: "Few-shot Classification as Multi-instance Verification"
- **Authors**: Xin Xu, Eibe Frank, Geoffrey Holmes
- **Year**: 2025
- **Approach**: Backbone-agnostic classification head
- **Key Feature**: Cross-domain few-shot learning without fine-tuning backbones
- **Constraint**: Fine-tuning of backbones is impossible or infeasible
- **Relation to Project**: Similar to our approach of freezing deconv network and only training input tensor

### MMRL++ (Guo & Gu, 2025)
- **Paper**: "Parameter-Efficient Vision-Language Models"
- **Authors**: Yuncheng Guo, Xiaodong Gu
- **Year**: 2025
- **Approach**: Inserts learnable representation tokens in higher encoder layers
- **Strategy**: Keeps lower layers frozen to preserve pre-trained knowledge
- **Application**: Few-shot data scenarios
- **Relation to Project**: Demonstrates effectiveness of partial freezing; we freeze all except input tensor

### TAMT (Wang et al., 2025)
- **Paper**: "Temporal-Aware Model Tuning"
- **Authors**: Yilong Wang, Zilin Gao, Qilong Wang, et al.
- **Year**: 2025
- **Approach**: Local temporal-aware adapters to recalibrate intermediate features
- **Application**: Cross-domain action recognition
- **Key Feature**: Minimal parameter adaptation of frozen pre-trained models
- **Relation to Project**: Shows frozen networks with minimal adaptation work for cross-domain tasks

### Low-Precision Adapters (Jie et al., 2023)
- **Authors**: Shibo Jie, Haoqing Wang, Zhi-Hong Deng
- **Year**: 2023
- **Approach**: 1-bit quantization of adapters inserted into frozen models
- **Application**: Few-shot fine-grained visual categorization
- **Sample Regime**: Works with limited samples (100 or fewer)
- **Relation to Project**: Directly relevant - shows 100-sample regime is viable for adaptation

### Neural Fine-Tuning Search (Eustratiadis et al., 2023)
- **Authors**: Panagiotis Eustratiadis, Łukasz Dudziak, Da Li, Timothy Hospedales
- **Year**: 2023
- **Approach**: Neural architecture search to determine optimal adapter placement and layer freezing
- **Architectures**: ResNets and Vision Transformers
- **Relation to Project**: Principled approach to deciding what to freeze; we freeze everything except input tensor based on efficiency

---

## Deconvolutional Networks & Generative Models

### Sandwich GAN (Peng et al., 2024)
- **Paper**: "Learning to See Through Dazzle"
- **Authors**: Xiaopeng Peng, Erin F. Fleet, Abbie T. Watnik, Grover A. Swartzlander
- **Year**: 2024
- **Approach**: Wraps two GANs around learnable image deconvolution module
- **Application**: Image restoration addressing laser-induced degradations
- **Relation to Project**: Shows deconvolution as learnable module for image-space operations

### Transformers with Deconvolution (Bai et al., 2021)
- **Paper**: "Towards End-to-End Image Compression and Analysis with Transformers"
- **Authors**: Yuanchao Bai, Xu Yang, Xianming Liu, Junjun Jiang, Yaowei Wang, Xiangyang Ji, Wen Gao
- **Year**: 2021
- **Approach**: Deconvolutional neural network with compressed features and Transformer representations
- **Application**: Image reconstruction while maintaining classification performance
- **Relation to Project**: Demonstrates deconv networks for spatial reconstruction; we use for bias map generation

---

## Implicit Bias in Neural Networks (General)

### Implicit L1/L2 Transitions (Jacobs & Burkholz 2024)
- **Authors**: Jacobs, Burkholz
- **Year**: 2024
- **Finding**: Gradient descent induces implicit ℓ2 regularization that transitions to ℓ1
- **Domain**: General neural network training
- **Relation to Project**: Theoretical understanding of implicit biases; we address spatial implicit bias in saliency

### Implicit L1/L2 Transitions (Matt & Stöger 2025)
- **Authors**: Matt, Stöger
- **Year**: 2025
- **Finding**: Similar findings on implicit regularization transitions
- **Relation to Project**: Complementary theoretical work on implicit regularization

### Flatness Regularization (Gatmiry et al., 2023)
- **Authors**: Gatmiry et al.
- **Year**: 2023
- **Approach**: Minimizing Hessian trace
- **Benefit**: Improves generalization
- **Relation to Project**: Alternative regularization approach; we focus on output entropy

### Flatness Regularization (Fojtik et al., 2025)
- **Authors**: Fojtik et al.
- **Year**: 2025
- **Contribution**: Additional work on flatness-based regularization
- **Relation to Project**: Complementary approach to improving generalization

### Weight Normalization (Chou et al., 2023)
- **Authors**: Chou et al.
- **Year**: 2023
- **Approach**: Maintains sparse implicit bias at practical scales
- **Relation to Project**: Addresses implicit bias in parameter space; we address spatial bias in output space

---

## Evaluation Datasets

### MIT1003
- **Usage**: Standard benchmark for saliency prediction
- **Centerbias**: Has empirical centerbias used in DeepGaze training
- **Our Usage**: One of 5 datasets in leave-one-out evaluation

### MIT300
- **Usage**: Held-out benchmark for saliency models
- **Historical Results**: DeepGaze II achieved 87% explainable IG
- **Relation to Project**: Referenced for baseline performance comparisons

### CAT2000
- **Our Usage**: One of 5 datasets in leave-one-out evaluation
- **Contribution**: Diverse scene categories for testing generalization

### COCO Freeview
- **Our Usage**: One of 5 datasets in leave-one-out evaluation
- **Characteristic**: Free-viewing task on COCO images

### Daemons
- **Our Usage**: One of 5 datasets in leave-one-out evaluation

### Figrim
- **Our Usage**: One of 5 datasets in leave-one-out evaluation

---

## Key Metrics

### Information Gain (IG)
- **Definition**: Measures how much model predictions improve over baseline
- **Usage in Field**: Standard metric in saliency prediction community
- **Interpretation**: Higher IG = better prediction of human fixation patterns
- **Baseline**: Typically compared to center bias or uniform distribution
- **Our Usage**: Primary evaluation metric for comparing regularized vs baseline models

### Entropy
- **Our Usage**: Measure of implicit bias in extracted bias maps
- **Hypothesis**: Higher entropy (closer to uniform) = less implicit bias = better OOD generalization
- **Calculation**: Entropy of extracted bias map using uniform color images
- **Target**: Entropy approaching uniform distribution entropy

---

## Research Gaps & Novel Contributions

### Identified Gaps
1. **No prior work on explicit saliency bias regularization**: Existing work uses architectural improvements (SalNAS, CaRDiff) or domain-adaptive priors (UNISAL), but doesn't explicitly regularize against implicit bias learning
2. **Limited understanding of implicit bias in dense prediction**: Most implicit bias work focuses on classification; spatial biases in dense prediction tasks are underexplored
3. **No few-shot bias adaptation frameworks**: Existing few-shot work adapts task models; adapting bias models separately is novel

### Our Novel Contributions
1. **First entropy-based regularization for implicit bias reduction** in saliency prediction
2. **Explicit deconvolutional bias modeling** that separates content-driven saliency from dataset-specific biases
3. **Few-shot adaptation framework** specifically for bias calibration (not task adaptation)
4. **Comprehensive cross-dataset evaluation** with 5-fold leave-one-out protocol
5. **Novel implicit bias extraction method** using uniform color images

### Positioning in Literature
- **Builds on**: DeepGaze II/IIE architecture and training methodology
- **Extends**: Cross-dataset saliency work (SalNAS, CaRDiff) by adding explicit bias mechanism
- **Applies**: Entropy regularization principles (CPR, Chapman et al.) to spatial prediction task
- **Innovates**: Combines entropy regularization + explicit bias modeling + few-shot adaptation in unified framework
- **Differs from**: Prior work by explicitly modeling and regularizing implicit biases rather than ignoring or working around them

---

## Summary of Related Work Landscape

**Saliency Prediction State-of-the-Art**: DeepGaze models achieve 87-93% explainable IG; recent work (SalNAS, CaRDiff) improves cross-dataset performance through architectural innovations

**Cross-Dataset Challenges**: Limited work on explicit bias reduction; existing approaches use zero-shot capabilities or architecture search

**Entropy Regularization**: Successfully applied in continual learning (CPR), OOD detection (de Mathelin), and bias removal (Chapman), but not in saliency prediction

**Few-Shot Adaptation**: Frozen network approaches with minimal parameter updates (100 samples) shown effective in classification and action recognition

**Research Opportunity**: Combining entropy regularization + explicit bias modeling + few-shot adaptation represents unexplored approach with strong theoretical and empirical foundations from related domains

**Expected Impact**: First principled approach to reducing dataset-specific biases in saliency models; enables better cross-dataset deployment without architectural changes
