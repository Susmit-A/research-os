# Spec Requirements Document

> Spec: Experiment 0.1 - Core Hypothesis Validation: Entropy Regularization Effect
> Created: 2025-10-21

## Overview

Implement and validate the core hypothesis that Shannon entropy regularization applied to uniform images during training reduces implicit spatial bias in DeepGaze 3 saliency models and improves out-of-distribution generalization. This critical triage experiment (2-3 days) determines the go/no-go decision for the entire research project by comparing baseline DeepGaze 3 against entropy-regularized DeepGaze 3 on MIT1003 training data with CAT2000 out-of-distribution validation.

## User Stories

### Machine Learning Researcher Validating Novel Regularization

As a machine learning researcher, I want to quickly validate whether entropy regularization reduces implicit spatial bias in saliency models, so that I can make an informed decision about investing 11 weeks into the full research project before committing significant computational resources.

**Detailed Workflow:**
The researcher trains two DeepGaze 3 models side-by-side: a baseline without regularization and an entropy-regularized version (lambda=1.0). During training, uniform images are periodically fed through the model to extract spatial bias maps, and Shannon entropy is computed on these bias maps. The entropy-regularized model adds this entropy as a loss term (maximizing entropy to reduce spatial concentration). After 25 epochs, the researcher compares: (1) bias entropy increase, (2) in-domain MIT1003 performance, and (3) out-of-distribution CAT2000 performance. Success means entropy increased ≥5%, in-domain degradation ≤2%, and OOD performance shows improvement trends.

### Research Engineer Implementing Entropy Regularization

As a research engineer, I want to adapt the existing DeepGaze 3 codebase from Matthias Kümmerer to include entropy regularization on uniform images, so that the implementation maintains compatibility with the original architecture while adding the novel regularization component.

**Detailed Workflow:**
Starting from Matthias Kümmerer's DeepGaze 3 implementation, the engineer adds a uniform image generation module that creates images with constant pixel values. During training, after every N batches, the model processes these uniform images to extract predicted saliency maps (representing the model's spatial bias). Shannon entropy is computed on these bias maps, and the negative entropy is added as a regularization loss term weighted by lambda=1.0. The engineer ensures the baseline model trains identically except for the entropy regularization component, enabling fair comparison.

### Data Scientist Analyzing Cross-Dataset Generalization

As a data scientist, I want to measure Information Gain on both in-domain (MIT1003) and out-of-distribution (CAT2000) datasets, so that I can quantify whether entropy regularization improves generalization without sacrificing in-domain performance.

**Detailed Workflow:**
After training completes, the data scientist evaluates both models on held-out MIT1003 validation data (101 images) and CAT2000 out-of-distribution data (50 images). For each image, Information Gain is computed by comparing the model's predicted saliency map against ground-truth fixation maps, using a center-prior baseline. The scientist produces comparison tables showing: baseline vs. entropy-regularized performance on MIT1003 (expecting ≤2% degradation), baseline vs. entropy-regularized performance on CAT2000 (expecting any improvement), and bias map entropy measurements (expecting ≥5% increase). These metrics inform the go/no-go decision.

## Spec Scope

1. **DeepGaze 3 Baseline Implementation** - Adapt Matthias Kümmerer's DeepGaze 3 code to train on MIT1003 dataset (902 train, 101 validation) with standard hyperparameters (25 epochs, multistep LR scheduler 0.001585→1.5e-7).

2. **Entropy Regularization Component** - Implement Shannon entropy computation on uniform image bias maps and add entropy maximization loss term with lambda=1.0 weight during training.

3. **Uniform Image Bias Extraction** - Generate uniform images (constant pixel values) and extract predicted saliency maps to quantify spatial bias at regular training intervals.

4. **Dual Model Training Pipeline** - Train both baseline and entropy-regularized DeepGaze 3 models in parallel on 4x A100 GPUs for fair comparison.

5. **Cross-Dataset Evaluation Suite** - Compute Information Gain metrics on MIT1003 validation (in-domain) and CAT2000 (out-of-distribution) for both models, plus bias entropy measurements.

## Out of Scope

- Training on datasets other than MIT1003 (full cross-dataset experiments deferred to Phase 1)
- Hyperparameter tuning for lambda (deferred to Phase 2.1 - lambda sweep)
- DeepGaze 2E comparison (only DeepGaze 3)
- Explicit bias modeling with deconvolutional networks (deferred to Phase 1.2)
- Few-shot bias adaptation (deferred to Phase 3)
- Statistical significance testing across multiple runs (single run for triage)
- Experiment tracking with W&B or TensorBoard (optional, not required for go/no-go decision)

## Expected Deliverable

1. **Trained Model Checkpoints** - Two DeepGaze 3 model checkpoints (baseline and entropy-regularized) trained for 25 epochs on MIT1003 data, saved with optimizer states for potential continuation.

2. **Performance Comparison Report** - Quantitative results table showing: (a) MIT1003 validation Information Gain for both models, (b) CAT2000 out-of-distribution Information Gain for both models, (c) bias entropy measurements before/after training for both models, (d) training loss curves demonstrating convergence stability.

3. **Go/No-Go Decision Summary** - Clear assessment of whether success criteria are met: entropy increase ≥5%, in-domain degradation ≤2%, OOD improvement trend observed, training stability confirmed, enabling informed decision about proceeding to full research project.
