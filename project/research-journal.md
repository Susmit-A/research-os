# Research Planning Journal

## Initial Research Idea

The user wants to study implicit bias in saliency models, particularly how models like DeepGaze 2E and DeepGaze 3 learn residual centerbiases even when trained with an explicit centerbias. The main contribution will be developing a regularizer that reduces this implicit bias to improve out-of-domain (OOD) generalization.

---

## Iteration 1: Initial Exploration

### Notebook Analysis

The uploaded notebook `residual_centerbias.ipynb` demonstrates:
- Models trained with external centerbias (MIT1003) still learn implicit residual bias
- Test method: Feed uniform color images with uniform prior, extract learned bias
- Key finding: More powerful models (DG3) learn more implicit bias than simpler models (DG2E)
- Entropy analysis: Compares entropy of residual bias to uniform distribution entropy
- Initial regularization attempts show promise in reducing implicit bias
- Code shows extraction method: average predictions over 8 uniform color images (0-255 intensity)
- Calculates KL divergence from uniform distribution as bias measure

### Related Work Search 1
- Search queries: ["DeepGaze saliency model", "Unisal unified saliency", "saliency bias decomposition", "center bias visual attention"]
- Key findings:
  - DeepGaze II (2016): Uses VGG-19 features, 87% information gain
  - DeepGaze IIE (2021): Enhanced calibration, 93% on MIT1003
  - UNISAL (2020): Unified image/video model with Domain-Adaptive Priors
  - DeepGaze III: Scanpath model with fixation history

### Questions Asked

**Q1:** Which specific model versions will you focus on?
**A1:** DeepGaze 2E (also written as DeepGaze IIE), DeepGaze 3 (saliency predictions only, not scanpath predictions), and UNISAL (image saliency only)

**Q2:** What's your dataset strategy for evaluation?
**A2:** Both in-domain and out-of-domain (OOD) datasets. Study how implicit bias affects OOD performance. Approach: Provide OOD image with OOD dataset's empirical centerbias, study how performance changes with implicit bias from training set vs implicit bias from OOD set

**Q3:** What do you mean by "monochrome images"?
**A3:** Uniform colored images

**Q4:** What do you mean by "uniform centerbias"?
**A4:** Replace MIT1003 center bias with uniform distribution of same size, where each value = 1/map.numel()

**Q5:** What evaluation metrics will you use?
**A5:** Primary metric: Information Gain (IG). Key analysis: Entropy of the extracted implicit bias. Hypothesis: Entropy closer to uniform density entropy = less information in implicit bias = less overfitting to training data = better OOD generalization

**Q6:** Will UNISAL be tested on video or just images?
**A6:** Only on saliency images (not video)

**Q7:** What's the main contribution beyond the decomposition analysis?
**A7:** **A regularizer that reduces the model's implicit bias**. Demonstrate its effect on OOD performance. This is the key novel contribution, not just the three-way decomposition analysis

**Q8:** What practical applications do you envision?
**A8:** Better cross-dataset generalization, improved model calibration, new training strategies for saliency models

**Q9:** Do you have existing code or experiments?
**A9:** Yes, uploaded residual_centerbias.ipynb with basic implementation. Should read this file to understand their current approach

### Insights Gained
- **Focus is on developing a regularization technique**, not just analysis
- The regularizer aims to reduce implicit bias learned during training
- Success metric: Higher entropy in implicit bias (closer to uniform) correlates with better OOD performance
- Three models for comprehensive evaluation: DeepGaze 2E, DeepGaze 3, UNISAL
- Novel extraction method using uniform color images to isolate implicit bias
- Key hypothesis: Models with less implicit bias will generalize better to OOD datasets
- Evaluation will compare performance with training set's implicit bias vs OOD dataset's implicit bias

---

## Iteration 2: Focused Investigation

### Related Work Search 2
- Search queries: ["implicit bias neural networks regularization", "saliency prediction cross dataset", "deepgaze centerbias"]
- Key findings:

**Implicit Bias & Regularization Literature:**
- **Implicit L1/L2 transitions**: Several papers show gradient descent induces implicit ℓ2 regularization that transitions to ℓ1 (Jacobs & Burkholz 2024, Matt & Stöger 2025)
- **Flatness regularization**: Minimizing Hessian trace improves generalization (Gatmiry et al. 2023, Fojtik et al. 2025)
- **Weight normalization**: Maintains sparse implicit bias at practical scales (Chou et al. 2023)
- **No direct work on saliency bias regularization found** - gap in literature

**Cross-Dataset Saliency:**
- SalNAS (2024): Self-knowledge distillation for better cross-dataset performance
- CaRDiff (2024): Zero-shot evaluation capabilities across datasets
- Limited work specifically on bias reduction for cross-dataset saliency

---

## Iteration 3: Corrected Understanding

### CRITICAL CORRECTION from User

**Previous misunderstanding:** Thought post-hoc meant predictions would be corrected after inference
**CORRECT understanding:**
1. **Post-hoc is for EVALUATION only**: The implicit bias extraction using uniform images is for analysis and evaluation purposes
2. **Correction happens during TRAINING**: The implicit bias will be corrected using an additional regularizer during training (not post-hoc)
3. **Additional component**: They will train an "implicit bias model" that predicts the implicit bias from the training set, then adapt this model using few samples on the OOD set

### Corrected Answers

**Q1. Correction approach:**
Predictions are NOT corrected post-hoc. This is only for evaluation. The implicit bias will be corrected using an additional regularizer during training.

**Q2. Correction formula:**
Not yet defined (need to ask about training regularizer formulation)

**Q3. OOD Datasets:**
Leave-one-out validation among:
- MIT1003
- CAT2000
- COCO Freeview
- Daemons
- Figrim

**Q4. Calibration validation:**
All three approaches:
- Validation split from training domain
- Cross-validation on multiple OOD datasets
- Optimize for max entropy while maintaining min performance

**Q5. Baseline comparisons:**
Original uncorrected models in leave-one-out OOD setting

**Q6. UNISAL integration:**
User asks for proposal. One option they suggest: Compare OOD performance with validation set centerbias and validation set-adapter implicit bias to compare with adapted UNISAL

**Q7. Computational efficiency:**
Not relevant since correction is not post-hoc

**Q8. Implicit bias adaptation:**
They will be training an "implicit bias model" that is trained on the training set to predict its implicit bias, and adapt this using few samples on the OOD set

### Key Research Components (Corrected)
1. **Training-time regularizer** to reduce implicit bias in saliency models
2. **Implicit bias extraction method** using uniform images (for evaluation/analysis)
3. **Implicit bias prediction model** that can be adapted with few-shot learning on OOD datasets
4. **Leave-one-out evaluation** across 5 datasets (MIT1003, CAT2000, COCO Freeview, Daemons, Figrim)

---

## Iteration 4: Targeted Clarification

### Follow-up Questions on Corrected Understanding

Based on the corrected understanding that this involves:
- Training-time regularization (not post-hoc correction)
- An implicit bias prediction model with few-shot adaptation
- Evaluation-only bias extraction

I need to clarify the following:

**Q1:** For the training-time regularizer, what formulation are you considering?
- Option A: Penalize KL divergence between model predictions on uniform images and uniform distribution
- Option B: Maximize entropy of predictions on uniform images
- Option C: Add auxiliary loss that predicts uniform output for uninformative inputs
- Option D: Something else?

**Q2:** For the "implicit bias prediction model" - can you describe its architecture and training?
- What are the inputs? (image features, dataset indicators, something else?)
- What does it predict? (per-pixel bias map, global bias parameters, bias correction factors?)
- How is it trained? (supervised on extracted biases, self-supervised, adversarial?)

**Q3:** The few-shot adaptation of the bias model - how exactly does this work?
- Do you fine-tune the bias model on a few OOD samples?
- Use meta-learning approaches like MAML?
- Or prompt/condition the model with OOD examples?

**Q4:** The relationship between these components:
- Does the regularizer use the implicit bias model during training?
- Or are they separate: regularizer reduces bias during training, bias model predicts remaining bias?
- How do they interact in the final system?

**Q5:** For training with the regularizer:
- Will you retrain DeepGaze 2E/3 from scratch with your regularizer?
- Or fine-tune existing checkpoints?
- For UNISAL, will you modify their training code or work with their pre-trained model?

**Q6:** Evaluation protocol clarification:
- In leave-one-out: train on 4 datasets, test on 5th
- Do you train one model per leave-out scenario (5 models total)?
- Or train once on all 4 and evaluate multiple times?

**Q7:** For UNISAL integration, here's a concrete proposal:
Compare three scenarios on OOD data:
a) UNISAL with its original domain-adaptive priors
b) UNISAL with your implicit bias model predictions as priors
c) Your regularized models with adapted implicit bias
Would this comparison make sense for your goals?

**Q8:** Success metrics beyond entropy:
- Minimum acceptable in-domain performance drop?
- Target OOD improvement percentage?
- How to balance bias reduction vs task performance?

### Areas Requiring Literature Search
1. Training-time regularization for bias reduction in dense prediction
2. Few-shot adaptation methods for bias models
3. Meta-learning for domain adaptation
4. Implicit bias modeling in neural networks