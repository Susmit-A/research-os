# Spec Summary (Lite)

Validate the core hypothesis that Shannon entropy regularization (lambda=1.0) applied to uniform images during DeepGaze 3 training reduces implicit spatial bias and improves out-of-distribution generalization. Train baseline and entropy-regularized DeepGaze 3 models on MIT1003 (902 train/101 val) for 25 epochs, then evaluate bias entropy increase (target ≥5%), in-domain MIT1003 Information Gain (acceptable degradation ≤2%), and OOD CAT2000 performance improvement to determine go/no-go for the full research project within 2-3 days.
