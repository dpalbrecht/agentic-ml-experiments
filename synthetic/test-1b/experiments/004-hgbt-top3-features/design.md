# Experiment 004: HGBT Top-3 Features

## Approach
Identical HGBT setup to experiment 003 (max_iter=2000, max_depth=7, learning_rate=0.1, min_samples_leaf=20, early stopping n_iter_no_change=10, validation_fraction=0.1) with one change: train on only the top-3 features by permutation importance from experiments 002/003 — `feature_0`, `feature_2`, `feature_1`.

## Rationale
Both HGBT experiments showed the same permutation importance pattern: feature_0, feature_2, feature_1 are 2–3 orders of magnitude more important than any other feature. The model has been fitting 47 noise features across 2000 iterations without early stopping firing, suggesting noise is preventing convergence on the true decision boundary. Restricting to top-3 eliminates noise and forces the model to learn only from informative features.

## What changed from previous experiments
- Feature set: all 50 → top-3 (feature_0, feature_2, feature_1)
- Everything else identical to experiment 003

## Evaluation
- **Primary metric**: Accuracy (mean ± std across folds)
- **Secondary metrics**: ROC-AUC, Macro F1
- **Validation**: Stratified 10-fold cross-validation (same as all experiments)
