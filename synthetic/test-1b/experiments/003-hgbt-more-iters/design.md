# Experiment 003: HGBT More Iterations

## Approach
Identical to experiment 002 (HistGradientBoostingClassifier, max_depth=7, learning_rate=0.1, min_samples_leaf=20, early stopping n_iter_no_change=10, validation_fraction=0.1) with a single change: `max_iter` raised from 500 → 2000.

## Rationale
Every fold in experiment 002 hit the 500-iteration cap — early stopping never triggered. This means the model was still actively improving at iteration 500 and was cut off prematurely. Raising the cap to 2000 allows early stopping to fire naturally when improvement genuinely stalls, giving the model room to converge.

## What changed from previous experiments
- `max_iter`: 500 → 2000 (everything else identical to 002)

## Evaluation
- **Primary metric**: Accuracy (mean ± std across folds)
- **Secondary metrics**: ROC-AUC, Macro F1
- **Validation**: Stratified 10-fold cross-validation (same as all experiments)
