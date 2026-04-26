# Experiment 008: HGBT + Engineered Features

## Approach
Append two explicit engineered features — `feature_0 × feature_2` and `feature_1²` — to the original 50-feature set (52 features total), then run HGBT with the best settings from experiment 005 (max_depth=5, learning_rate=0.1, max_iter=2000, min_samples_leaf=20, early stopping).

## Rationale
Experiment 007 revealed the dominant boundary terms are `feature_0 × feature_2` (coef 1.37) and `feature_1²` (coef 1.06) — the key non-linear structure the HGBT has been reconstructing through axis-aligned splits. By handing these terms to the model explicitly, we reduce the burden on the tree to discover them from scratch, potentially allowing it to learn the residual signal more efficiently and break through the consistent 84% ceiling.

## What changed from previous experiments
- Feature set: original 50 → 52 (original 50 + `f0_x_f2` + `f1_sq`)
- Model and hyperparameters: identical to best HGBT (005: max_depth=5, lr=0.1, max_iter=2000, early stopping)

## Evaluation
- **Primary metric**: Accuracy (mean ± std, outer 10-fold CV)
- **Secondary metrics**: ROC-AUC, Macro F1
- **Validation**: Stratified 10-fold cross-validation (same as all experiments)
