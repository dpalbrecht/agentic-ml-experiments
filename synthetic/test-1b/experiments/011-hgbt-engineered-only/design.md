# Experiment 011: HGBT on Engineered Features Only

## Approach
HGBT (max_depth=5, lr=0.1, max_iter=2000, early stopping) trained on only the top engineered features — no original noise features. Feature set: [f0_x_f2, f1_sq, f0_sq, f1sq_x_f2, f0sq_x_f2, f0sq_x_f1, feature_0, feature_1, feature_2] — the 9 features with positive permutation importance from experiment 010 (the 3 original top features + 6 polynomial terms that showed signal).

## Rationale
Experiments 008–010 show diminishing returns from adding more polynomial terms. The 50 original noise features are still present and the model hits the iteration cap in every fold, suggesting noise from the original features is preventing convergence on the clean polynomial boundary. Removing all 47 noise features may allow the model to focus exclusively on the true signal, analogous to how removing noise helped (slightly) in experiment 004 — but now with much more informative engineered features as the core set.

## What changed from experiment 010
- Dropped 57 features (47 original noise + 10 lowest-importance engineered terms)
- Kept only 9 features with confirmed positive permutation importance

## Evaluation
- **Primary metric**: Accuracy (mean ± std, outer 10-fold CV)
- **Secondary metrics**: ROC-AUC, Macro F1
- **Validation**: Stratified 10-fold cross-validation (same as all experiments)
