# Experiment 002: Histogram Gradient Boosting

## Approach
HistGradientBoostingClassifier with sensible defaults: `max_iter=500`, `max_depth=7`, `learning_rate=0.1`, `min_samples_leaf=20`, early stopping (`n_iter_no_change=10`, `validation_fraction=0.1`). No feature scaling. All 50 features. Outer evaluation via stratified 10-fold CV. Permutation importance computed on each held-out fold.

## Rationale
Both linear experiments (L2 and L1 logistic regression) plateaued at ~72% — the majority-class floor — despite different regularization strategies. The consistent failure suggests the target relationship is non-linear. HistGBT can capture non-linear decision boundaries and feature interactions that logistic regression fundamentally cannot. Early stopping and `min_samples_leaf=20` guard against overfitting on the ~48 noise features.

## What changed from previous experiments
- Model family: logistic regression → gradient boosted trees
- No StandardScaler (not needed for tree-based models)
- Feature importance: coefficient magnitudes → permutation importance

## Evaluation
- **Primary metric**: Accuracy (mean ± std across folds)
- **Secondary metrics**: ROC-AUC, Macro F1
- **Validation**: Stratified 10-fold cross-validation (same as all experiments)
