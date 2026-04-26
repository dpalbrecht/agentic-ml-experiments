# Experiment 001: Lasso Logistic Regression — Research

## Approach Researched
L1-regularized logistic regression with cross-validated C selection for binary classification on a high-noise feature set.

## Key Findings
- **Solver**: Use `liblinear` — the only solver supporting L1 for binary classification in scikit-learn; faster than `saga` on small datasets (~5K rows).
- **C grid**: Search `np.logspace(-4, 1, 20)` (1e-4 to 10). With ~48 noise features, stronger regularization (lower C) is expected to win.
- **Feature zeroing**: Read selected features via `np.where(model.coef_[0] != 0)`. Expect most noise features to zero out.
- **StandardScaler**: Still required — L1 penalizes raw coefficient magnitude, so unscaled features would dominate selection. Fit scaler on training folds only (Pipeline handles this correctly).
- **Nested CV**: `LogisticRegressionCV` does inner CV for C selection; wrapping in an outer `cross_validate` with `StratifiedKFold(10)` is safe — inner CV never sees the outer test fold.

## Sources
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html
- https://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic_l1_l2_sparsity.html
