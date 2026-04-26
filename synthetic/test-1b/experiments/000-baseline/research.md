# Experiment 000: Baseline — Research

## Approach Researched
Logistic regression (scikit-learn defaults) for binary classification on a 50-feature standardized dataset.

## Key Findings
- **Scaling**: Wrap in a `Pipeline` with `StandardScaler` even though features are ~N(0,1) — ensures numerical stability and correct CV behavior (scaler fit only on train folds).
- **Solver**: `lbfgs` (scikit-learn default) is optimal for this dataset size (~5,000 rows). `liblinear` targets smaller data; `saga` is overkill here.
- **Regularization**: C=1.0 (default L2) is a reasonable starting point. The 48 noise features may benefit from lower C in future experiments.
- **Feature importance**: Use `np.abs(model.coef_[0])` on the standardized features — coefficients are directly comparable since all features have the same scale.
- **Pitfall**: Noise-heavy feature spaces inflate coefficient variance; baseline results may be weak but CV mitigates spurious folds.

## Sources
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- https://scikit-learn.org/stable/modules/linear_model.html
