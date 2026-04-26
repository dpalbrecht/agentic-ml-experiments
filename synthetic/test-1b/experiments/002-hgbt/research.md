# Experiment 002: Histogram Gradient Boosting — Research

## Approach Researched
HistGradientBoostingClassifier (scikit-learn) for binary classification on a high-noise feature set.

## Key Findings
- **No scaling needed**: HistGBT bins features internally (255 bins), so feature scale is irrelevant. Skip StandardScaler.
- **Starting hyperparameters**: `max_iter=500`, `max_depth=7`, `learning_rate=0.1`, `min_samples_leaf=20`. With 5K rows and 48 noise features, `min_samples_leaf=20` guards against overfitting on noise.
- **Early stopping**: `n_iter_no_change=10`, `validation_fraction=0.1`, `tol=1e-4` — prevents overfitting on noise features; actual tree count will be well under 500 in most cases.
- **Feature importance**: No built-in `.feature_importances_`. Use `permutation_importance()` on each held-out fold — more reliable for noisy data than impurity-based measures (which are biased toward high-cardinality features).
- **Class weight**: Skip `class_weight='balanced'` — our metric is accuracy and errors are symmetric; balancing would optimise for a different objective.

## Sources
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
- https://scikit-learn.org/stable/modules/permutation_importance.html
