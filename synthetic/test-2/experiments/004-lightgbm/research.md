# Experiment 004: LightGBM — Research

## Approach Researched
LightGBM gradient boosted trees for imbalanced binary classification with temporal threshold tuning.

## Key Findings
- **`is_unbalance=True`** automatically adjusts for class imbalance by setting different initial prediction scores that account for class frequency. Equivalent in effect to `scale_pos_weight` but simpler to configure.
- **Probability calibration caveat**: `is_unbalance` produces poorly calibrated probability estimates. However, this doesn't matter for threshold tuning — we only need probabilities to be correctly rank-ordered (ordinal), not calibrated to true frequencies.
- **`importance_type='gain'`** reports total gain from splits using each feature, which is more informative than split count for understanding which features actually improve predictions.
- **No OOB predictions**: Unlike RF, GBMs build trees sequentially where each tree sees all training data residuals. Threshold tuning requires an explicit held-out validation set.
- **Precision-recall curve for threshold tuning**: Rather than sweeping a coarse grid (0.05–0.95 in 0.01 steps), use `precision_recall_curve` to evaluate F2 at every unique probability the model outputs. This is vectorized, faster, and finds the true optimum rather than the best grid point.
- **Default hyperparameters**: n_estimators=100, learning_rate=0.1, max_depth=-1 (no limit), num_leaves=31. These are reasonable starting points.
- **LGBMClassifier** follows the sklearn API (fit/predict/predict_proba), making it drop-in compatible.

## Sources
- [LightGBM Parameters Documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- [LGBMClassifier API](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)
- [is_unbalance vs scale_pos_weight — GitHub Issue #6807](https://github.com/microsoft/LightGBM/issues/6807)
