# Experiment 002: Histogram Gradient Boosting — Results

## Approach
HistGradientBoostingClassifier: max_iter=500, max_depth=7, learning_rate=0.1, min_samples_leaf=20, early stopping (n_iter_no_change=10, validation_fraction=0.1). No scaling. All 50 features.

## Results
**Primary metric (Accuracy):** 0.8414 ± 0.0099
**Secondary metrics:**
- ROC-AUC: 0.8325 ± 0.0269
- Macro F1: 0.7825 ± 0.0147

## Validation Details
Stratified 10-fold CV:
| Fold | Accuracy | ROC-AUC | Macro F1 | Iterations |
|------|----------|---------|----------|------------|
| 1  | 0.8540 | 0.8836 | 0.8027 | 500 |
| 2  | 0.8320 | 0.8139 | 0.7671 | 500 |
| 3  | 0.8380 | 0.7976 | 0.7773 | 500 |
| 4  | 0.8260 | 0.8174 | 0.7674 | 500 |
| 5  | 0.8440 | 0.8164 | 0.7897 | 500 |
| 6  | 0.8440 | 0.8078 | 0.7886 | 500 |
| 7  | 0.8380 | 0.8708 | 0.7786 | 500 |
| 8  | 0.8320 | 0.8231 | 0.7614 | 500 |
| 9  | 0.8460 | 0.8493 | 0.7833 | 500 |
| 10 | 0.8600 | 0.8448 | 0.8092 | 500 |

All folds hit the 500-iteration cap — early stopping never triggered.

## vs. Baseline
0.7186 → 0.8414 (**+12.3 pp**) — large, confirming the relationship is non-linear.

## vs. Best Previous
001-lasso-lr (0.7200) → 0.8414 (**+12.1 pp**)

## vs. Success Threshold
90% required → 84.14% achieved → **NOT MET (gap: -5.86 pp)**

## Feature Importance
Top features by mean permutation importance:
| Feature | Importance |
|---------|------------|
| feature_0  | +0.1580 |
| feature_2  | +0.1506 |
| feature_1  | +0.0980 |
| feature_49 | +0.0005 |
| feature_35 | +0.0004 |
| feature_26 | +0.0002 |

Only 7/50 features have positive permutation importance. Importance is highly concentrated in feature_0, feature_2, and feature_1 — notably different from the linear correlation leaders (feature_48, feature_49), confirming a non-linear signal structure.

## Observations
Switching to gradient boosting produced a massive +12.3 pp jump, confirming a non-linear decision boundary. Two important signals: (1) all folds hit the 500-iteration cap, suggesting the model hasn't converged and more iterations could help; (2) permutation importance reveals a different feature set (feature_0, feature_2, feature_1) than linear analysis found (feature_48, feature_49), likely reflecting interaction effects captured only by trees. The next experiment should increase max_iter (e.g., 1000–2000) since the model is still learning at 500.

---
*Run on: 2026-04-25*
