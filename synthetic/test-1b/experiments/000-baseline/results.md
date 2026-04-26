# Experiment 000: Baseline — Results

## Approach
Logistic regression with scikit-learn defaults (C=1.0, solver=lbfgs, L2 penalty) wrapped in a StandardScaler pipeline, trained on all 50 features.

## Results
**Primary metric (Accuracy):** 0.7186 ± 0.0054
**Secondary metrics:**
- ROC-AUC: 0.5990 ± 0.0224
- Macro F1: 0.4503 ± 0.0107

## Validation Details
Stratified 10-fold CV:
| Fold | Accuracy |
|------|----------|
| 1  | 0.7260 |
| 2  | 0.7200 |
| 3  | 0.7180 |
| 4  | 0.7160 |
| 5  | 0.7160 |
| 6  | 0.7140 |
| 7  | 0.7260 |
| 8  | 0.7180 |
| 9  | 0.7080 |
| 10 | 0.7240 |

## Feature Importance
Top 10 features by mean |coefficient| across folds:
| Feature | |coef| |
|---------|--------|
| feature_49 | 0.3015 |
| feature_48 | 0.2744 |
| feature_3  | 0.0781 |
| feature_6  | 0.0753 |
| feature_20 | 0.0668 |
| feature_23 | 0.0647 |
| feature_13 | 0.0479 |
| feature_10 | 0.0444 |
| feature_43 | 0.0437 |
| feature_19 | 0.0430 |

Importance is moderately spread (top-5 share: 40.3%). `feature_49` and `feature_48` dominate — consistent with the data assessment finding that those two carry nearly all signal. The remaining 48 noise features collectively dilute the model.

## vs. Success Threshold
**90% accuracy required → 71.86% achieved → NOT MET (gap: -18.1 pp)**

The baseline barely exceeds the 71.9% majority-class floor, confirming that logistic regression on all 50 features cannot discriminate effectively when ~48 features are noise.

---
*Run on: 2026-04-25*
