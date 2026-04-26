# Experiment 001: Lasso Logistic Regression — Results

## Approach
L1-regularized logistic regression with cross-validated C selection (LogisticRegressionCV, penalty='l1', solver='liblinear', C grid: 1e-4 to 10 in 20 steps), wrapped in StandardScaler pipeline.

## Results
**Primary metric (Accuracy):** 0.7200 ± 0.0046
**Secondary metrics:**
- ROC-AUC: 0.6018 ± 0.0203
- Macro F1: 0.4430 ± 0.0102

## Validation Details
Stratified 10-fold CV:
| Fold | Accuracy | Best C | Non-zero features |
|------|----------|--------|-------------------|
| 1  | 0.7220 | 0.2637 | 48/50 |
| 2  | 0.7220 | 0.0785 | 40/50 |
| 3  | 0.7120 | 0.0785 | 39/50 |
| 4  | 0.7200 | 0.0785 | 37/50 |
| 5  | 0.7160 | 1.6238 | 50/50 |
| 6  | 0.7160 | 0.0785 | 38/50 |
| 7  | 0.7260 | 0.1438 | 47/50 |
| 8  | 0.7240 | 0.1438 | 42/50 |
| 9  | 0.7160 | 0.0428 | 23/50 |
| 10 | 0.7260 | 0.1438 | 45/50 |

Mean best C: 0.268 | Mean non-zero features: 40.9/50

## vs. Baseline
0.7186 → 0.7200 (+0.0014) — negligible improvement

## vs. Best Previous
000-baseline (0.7186) → 0.7200 (+0.0014)

## vs. Success Threshold
90% required → 72.00% achieved → **NOT MET (gap: -18.0 pp)**

## Feature Importance
Top 10 features by mean |coefficient|:
| Feature | |coef| |
|---------|--------|
| feature_49 | 0.2872 |
| feature_48 | 0.2599 |
| feature_3  | 0.0664 |
| feature_6  | 0.0630 |
| feature_20 | 0.0551 |
| feature_23 | 0.0515 |
| feature_13 | 0.0367 |
| feature_10 | 0.0333 |
| feature_19 | 0.0325 |
| feature_43 | 0.0317 |

Selected in all 10 folds (22 features): feature_48, feature_49, plus 20 others. L1 only zeroed ~9 features on average — far fewer than the ~48 expected noise features.

## Observations
L1 regularization provided almost no benefit over the L2 baseline (+0.14 pp). The sparse folds (Fold 9: 23 features) performed no better than the dense ones, which tells us noise dilution is not the main problem for this dataset. The linear signal in feature_48 and feature_49 (|r| ~0.12–0.13) appears too weak for any linear model to exploit effectively. This strongly suggests the target relationship is non-linear — the next experiment should try a tree-based model (e.g., gradient boosting or random forest) that can capture interaction effects and non-linear boundaries.

---
*Run on: 2026-04-25*
