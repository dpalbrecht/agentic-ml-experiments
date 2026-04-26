# Experiment 007: Polynomial Features + Logistic Regression — Results

## Approach
Top-3 features (feature_0, feature_1, feature_2) → PolynomialFeatures(degree=2) → StandardScaler → LogisticRegressionCV(L2). 9 engineered features: 3 originals + 3 squares + 3 cross-products.

## Results
**Primary metric (Accuracy):** 0.8224 ± 0.0173
**Secondary metrics:**
- ROC-AUC: 0.8326 ± 0.0317
- Macro F1: 0.7391 ± 0.0295

## Validation Details
Stratified 10-fold CV:
| Fold | Accuracy | ROC-AUC | Macro F1 | Best C |
|------|----------|---------|----------|--------|
| 1  | 0.8460 | 0.8834 | 0.7778 | 2.976 |
| 2  | 0.8380 | 0.8041 | 0.7734 | 2.976 |
| 3  | 0.7940 | 0.7870 | 0.6885 | 1.438 |
| 4  | 0.8140 | 0.8111 | 0.7333 | 1.438 |
| 5  | 0.8000 | 0.7892 | 0.7047 | 0.336 |
| 6  | 0.8320 | 0.8431 | 0.7536 | 12.74 |
| 7  | 0.8200 | 0.8662 | 0.7306 | 1.438 |
| 8  | 0.8240 | 0.8324 | 0.7384 | 0.695 |
| 9  | 0.8100 | 0.8515 | 0.7147 | 54.56 |
| 10 | 0.8460 | 0.8580 | 0.7764 | 2.976 |

## vs. Baseline
0.7186 → 0.8224 (+10.4 pp) — massive improvement over raw LR, confirming the quadratic terms carry real signal.

## vs. Best Previous
005-hgbt (0.8440) → 0.8224 (**-2.2 pp**)

## vs. Success Threshold
90% required → 82.24% achieved → **NOT MET (gap: -7.76 pp)**

## Feature Importance
Mean |coefficient| across folds — all 9 features:
| Feature | |coef| |
|---------|--------|
| feature_0 × feature_2 | 1.3654 |
| feature_1²             | 1.0588 |
| feature_0²             | 0.2915 |
| feature_2²             | 0.1789 |
| feature_0 × feature_1  | 0.1525 |
| feature_1 × feature_2  | 0.0728 |
| feature_2              | 0.0309 |
| feature_1              | 0.0187 |
| feature_0              | 0.0112 |

The linear terms (feature_0, feature_1, feature_2) have near-zero coefficients — confirming no meaningful linear signal. The boundary is driven almost entirely by `feature_0 × feature_2` and `feature_1²`, with smaller contributions from the other squared terms.

## Observations
Polynomial LR reveals the boundary structure: the decision is governed by `feature_0 × feature_2` (cross-product) and `feature_1²` (quadratic) — both inherently non-linear, explaining why raw LR failed completely. The 2.2 pp gap below HGBT is expected: logistic regression on degree-2 features can only model a quadratic boundary, while HGBT approximates higher-order surfaces through splits. The natural next experiment is to explicitly add the two dominant engineered features (`feature_0 × feature_2` and `feature_1²`) to the original 50-feature set and run HGBT — giving the tree model the key non-linear combinations as direct input features rather than requiring it to reconstruct them from splits.

---
*Run on: 2026-04-25*
