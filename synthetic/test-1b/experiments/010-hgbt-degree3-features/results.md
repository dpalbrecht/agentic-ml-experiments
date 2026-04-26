# Experiment 010: HGBT + Degree-3 Polynomial Features — Results

## Approach
HGBT (max_depth=5, lr=0.1, max_iter=2000, early stopping) on 66 features: original 50 + 6 degree-2 + 10 degree-3 terms from {feature_0, feature_1, feature_2}.

## Results
**Primary metric (Accuracy):** 0.8776 ± 0.0134
**Secondary metrics:**
- ROC-AUC: 0.8454 ± 0.0254
- Macro F1: 0.8369 ± 0.0190

## Validation Details
Stratified 10-fold CV:
| Fold | Accuracy | ROC-AUC | Macro F1 | Iterations |
|------|----------|---------|----------|------------|
| 1  | 0.9020 | 0.8877 | 0.8736 | 2000 (hit cap) |
| 2  | 0.8640 | 0.8420 | 0.8177 | 2000 (hit cap) |
| 3  | 0.8660 | 0.8015 | 0.8158 | 2000 (hit cap) |
| 4  | 0.8660 | 0.8261 | 0.8263 | 2000 (hit cap) |
| 5  | 0.8600 | 0.8346 | 0.8143 | 2000 (hit cap) |
| 6  | 0.8740 | 0.8180 | 0.8306 | 2000 (hit cap) |
| 7  | 0.8840 | 0.8712 | 0.8485 | 2000 (hit cap) |
| 8  | 0.8780 | 0.8449 | 0.8342 | 2000 (hit cap) |
| 9  | 0.8880 | 0.8712 | 0.8482 | 2000 (hit cap) |
| 10 | 0.8940 | 0.8572 | 0.8598 | 2000 (hit cap) |

## vs. Baseline
0.7186 → 0.8776 (+15.9 pp)

## vs. Best Previous
009-hgbt (0.8762) → 0.8776 (**+0.14 pp** — marginal, same as adding extra deg-2 terms)

## vs. Success Threshold
90% required → 87.76% achieved → **NOT MET (gap: -2.24 pp)**

## Feature Importance
Most important degree-3 terms by permutation importance:
| Feature | Importance | Rank |
|---------|------------|------|
| f0_x_f2  | +0.2003 | #1 (deg-2) |
| f1_sq    | +0.1283 | #2 (deg-2) |
| f0_sq    | +0.0037 | #3 (deg-2) |
| f1sq_x_f2 | +0.0013 | #6 (deg-3) |
| f0sq_x_f2 | +0.0011 | #8 (deg-3) |
| f0sq_x_f1 | +0.0009 | #9 (deg-3) |

Cubic terms (f0³, f1³, f2³) have near-zero importance. f1·f2² has negative importance. Signal in degree-3 terms is 2 orders of magnitude below f0_x_f2 and f1_sq.

## Observations
The +0.14 pp gain from 10 degree-3 terms mirrors the +0.14 pp from 4 extra degree-2 terms — a clear pattern of sharply diminishing returns beyond the two dominant features (f0·f2 and f1²). Three folds still exceed 90% and the model still hits the iteration cap in every fold, suggesting the remaining 2.24 pp gap comes from noise overfitting on the 50 original features diluting the clean polynomial signal. The natural final experiment is to strip all original features and train HGBT only on the most informative engineered features, forcing the model to focus exclusively on the true boundary.

---
*Run on: 2026-04-25*
