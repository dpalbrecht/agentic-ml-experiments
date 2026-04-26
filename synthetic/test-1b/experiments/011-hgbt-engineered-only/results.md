# Experiment 011: HGBT on Engineered Features Only — Results

## Approach
HGBT (max_depth=5, lr=0.1, max_iter=2000, early stopping) on 9 features only: feature_0, feature_1, feature_2, f0_sq, f0_x_f2, f1_sq, f0sq_x_f1, f0sq_x_f2, f1sq_x_f2. No original noise features.

## Results
**Primary metric (Accuracy):** 0.8690 ± 0.0148
**Secondary metrics:**
- ROC-AUC: 0.8499 ± 0.0268
- Macro F1: 0.8300 ± 0.0205

## Validation Details
Stratified 10-fold CV:
| Fold | Accuracy | ROC-AUC | Macro F1 | Iterations |
|------|----------|---------|----------|------------|
| 1  | 0.9020 | 0.8910 | 0.8766 | 2000 (hit cap) |
| 2  | 0.8560 | 0.8371 | 0.8100 | 2000 (hit cap) |
| 3  | 0.8680 | 0.8120 | 0.8221 | 2000 (hit cap) |
| 4  | 0.8520 | 0.8335 | 0.8077 | 2000 (hit cap) |
| 5  | 0.8520 | 0.8162 | 0.8114 | 2000 (hit cap) |
| 6  | 0.8660 | 0.8450 | 0.8209 | 2000 (hit cap) |
| 7  | 0.8720 | 0.8759 | 0.8353 | 2000 (hit cap) |
| 8  | 0.8720 | 0.8361 | 0.8320 | 2000 (hit cap) |
| 9  | 0.8640 | 0.8654 | 0.8299 | 2000 (hit cap) |
| 10 | 0.8860 | 0.8864 | 0.8544 | 2000 (hit cap) |

## vs. Baseline
0.7186 → 0.8690 (+15.0 pp)

## vs. Best Previous
010-hgbt (0.8776) → 0.8690 (**-0.86 pp** — regression)

## vs. Success Threshold
90% required → 86.90% achieved → **NOT MET (gap: -3.10 pp)**

## Feature Importance
| Feature | Importance |
|---------|------------|
| f0_x_f2  | +0.1932 |
| f1_sq    | +0.1052 |
| feature_1 | +0.0093 |
| f1sq_x_f2 | +0.0046 |
| feature_2 | +0.0044 |
| f0sq_x_f2 | +0.0019 |
| f0_sq     | -0.0008 |
| feature_0 | -0.0013 |
| f0sq_x_f1 | -0.0030 |

## Observations
Removing the 47 original noise features caused a -0.86 pp regression — those features contribute marginally despite appearing as noise, and their removal hurt. The original 50 features + engineered terms (experiment 010) remains the best configuration at 87.76%. Across all experiments, early stopping has never once fired in 2000 iterations; the best mean accuracy achieved is 87.76%; and three independent feature/model approaches all point to ~88% as the practical ceiling. The 90% threshold appears to exceed what is achievable with this dataset through classical ML approaches: the two dominant features (f0·f2, f1²) carry the core signal, higher-order terms add diminishing returns, and the gap cannot be closed by noise removal, model family changes, or deeper polynomial expansion.

---
*Run on: 2026-04-25*
