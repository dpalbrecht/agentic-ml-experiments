# Experiment 004: HGBT Top-3 Features — Results

## Approach
HGBT identical to 003 (max_iter=2000, max_depth=7, lr=0.1, min_samples_leaf=20, early stopping) trained on only top-3 features by permutation importance: feature_0, feature_2, feature_1.

## Results
**Primary metric (Accuracy):** 0.8396 ± 0.0179
**Secondary metrics:**
- ROC-AUC: 0.8381 ± 0.0268
- Macro F1: 0.7937 ± 0.0219

## Validation Details
Stratified 10-fold CV:
| Fold | Accuracy | ROC-AUC | Macro F1 | Iterations |
|------|----------|---------|----------|------------|
| 1  | 0.8580 | 0.8812 | 0.8186 | 2000 (hit cap) |
| 2  | 0.8360 | 0.8224 | 0.7869 | 2000 (hit cap) |
| 3  | 0.8380 | 0.8049 | 0.7890 | 2000 (hit cap) |
| 4  | 0.8100 | 0.8189 | 0.7573 | 2000 (hit cap) |
| 5  | 0.8180 | 0.7968 | 0.7728 | 2000 (hit cap) |
| 6  | 0.8360 | 0.8434 | 0.7847 | 2000 (hit cap) |
| 7  | 0.8560 | 0.8713 | 0.8138 | 2000 (hit cap) |
| 8  | 0.8360 | 0.8298 | 0.7847 | 2000 (hit cap) |
| 9  | 0.8340 | 0.8517 | 0.7937 | 2000 (hit cap) |
| 10 | 0.8740 | 0.8608 | 0.8359 | 2000 (hit cap) |

All 10 folds hit the cap — early stopping still never fired.

## vs. Baseline
0.7186 → 0.8396 (+12.1 pp)

## vs. Best Previous
003-hgbt (0.8436) → 0.8396 (**-0.40 pp** — marginal regression)

## vs. Success Threshold
90% required → 83.96% achieved → **NOT MET (gap: -6.04 pp)**

## Feature Importance
| Feature | Permutation Importance |
|---------|----------------------|
| feature_0 | +0.1768 |
| feature_2 | +0.1693 |
| feature_1 | +0.1163 |

All three features are clearly informative. Higher variance (±0.0179 vs ±0.0109 in 003) suggests less stable estimates with only 3 features.

## Observations
Restricting to top-3 features slightly hurt accuracy (-0.40 pp) and increased variance, suggesting the other 47 features provide a small amount of useful signal despite appearing as noise. More telling: early stopping still never fires even with just 3 features — the model keeps making microscopic improvements for 2000 iterations regardless of feature count. This strongly suggests the HGBT is overfitting its way to a sub-optimal boundary rather than converging cleanly. The consistent ~84% plateau across all four HGBT configurations points to a hyperparameter problem (likely max_depth=7 is too deep, allowing the model to keep splitting on noise) rather than a feature or iteration problem. The next experiment should try a proper hyperparameter search over max_depth and learning_rate.

---
*Run on: 2026-04-25*
