# Experiment 003: HGBT More Iterations — Results

## Approach
Identical to experiment 002 with one change: `max_iter` raised from 500 → 2000. All other settings unchanged (max_depth=7, learning_rate=0.1, min_samples_leaf=20, early stopping n_iter_no_change=10, validation_fraction=0.1).

## Results
**Primary metric (Accuracy):** 0.8436 ± 0.0109
**Secondary metrics:**
- ROC-AUC: 0.8324 ± 0.0270
- Macro F1: 0.7870 ± 0.0143

## Validation Details
Stratified 10-fold CV:
| Fold | Accuracy | ROC-AUC | Macro F1 | Iterations |
|------|----------|---------|----------|------------|
| 1  | 0.8520 | 0.8856 | 0.7983 | 2000 (hit cap) |
| 2  | 0.8240 | 0.8201 | 0.7588 | 2000 (hit cap) |
| 3  | 0.8440 | 0.7960 | 0.7874 | 2000 (hit cap) |
| 4  | 0.8360 | 0.8181 | 0.7802 | 2000 (hit cap) |
| 5  | 0.8460 | 0.8140 | 0.7930 | 2000 (hit cap) |
| 6  | 0.8360 | 0.8041 | 0.7790 | 2000 (hit cap) |
| 7  | 0.8360 | 0.8677 | 0.7777 | 2000 (hit cap) |
| 8  | 0.8420 | 0.8271 | 0.7803 | 2000 (hit cap) |
| 9  | 0.8620 | 0.8441 | 0.8081 | 2000 (hit cap) |
| 10 | 0.8580 | 0.8472 | 0.8070 | 2000 (hit cap) |

All 10 folds hit the 2000-iteration cap — early stopping still never triggered.

## vs. Baseline
0.7186 → 0.8436 (+12.5 pp)

## vs. Best Previous
002-hgbt (0.8414) → 0.8436 (**+0.22 pp** — negligible)

## vs. Success Threshold
90% required → 84.36% achieved → **NOT MET (gap: -5.64 pp)**

## Feature Importance
Top features by mean permutation importance (consistent with experiment 002):
| Feature | Importance |
|---------|------------|
| feature_0  | +0.1599 |
| feature_2  | +0.1526 |
| feature_1  | +0.0998 |
| feature_49 | +0.0009 |
| feature_47 | +0.0009 |

Only 10/50 features have positive importance. The top-3 (feature_0, feature_2, feature_1) dominate with 2-3 orders of magnitude more importance than the rest — identical pattern to experiment 002.

## Observations
Quadrupling iterations (500 → 2000) yielded only +0.22 pp — the model is clearly plateauing and more iterations alone won't reach 90%. The model continues to hit the cap without early stopping, suggesting it's finding infinitesimal improvements across many iterations by fitting noise. The consistent permutation importance across experiments 002 and 003 is a strong signal: feature_0, feature_2, and feature_1 are the true predictors, with everything else being noise. The next experiment should train HGBT on only those top-3 features — removing the 47 noise features may allow the model to converge on the true decision boundary without distraction.

---
*Run on: 2026-04-25*
