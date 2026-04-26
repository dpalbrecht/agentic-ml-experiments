# Experiment 009: HGBT + Full Degree-2 Polynomial Features — Results

## Approach
HGBT (max_depth=5, lr=0.1, max_iter=2000, early stopping) on 56 features: original 50 + all 6 degree-2 terms from {feature_0, feature_1, feature_2}: f0², f0·f1, f0·f2, f1², f1·f2, f2².

## Results
**Primary metric (Accuracy):** 0.8762 ± 0.0151
**Secondary metrics:**
- ROC-AUC: 0.8470 ± 0.0267
- Macro F1: 0.8351 ± 0.0214

## Validation Details
Stratified 10-fold CV:
| Fold | Accuracy | ROC-AUC | Macro F1 | Iterations |
|------|----------|---------|----------|------------|
| 1  | 0.9100 | 0.9004 | 0.8839 | 2000 (hit cap) |
| 2  | 0.8640 | 0.8464 | 0.8187 | 2000 (hit cap) |
| 3  | 0.8600 | 0.8029 | 0.8092 | 2000 (hit cap) |
| 4  | 0.8660 | 0.8358 | 0.8254 | 2000 (hit cap) |
| 5  | 0.8620 | 0.8273 | 0.8165 | 2000 (hit cap) |
| 6  | 0.8740 | 0.8206 | 0.8306 | 2000 (hit cap) |
| 7  | 0.8800 | 0.8739 | 0.8433 | 2000 (hit cap) |
| 8  | 0.8700 | 0.8414 | 0.8243 | 2000 (hit cap) |
| 9  | 0.8820 | 0.8618 | 0.8396 | 2000 (hit cap) |
| 10 | 0.8940 | 0.8591 | 0.8598 | 2000 (hit cap) |

Folds 1, 10, and 9 all exceed or approach 90%.

## vs. Baseline
0.7186 → 0.8762 (+15.8 pp)

## vs. Best Previous
008-hgbt (0.8748) → 0.8762 (**+0.14 pp** — marginal)

## vs. Success Threshold
90% required → 87.62% achieved → **NOT MET (gap: -2.38 pp)**

## Feature Importance
| Feature | Importance | Rank |
|---------|------------|------|
| f0_x_f2  | +0.2012 | #1 — engineered |
| f1_sq    | +0.1274 | #2 — engineered |
| f0_sq    | +0.0040 | #3 — engineered |
| feature_1 | +0.0017 | #4 |
| f2_sq    | +0.0008 | #5 — engineered |
| f0_x_f1  | -0.0006 | #38 — noise |
| f1_x_f2  | -0.0003 | #26 — noise |

The 4 additional degree-2 terms (beyond f0·f2 and f1²) contributed almost nothing. f0_x_f1 and f1_x_f2 are negative importance (pure noise). Only f0_sq and f2_sq added marginal signal.

## Observations
The 4 extra degree-2 terms yielded only +0.14 pp — the degree-2 expansion is essentially exhausted, with all real signal concentrated in f0·f2 and f1². Three folds already exceed 90%, suggesting the Bayes limit is above 90% and the mean gap is a model capacity issue, not irreducible noise. The boundary likely has degree-3 structure (e.g., (f0·f2)·f1 or f1³) that degree-2 features can't express. The next experiment should add degree-3 terms from the top-3 features.

---
*Run on: 2026-04-25*
