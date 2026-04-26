# Experiment 008: HGBT + Engineered Features — Results

## Approach
HGBT (max_depth=5, lr=0.1, max_iter=2000, early stopping) on 52 features: original 50 + `f0_x_f2` (feature_0 × feature_2) + `f1_sq` (feature_1²).

## Results
**Primary metric (Accuracy):** 0.8748 ± 0.0150
**Secondary metrics:**
- ROC-AUC: 0.8456 ± 0.0221
- Macro F1: 0.8330 ± 0.0209

## Validation Details
Stratified 10-fold CV:
| Fold | Accuracy | ROC-AUC | Macro F1 | Iterations |
|------|----------|---------|----------|------------|
| 1  | 0.9040 | 0.8872 | 0.8753 | 2000 (hit cap) |
| 2  | 0.8620 | 0.8390 | 0.8155 | 2000 (hit cap) |
| 3  | 0.8620 | 0.8087 | 0.8092 | 2000 (hit cap) |
| 4  | 0.8640 | 0.8271 | 0.8215 | 2000 (hit cap) |
| 5  | 0.8580 | 0.8374 | 0.8131 | 2000 (hit cap) |
| 6  | 0.8740 | 0.8269 | 0.8306 | 2000 (hit cap) |
| 7  | 0.8740 | 0.8687 | 0.8367 | 2000 (hit cap) |
| 8  | 0.8660 | 0.8407 | 0.8189 | 2000 (hit cap) |
| 9  | 0.8920 | 0.8632 | 0.8528 | 2000 (hit cap) |
| 10 | 0.8920 | 0.8566 | 0.8568 | 2000 (hit cap) |

Fold 1 reached 90.40% — first fold to cross the success threshold.

## vs. Baseline
0.7186 → 0.8748 (+15.6 pp)

## vs. Best Previous
005-hgbt (0.8440) → 0.8748 (**+3.08 pp** — largest single-step gain since switching to HGBT)

## vs. Success Threshold
90% required → 87.48% achieved → **NOT MET (gap: -2.52 pp)**

## Feature Importance
Top 10 by mean permutation importance:
| Feature | Importance | Note |
|---------|------------|------|
| f0_x_f2 | +0.2064 | engineered — rank #1 |
| f1_sq   | +0.1277 | engineered — rank #2 |
| feature_1 | +0.0021 | original |
| feature_0 | +0.0006 | original |
| feature_24 | +0.0004 | original |

The two engineered features now dominate by 2 orders of magnitude over all originals, confirming they capture the core signal. The remaining gap to 90% sits in residual terms.

## Observations
Adding `f0_x_f2` and `f1_sq` broke through the 84% ceiling with a +3.08 pp jump — the largest gain since first switching to HGBT. Fold 1 hit 90.4%, showing 90% is achievable. The remaining 2.52 pp gap likely comes from the smaller polynomial terms identified in experiment 007 (feature_0², feature_2², feature_0×feature_1) which were not added here. Adding the full degree-2 expansion of the top-3 features to the original 50 is the logical next step.

---
*Run on: 2026-04-25*
