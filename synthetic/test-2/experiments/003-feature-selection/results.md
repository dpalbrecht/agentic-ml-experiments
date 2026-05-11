# Experiment 003: Feature Selection (Top 3) — Results

## Approach
Same threshold-tuned RF as experiment 002, reduced from 11 to 3 features (product_usage_score, days_since_last_login, monthly_spend). OOB threshold tuning, 500 trees.

## Results
**Primary metric (F2): 0.9114 ± 0.0233**
**Secondary metrics:** Recall: 0.9419 ± 0.0208 | Precision: 0.8121 ± 0.0698

## Validation Details

| Fold | Test Month | Threshold | F2 | Recall | Precision | OOB F2 |
|---|---|---|---|---|---|---|
| 1 | 2023-07 | 0.19 | 0.8658 | 0.9302 | 0.6780 | 0.9139 |
| 2 | 2023-08 | 0.30 | 0.9423 | 0.9800 | 0.8167 | 0.9127 |
| 3 | 2023-09 | 0.24 | 0.9251 | 0.9545 | 0.8235 | 0.9161 |
| 4 | 2023-10 | 0.20 | 0.9138 | 0.9138 | 0.9138 | 0.9182 |
| 5 | 2023-11 | 0.21 | 0.9135 | 0.9344 | 0.8382 | 0.9199 |
| 6 | 2023-12 | 0.19 | 0.9077 | 0.9385 | 0.8026 | 0.9205 |

## vs. Baseline
0.9091 → 0.9114 (**+0.0023**) — marginal improvement over baseline

## vs. Best Previous
Best is experiment 002 at 0.9219. This is **-0.0105** below — worse on 5 of 6 folds.

## vs. Success Threshold
F2 ≥ 0.90 → **MET** (margin: +0.0114)

## Feature Importance

| Rank | Feature | Impurity | Permutation |
|---|---|---|---|
| 1 | product_usage_score | 0.5354 | 0.5854 |
| 2 | days_since_last_login | 0.2939 | 0.3091 |
| 3 | monthly_spend | 0.1707 | 0.0370 |

## Observations
Dropping from 11 to 3 features slightly hurt performance (F2: 0.9219 → 0.9114). The RF appears to benefit from having the additional features as splitting dimensions, even though their individual permutation importance is near zero. The "noise" features may be providing useful randomization or weak interaction effects that collectively help. Fold 1 (Jul 2023) remains the weakest at 0.8658.

**Conclusion: keep the full 11-feature set.** The noise features aren't hurting enough to justify removing them, and the 11-feature threshold-tuned RF (exp 002, F2=0.9219) remains the best model.

---
*Run on: 2026-05-10*
