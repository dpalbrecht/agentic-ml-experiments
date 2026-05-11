# Experiment 002: RF Threshold Tuning — Results

## Approach
Same Random Forest as 001 (500 trees, `class_weight='balanced_subsample'`), with decision threshold tuned per-fold using OOB predictions to maximize F2. Thresholds ranged from 0.17–0.27 (mean 0.228), far below the default 0.5.

## Results
**Primary metric (F2): 0.9219 ± 0.0247**
**Secondary metrics:** Recall: 0.9543 ± 0.0284 | Precision: 0.8165 ± 0.0619

## Validation Details

| Fold | Test Month | Threshold | F2 | Recall | Precision | OOB F2 |
|---|---|---|---|---|---|---|
| 1 | 2023-07 | 0.24 | 0.8734 | 0.9302 | 0.7018 | 0.9208 |
| 2 | 2023-08 | 0.17 | 0.9470 | 1.0000 | 0.7812 | 0.9145 |
| 3 | 2023-09 | 0.22 | 0.9430 | 0.9773 | 0.8269 | 0.9226 |
| 4 | 2023-10 | 0.27 | 0.9107 | 0.9138 | 0.8983 | 0.9229 |
| 5 | 2023-11 | 0.25 | 0.9295 | 0.9508 | 0.8529 | 0.9201 |
| 6 | 2023-12 | 0.22 | 0.9281 | 0.9538 | 0.8378 | 0.9283 |

## vs. Baseline
0.9091 → 0.9219 (**+0.0128**) — new best

## vs. Best Previous
Best was baseline (000) at 0.9091. This is +0.0128 above.

## vs. Success Threshold
F2 ≥ 0.90 → **MET** (margin: +0.0219)

## Feature Importance

**Permutation importance (F2-based, with tuned threshold):**

| Rank | Feature | Importance | Notes |
|---|---|---|---|
| 1 | product_usage_score | 0.6017 | Dominant |
| 2 | days_since_last_login | 0.3257 | Strong |
| 3 | monthly_spend | 0.0411 | Moderate |
| 4 | satisfaction_survey | 0.0010 | Near zero |
| 5 | noise_0 | 0.0004 | Near zero |
| 6 | noise_3 | 0.0002 | Near zero |
| 7 | noise_4 | 0.0001 | Near zero |
| 8 | noise_1 | -0.0015 | Negative (hurts) |
| 9 | noise_2 | -0.0017 | Negative (hurts) |
| 10 | tenure_days | -0.0026 | Negative (hurts) |
| 11 | support_tickets_30d | -0.0031 | Negative (hurts) |

Signal remains concentrated in 3 features. Notably, tenure_days and support_tickets_30d now show *negative* permutation importance — they may be actively hurting performance. All noise features are at or below zero.

## Observations
Threshold tuning was highly effective — dropping the threshold from 0.5 to ~0.23 recovered the recall lost in experiment 001 (0.86→0.95) while maintaining better precision than the baseline LR (0.82 vs 0.77). Fold 1 (Jul 2023) remains the weakest at F2=0.87. The feature importance picture is now very clear: 8 of 11 features contribute nothing or hurt. Feature selection is the obvious next experiment.

---
*Run on: 2026-05-10*
