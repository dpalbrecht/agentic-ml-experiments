# Experiment 001: Random Forest — Results

## Approach
Random Forest (500 trees, `class_weight='balanced_subsample'`, default hyperparameters). 11 features, no scaling. Sliding window temporal CV (12-month train, 1-month test, 6 folds).

## Results
**Primary metric (F2): 0.8666 ± 0.0321**
**Secondary metrics:** Recall: 0.8589 ± 0.0416 | Precision: 0.9045 ± 0.0503

## Validation Details

| Fold | Test Month | Train | Test | Pos | F2 | Recall | Precision |
|---|---|---|---|---|---|---|---|
| 1 | 2023-07 | 3995 | 308 | 43 | 0.8676 | 0.8837 | 0.8085 |
| 2 | 2023-08 | 3959 | 359 | 50 | 0.9127 | 0.9200 | 0.8846 |
| 3 | 2023-09 | 3998 | 321 | 44 | 0.8525 | 0.8409 | 0.9024 |
| 4 | 2023-10 | 3975 | 329 | 58 | 0.8627 | 0.8448 | 0.9423 |
| 5 | 2023-11 | 3958 | 326 | 61 | 0.8108 | 0.7869 | 0.9231 |
| 6 | 2023-12 | 3954 | 322 | 65 | 0.8934 | 0.8769 | 0.9661 |

## vs. Baseline
0.9091 → 0.8666 (**-0.0425**) — worse than baseline

## vs. Best Previous
Best is baseline (000) at 0.9091. This is -0.0425 below.

## vs. Success Threshold
F2 ≥ 0.90 → **NOT MET** (gap of 0.0334)

## Feature Importance

**Impurity-based (MDI):**

| Rank | Feature | Importance |
|---|---|---|
| 1 | product_usage_score | 0.5274 |
| 2 | days_since_last_login | 0.2395 |
| 3 | monthly_spend | 0.1163 |
| 4 | tenure_days | 0.0193 |
| 5 | noise_1 | 0.0180 |
| 6–10 | noise_0/2/3/4 | 0.0158–0.0174 |
| 11 | satisfaction_survey | 0.0055 |

**Permutation importance (F2-based):**

| Rank | Feature | Importance |
|---|---|---|
| 1 | product_usage_score | 0.6093 |
| 2 | days_since_last_login | 0.3757 |
| 3 | monthly_spend | 0.1241 |
| 4 | tenure_days | 0.0081 |
| 5 | noise_4 | 0.0080 |
| 6–9 | noise_0/2/3, support_tickets | 0.0010–0.0042 |
| 10 | satisfaction_survey | -0.0036 |
| 11 | noise_1 | -0.0042 |

Signal is heavily concentrated in 3 features. The 5 noise features, satisfaction_survey, and support_tickets_30d have near-zero or negative permutation importance — they are not helping.

## Observations
The Random Forest trades recall for precision: recall dropped from 0.95 → 0.86 while precision rose from 0.77 → 0.90. Since F2 weights recall 2x, this is a net loss. The RF's default 0.5 threshold is too conservative for a recall-heavy metric. Permutation importance confirms that only 3 features carry meaningful signal — the other 8 features are noise or near-noise, which may be hurting the RF more than the regularized logistic regression.

---
*Run on: 2026-05-10*
