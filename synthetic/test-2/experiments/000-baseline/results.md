# Experiment 000: Baseline — Results

## Approach
Logistic regression with `class_weight='balanced'`, StandardScaler, 11 features (all except leakage columns). Sliding window temporal CV: 12-month train, 1-month test, 6 folds on Jul–Dec 2023.

## Results
**Primary metric (F2): 0.9091 ± 0.0168**
**Secondary metrics:** Recall: 0.9535 ± 0.0229 | Precision: 0.7702 ± 0.0510

## Validation Details

| Fold | Test Month | Train | Test | Pos | F2 | Recall | Precision |
|---|---|---|---|---|---|---|---|
| 1 | 2023-07 | 3995 | 308 | 43 | 0.8798 | 0.9535 | 0.6721 |
| 2 | 2023-08 | 3959 | 359 | 50 | 0.9363 | 1.0000 | 0.7463 |
| 3 | 2023-09 | 3998 | 321 | 44 | 0.9031 | 0.9318 | 0.8039 |
| 4 | 2023-10 | 3975 | 329 | 58 | 0.9091 | 0.9310 | 0.8308 |
| 5 | 2023-11 | 3958 | 326 | 61 | 0.9091 | 0.9508 | 0.7733 |
| 6 | 2023-12 | 3954 | 322 | 65 | 0.9172 | 0.9538 | 0.7949 |

## Feature Importance

| Rank | Feature | Mean |coef| |
|---|---|---|
| 1 | product_usage_score | 3.8701 |
| 2 | monthly_spend | 1.6866 |
| 3 | days_since_last_login | 1.5324 |
| 4 | tenure_days | 0.3256 |
| 5 | noise_3 | 0.2299 |
| 6 | support_tickets_30d | 0.1686 |
| 7 | noise_4 | 0.1159 |
| 8 | noise_0 | 0.0739 |
| 9 | noise_2 | 0.0650 |
| 10 | noise_1 | 0.0426 |
| 11 | satisfaction_survey | 0.0119 |

Signal is heavily concentrated in the top 3 features (product_usage_score, monthly_spend, days_since_last_login). The noise features and satisfaction_survey have near-zero coefficients — consistent with the data assessment but retained per design decision.

## vs. Success Threshold
F2 ≥ 0.90 → **MET** (0.9091, margin of +0.0091)

Threshold is met on average but fold 1 (Jul 2023, F2=0.8798) falls below. The margin is thin.

---
*Run on: 2026-05-10*
