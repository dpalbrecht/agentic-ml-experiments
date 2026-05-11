# Data Assessment

## Overview
8,000 rows, 15 columns, single file (`data/customer_churn.csv`). Date range: 2022-01-01 to 2023-12-30 (729 unique dates). No missing values, no duplicate rows.

## Target: churned
Binary (0/1). 6,959 not churned, 1,041 churned. **Imbalance ratio: 6.68:1** — will need stratified splits and an imbalance-aware metric. Aligns with problem frame's emphasis on recall (false negatives are costly).

## Features

| Column | dtype | Missing % | Notes |
|---|---|---|---|
| record_date | object | 0 | Date string, 729 unique values. Not a modeling feature. |
| tenure_days | int64 | 0 | Range 30–3,828. Right-skewed, 1.8% outliers. |
| monthly_spend | float64 | 0 | Range 5.32–1,322.73. Right-skewed, 1.6% outliers. |
| support_tickets_30d | int64 | 0 | Range 0–11. |
| product_usage_score | float64 | 0 | Range 4.6–98.7. Strongest clean signal (r=-0.64). |
| satisfaction_survey | int64 | 0 | Range 1–5. Near-zero correlation with target. |
| days_since_last_login | int64 | 0 | Range 0–129. Right-skewed, 1.5% outliers. |
| cancellation_request | int64 | 0 | Binary 0/1. **⚠ LEAKAGE** (r=+0.88). |
| account_status_code | int64 | 0 | Values {1, 2, 3}. **⚠ LEAKAGE** (r=+0.88). |
| noise_0–noise_4 | float64 | 0 | Standard-normal random noise. All |r| < 0.03 with target. |

## Issues Found
- **Class imbalance**: 6.68:1 ratio requires stratified splitting and recall-focused evaluation.
- **Leakage (see below)**: Two features must be dropped before modeling.
- **Right-skewed distributions**: tenure_days, monthly_spend, days_since_last_login have long tails. Tree models handle this natively; linear models may benefit from log transforms.

## Leakage Risk
Two features are almost certainly leakage:

1. **cancellation_request** (r=+0.88): Of 1,041 churned customers, 1,010 had a cancellation request. This is a consequence of churning, not a predictor — a customer requests cancellation *because* they are about to churn. Using it would give artificially high accuracy that won't generalize.
2. **account_status_code** (r=+0.88): Code 3 has exactly 1,041 entries — the same count as churned customers. This is almost certainly a post-hoc status flag reflecting churn, not a leading indicator.

Both must be excluded from all experiments.

## Signal Structure
Signal is **concentrated in a few features** after removing leakage:

| Feature | Correlation with target | Strength |
|---|---|---|
| product_usage_score | -0.64 | Strong |
| days_since_last_login | +0.51 | Moderate |
| monthly_spend | -0.26 | Weak |
| tenure_days | -0.06 | Very weak |
| support_tickets_30d | +0.05 | Very weak |
| satisfaction_survey | -0.04 | Negligible |

Only 2–3 features carry meaningful signal. The 5 noise columns and satisfaction_survey are near-zero. Feature selection should be explored during experimentation — a model using only the top 3–5 features may outperform one using all features due to noise reduction.

## Recommended Feature Set

**Use:**
- product_usage_score — strongest predictor
- days_since_last_login — strong predictor
- monthly_spend — moderate predictor
- tenure_days — weak but plausibly useful
- support_tickets_30d — weak but plausibly useful

**Drop:**
- cancellation_request — leakage
- account_status_code — leakage
- noise_0 through noise_4 — explicitly labeled noise, confirmed near-zero signal
- record_date — temporal metadata, not a feature
- satisfaction_survey — near-zero signal (borderline; could revisit if needed)

---
*Assessed on: 2026-05-10*
