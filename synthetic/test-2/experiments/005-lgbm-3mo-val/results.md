# Experiment 005: LightGBM 3-Month Validation — Results

## Approach
Same LightGBM as experiment 004, but threshold tuned on a 3-month temporal validation window (last 3 months of training) instead of 1 month. Sub-train on 9 months, validate on 3 months (~1,000 samples), refit on full 12 months.

## Results
**Primary metric (F2): 0.9104 ± 0.0157**
**Secondary metrics:** Recall: 0.9422 ± 0.0267 | Precision: 0.8061 ± 0.0481

## Validation Details

| Fold | Test Month | Val Window | Val Size | Val Pos | Threshold | F2 | Recall | Precision |
|---|---|---|---|---|---|---|---|---|
| 1 | 2023-07 | Apr–Jun 2023 | 980 | 167 | 0.27 | 0.8929 | 0.9302 | 0.7692 |
| 2 | 2023-08 | May–Jul 2023 | 949 | 157 | 0.22 | 0.9387 | 0.9800 | 0.8033 |
| 3 | 2023-09 | Jun–Aug 2023 | 965 | 149 | 0.41 | 0.8929 | 0.9091 | 0.8333 |
| 4 | 2023-10 | Jul–Sep 2023 | 988 | 137 | 0.22 | 0.9107 | 0.9138 | 0.8983 |
| 5 | 2023-11 | Aug–Oct 2023 | 1009 | 152 | 0.02 | 0.9091 | 0.9508 | 0.7733 |
| 6 | 2023-12 | Sep–Nov 2023 | 976 | 163 | 0.02 | 0.9184 | 0.9692 | 0.7590 |

## vs. Baseline
0.9091 → 0.9104 (**+0.0013**) — marginal improvement

## vs. Best Previous
Best is experiment 002 (RF + OOB threshold) at 0.9219. This is **-0.0115** below.

## vs. Success Threshold
F2 ≥ 0.90 → **MET** (margin: +0.0104)

## Threshold Stability

| Experiment | Method | Range | Spread |
|---|---|---|---|
| 002 (RF) | OOB (~4,000 samples) | 0.17–0.27 | 0.10 |
| 004 (LGBM) | 1-month val (~300 samples) | 0.05–0.92 | 0.87 |
| 005 (LGBM) | 3-month val (~1,000 samples) | 0.02–0.41 | 0.39 |

3-month window cut threshold spread from 0.87 to 0.39 — a big improvement but still 4x wider than the RF OOB approach.

## Feature Importance

**Permutation importance (F2-based):**

| Rank | Feature | Importance |
|---|---|---|
| 1 | product_usage_score | 0.5818 |
| 2 | days_since_last_login | 0.2962 |
| 3 | monthly_spend | 0.0497 |
| 4 | noise_2 | 0.0032 |
| 5 | tenure_days | 0.0028 |
| 6–11 | remaining | ≤ 0.0018 |

Same top 3 across all experiments.

## Observations
The 3-month validation window fixed most of experiment 004's threshold instability (+0.014 F2) and now meets the success threshold. However, the RF + OOB approach (exp 002) remains superior for two reasons: (1) OOB uses ~4,000 samples vs ~1,000, giving more stable thresholds, and (2) the RF doesn't sacrifice any training data for threshold tuning. The LightGBM model itself performs comparably to the RF, but its threshold tuning disadvantage keeps it behind.

---
*Run on: 2026-05-10*
