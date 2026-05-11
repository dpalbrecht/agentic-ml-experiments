# Experiment 004: LightGBM — Results

## Approach
LightGBM with `is_unbalance=True`, default hyperparameters, 11 features. Threshold tuned via precision-recall curve on temporal validation split (last month of training window). Refitted on full training window before test prediction.

## Results
**Primary metric (F2): 0.8965 ± 0.0287**
**Secondary metrics:** Recall: 0.9169 ± 0.0387 | Precision: 0.8297 ± 0.0615

## Validation Details

| Fold | Test Month | Val Month | Threshold | F2 | Recall | Precision | Val F2 |
|---|---|---|---|---|---|---|---|
| 1 | 2023-07 | 2023-06 | 0.14 | 0.8811 | 0.9302 | 0.7273 | 0.9441 |
| 2 | 2023-08 | 2023-07 | 0.92 | 0.8502 | 0.8400 | 0.8936 | 0.9091 |
| 3 | 2023-09 | 2023-08 | 0.58 | 0.9050 | 0.9091 | 0.8889 | 0.9449 |
| 4 | 2023-10 | 2023-09 | 0.06 | 0.9459 | 0.9655 | 0.8750 | 0.9442 |
| 5 | 2023-11 | 2023-10 | 0.05 | 0.8946 | 0.9180 | 0.8116 | 0.9564 |
| 6 | 2023-12 | 2023-11 | 0.10 | 0.9024 | 0.9385 | 0.7821 | 0.9295 |

## vs. Baseline
0.9091 → 0.8965 (**-0.0126**) — worse than baseline

## vs. Best Previous
Best is experiment 002 (RF + threshold) at 0.9219. This is **-0.0254** below.

## vs. Success Threshold
F2 ≥ 0.90 → **NOT MET** (gap of 0.0035)

## Feature Importance

**Permutation importance (F2-based):**

| Rank | Feature | Importance |
|---|---|---|
| 1 | product_usage_score | 0.5740 |
| 2 | days_since_last_login | 0.3093 |
| 3 | monthly_spend | 0.0541 |
| 4 | noise_3 | 0.0060 |
| 5 | noise_2 | 0.0055 |
| 6–11 | remaining | ≤ 0.002 |

Same top 3 features as all prior experiments.

## Observations
The LightGBM model itself is likely fine — validation F2 scores are high (0.91–0.96). The problem is **threshold instability**: thresholds ranged from 0.05 to 0.92 across folds. A single validation month (~300 samples, ~40–60 positive) is too small and noisy for reliable threshold tuning via the PR curve. By contrast, experiment 002's OOB approach used ~4,000 samples and produced stable thresholds (0.17–0.27). 

Next step: stabilize threshold tuning by using a longer validation window (e.g., last 3 months of training) or internal temporal cross-validation, rather than a single month.

---
*Run on: 2026-05-10*
