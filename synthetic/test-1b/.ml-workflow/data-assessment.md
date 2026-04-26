# Data Assessment

## Overview
5,000 rows × 51 columns (`data/data.csv`). 50 numeric features (`feature_0`–`feature_49`) + 1 binary target. No missing values. No duplicate rows.

## Target: `target`
| Class | Count | % |
|-------|-------|---|
| 0 | 3,593 | 71.9% |
| 1 | 1,407 | 28.1% |

Balance ratio: 2.55:1 — within the balanced threshold (< 3:1). No special handling required.

## Features
All 50 features are continuous, approximately standard-normal (mean ≈ 0, std ≈ 1, range ≈ -4.5 to +4.5). No categorical or mixed-type columns.

| Column | dtype | Missing % | Notes |
|--------|-------|-----------|-------|
| feature_0–feature_47 | float64 | 0% | Standard normal; weak/no signal |
| feature_48 | float64 | 0% | Strongest signal: \|r\|=0.122 |
| feature_49 | float64 | 0% | Strongest signal: \|r\|=0.134 |

## Issues Found
- **Outliers**: All 50 features have 15–22 values beyond 3 std. These counts are consistent with expected tails of a standard normal distribution (≈13 expected per 5,000 rows at 0.27%), not anomalies. No action needed.
- No zero-variance columns, no high inter-feature correlations (max pair |r| < 0.95), no missing data, no duplicates.

## Leakage Risk
None identified. No feature has |r| > 0.99 with the target.

## Signal Structure
Signal is **highly concentrated in 2 features**:
- `feature_49`: |r| = 0.134
- `feature_48`: |r| = 0.122

The remaining 48 features show near-zero correlation with the target (median |r| = 0.012; no feature exceeds |r| = 0.04 outside the top 2). This pattern strongly suggests the dataset contains ~48 noise features and ~2 informative ones. **Feature selection is a critical experiment** — models trained on all 50 features may underperform due to noise dilution, particularly for linear models.

## Recommended Feature Set
- **Baseline**: use all 50 features. Tree-based models (e.g., random forest, gradient boosting) should handle noise gracefully.
- **Key experiment**: feature selection — compare performance using only top-k features by univariate score vs. the full set.

---
*Assessed on: 2026-04-23*
