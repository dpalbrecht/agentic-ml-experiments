# Data Assessment

## Overview
5,000 rows × 51 columns (50 features + target). Single file: `data/data.csv`.

## Target: `target`
| Class | Count | % |
|-------|-------|---|
| 0     | 3,593 | 71.9% |
| 1     | 1,407 | 28.1% |

Imbalance ratio: 2.55:1 — below the 3:1 warning threshold. Classes are sufficiently balanced for standard metrics.

## Features
All 50 features (`feature_0` – `feature_49`) are `float64`. All have mean ≈ 0, std ≈ 1 — consistent with standard-normal generation. No missing values in any column.

| Metric | Value |
|--------|-------|
| Missing values | 0 across all columns |
| Zero-variance columns | None |
| High-correlation pairs (>0.95) | None |
| Duplicate rows | 0 |

All features have a small number of outliers (>3 std), ranging from ~10–22 per feature out of 5,000 rows (<0.5%). This is statistically expected for standard-normal data and not a quality issue.

## Issues Found
- None. Data is clean, complete, and well-formed.

## Leakage Risk
None identified. No feature exceeds 0.9 correlation with `target`. The two most correlated features are `feature_49` (|r| = 0.134) and `feature_48` (|r| = 0.122) — weak individual signals, no leakage concern.

## Recommended Feature Set
Use all 50 features. No columns to drop:
- No missing data to exclude
- No zero-variance columns
- No high-correlation redundancy
- No leakage risk

Individual feature–target correlations are weak (max 0.134), suggesting this is a noisy, low-signal problem where the target is determined by complex feature interactions rather than individual features. Ensemble methods (gradient boosting, random forest) are likely to outperform linear models.

---
*Assessed on: 2026-04-23*
