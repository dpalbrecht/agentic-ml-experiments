# Experiment 005: LightGBM 3-Month Validation

## Approach
Same LightGBM as experiment 004 (`is_unbalance=True`, default hyperparameters, 11 features). Threshold tuned via PR curve on a 3-month temporal validation window (last 3 months of the 12-month training window). Sub-train on months 1–9, validate on months 10–12, refit on full 12 months.

## Rationale
Experiment 004 showed the LightGBM model is strong (validation F2 0.91–0.96) but threshold tuning was unstable because a single validation month (~300 samples) is too noisy for the PR curve. A 3-month window provides ~1,000 samples — enough to stabilize threshold selection.

## What changed from previous experiments
- Experiment 004 → 005: Validation window for threshold tuning expanded from 1 month to 3 months. Everything else identical.

## Evaluation
- **Primary metric:** F2 score
- **Secondary metrics:** Recall, Precision
- **Validation:** Sliding window temporal CV — 12-month train, 1-month test, 6 folds (test months Jul–Dec 2023)
