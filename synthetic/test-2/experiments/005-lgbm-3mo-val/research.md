# Experiment 005: LightGBM 3-Month Validation — Research

## Approach Researched
Stabilizing LightGBM threshold tuning by using a 3-month temporal validation window instead of 1 month.

## Key Findings
- No external research needed — this experiment is motivated entirely by internal evidence from experiment 004.
- Experiment 004's threshold instability (range 0.05–0.92) was caused by using a single validation month (~300 samples, ~40–60 positives). The PR curve on such small data is noisy and the optimal threshold shifts dramatically depending on the specific month's composition.
- For comparison, experiment 002's OOB threshold tuning used ~4,000 samples and produced thresholds in a tight 0.17–0.27 range.
- A 3-month validation window provides ~1,000 samples with ~120–170 positives — a 3x increase that should significantly stabilize the PR curve.
- Trade-off: sub-training uses 9 months instead of 11, losing ~660 rows of training data. However, the refit step uses all 12 months before test prediction, so the final model is unaffected.

## Sources
- Internal experiment 004 results (threshold instability analysis)
- Internal experiment 002 results (OOB threshold stability baseline)
