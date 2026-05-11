# Experiment 002: RF Threshold Tuning

## Approach
Same Random Forest as experiment 001 (500 trees, `class_weight='balanced_subsample'`), but with F2-optimized decision threshold instead of default 0.5. Threshold is tuned per-fold using OOB predictions on the training set — no test data leakage.

## Rationale
Experiment 001 showed the RF has high precision (0.90) but lower recall (0.86) compared to baseline LR (precision 0.77, recall 0.95). The default 0.5 threshold is too conservative for F2 which weights recall 2x. Lowering the threshold should convert excess precision into recall, directly improving F2.

## What changed from previous experiments
- Experiment 001 → 002: Only the decision threshold changes. Instead of default 0.5, sweep thresholds on OOB predictions to find the F2-maximizing threshold per fold.

## Evaluation
- **Primary metric:** F2 score
- **Secondary metrics:** Recall, Precision
- **Validation:** Sliding window temporal CV — 12-month train, 1-month test, 6 folds (test months Jul–Dec 2023)
