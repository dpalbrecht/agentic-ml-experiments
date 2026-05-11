# Experiment 004: LightGBM

## Approach
LightGBM classifier with `is_unbalance=True`, default hyperparameters (n_estimators=100, learning_rate=0.1, num_leaves=31). Full 11-feature set. F2-optimized threshold tuned via temporal validation split (last month of training window).

## Rationale
The RF line has been explored: threshold tuning was highly effective (exp 002, F2=0.9219) but feature selection hurt (exp 003). Gradient boosting is the natural next model family — GBMs learn sequentially (each tree corrects prior errors) and typically outperform bagging on tabular data. They also handle irrelevant features more gracefully since weak features naturally receive fewer splits across boosting rounds.

## What changed from previous experiments
- Model family: Random Forest → LightGBM (gradient boosting)
- Threshold tuning method: OOB predictions → temporal validation split (last month of 12-month train window used for threshold tuning, then refit on full window)
- Imbalance handling: `class_weight='balanced_subsample'` → `is_unbalance=True`

## Evaluation
- **Primary metric:** F2 score
- **Secondary metrics:** Recall, Precision
- **Validation:** Sliding window temporal CV — 12-month train, 1-month test, 6 folds (test months Jul–Dec 2023)
