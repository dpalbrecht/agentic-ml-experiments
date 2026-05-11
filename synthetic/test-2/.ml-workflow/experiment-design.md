# Experiment Design

## Primary Metric
**F2 score** — weights recall 2x over precision, directly encoding the problem frame's asymmetric error priority (false negatives are more costly than false positives). Robust to class imbalance.

## Secondary Metrics
- **Recall** — how many actual churners are caught
- **Precision** — how many flagged customers actually churn

## Validation Strategy
**Sliding window temporal CV (12-month train → 1-month test, 6 folds on most recent months)** — the model deploys monthly on new customer data, so evaluation must respect temporal ordering and use a fixed-size training window that mirrors production retraining.

- **Training window**: 12 months (sliding, fixed size ~4,000 rows)
- **Test window**: 1 month (~330 rows)
- **Folds**: 6, covering test months Jul 2023 – Dec 2023
- **Rationale for recency bias**: Churn rate trends upward over time (8–10% in early 2022 → 17–20% in late 2023). Evaluating on the most recent 6 months better reflects the distribution the model will face in deployment.

### Why not stratified k-fold?
Stratified 5-fold CV was considered and rejected. It randomly shuffles data across time, allowing the model to train on future data and predict past data. For a churn model deployed monthly, this creates temporal leakage and inflates performance estimates. Time-based splits are the correct choice here.

### Why not expanding window?
An expanding window was considered and rejected. It grows the training set each fold, meaning earlier folds train on less data and later folds on more. A fixed 12-month sliding window keeps training set size consistent across folds and better matches the production pattern of retraining on a rolling window of recent data.

## Baseline Approach
**Logistic regression with default hyperparameters** using all features except the two identified leakage columns (`cancellation_request`, `account_status_code`).

Feature set for baseline (11 features):
- tenure_days
- monthly_spend
- support_tickets_30d
- product_usage_score
- satisfaction_survey
- days_since_last_login
- noise_0, noise_1, noise_2, noise_3, noise_4

The noise features are retained despite suspect naming and near-zero univariate correlation — univariate correlation only tells part of the story, and they may contribute signal in combination with other features. They should only be dropped if experimentation demonstrates they are not helping.

`record_date` is used for temporal CV splits, not as a model feature.

## Success Threshold
**F2 ≥ 0.90** — high bar reflecting the real-world cost of missed churners in monthly deployment.

---
*Designed on: 2026-05-10*
