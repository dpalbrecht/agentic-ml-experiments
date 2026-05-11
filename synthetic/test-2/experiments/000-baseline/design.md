# Experiment 000: Baseline

## Approach
Logistic regression with `class_weight='balanced'` and StandardScaler preprocessing. Uses all features except the two leakage columns (`cancellation_request`, `account_status_code`) and the date column (`record_date`).

11 features: tenure_days, monthly_spend, support_tickets_30d, product_usage_score, satisfaction_survey, days_since_last_login, noise_0, noise_1, noise_2, noise_3, noise_4.

## Rationale
Establishes the floor. All future experiments must beat this. Logistic regression is the simplest model that actually learns feature-target relationships. `class_weight='balanced'` compensates for the 6.68:1 class imbalance so the model doesn't collapse to all-negative predictions.

## Evaluation
- **Primary metric:** F2 score (fbeta_score with beta=2)
- **Secondary metrics:** Recall, Precision
- **Validation:** Sliding window temporal CV — 12-month training window, 1-month test window, 6 folds on the most recent months (test months: Jul–Dec 2023). Fixed training size per fold.
