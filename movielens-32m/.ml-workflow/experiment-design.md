# Experiment Design

## Primary Metric
NDCG@10 — standard ranking metric for implicit-feedback recommenders. Rewards placing relevant items higher in the top-10 recommendation list. Symmetric error priority makes a balanced ranking metric the right choice.

## Secondary Metrics
- **Recall@10** — fraction of held-out items appearing in top-10 (coverage check)
- **Hit Rate@10** — fraction of users with at least one hit in top-10 (user-level success rate)

## Validation Strategy
Global time cutoff split. All interactions before the cutoff are training data; all interactions on or after the cutoff are test data. This mirrors production where the model is trained on historical data and evaluated on future behavior.

Cutoff selection: choose a date that puts roughly the last 3 months of data into the test set. Users with no test interactions are excluded from evaluation. Users with no training interactions are excluded entirely.

## Baseline Approach
Global popularity: rank all movies by total interaction count in the training set. For each test user, recommend the top-10 most popular movies they haven't already seen in training. No personalization — this is the floor that all experiments must beat.

## Success Threshold
10% relative improvement in NDCG@10 over the popularity baseline. If baseline NDCG@10 = X, success means achieving ≥ 1.1X.

---
*Designed on: 2026-04-14*
