# Experiment 000: Baseline

## Approach
Global popularity: rank all movies by total interaction count in the training set. For each test user, recommend the top-10 most popular movies they haven't already seen in training. No personalization.

## Rationale
Establishes the floor. All future experiments must beat this. Popularity baselines are notoriously strong in recommender systems — a meaningful lower bound.

## Evaluation
- **Primary metric:** NDCG@10 (binary relevance — 1 if user interacted with item in test set)
- **Secondary metrics:** Recall@10, Hit Rate@10
- **Validation strategy:** Global time cutoff split. Cutoff chosen to put ~last 3 months in the test set (approx. 2023-07-13). Users with no training or no test interactions excluded.
