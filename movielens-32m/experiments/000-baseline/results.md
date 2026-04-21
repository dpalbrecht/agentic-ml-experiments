# Experiment 000: Baseline — Results

## Approach
Global popularity baseline: ranked all movies by interaction count in training data, recommended top-10 most popular unseen movies per user. No personalization.

## Results
**Primary metric (NDCG@10):** 0.0473
**Secondary metrics:**
- Recall@10: 0.0206
- Hit Rate@10: 0.2077

## Validation Details
- Global time cutoff: 2023-07-13
- Train: 31,713,721 interactions (199,740 users)
- Test: 286,483 interactions (4,799 users)
- Eval users (in both train and test): 3,591
- The 3-month test window captures a relatively small set of active users, which is expected for a global cutoff on a dataset ending 2023-10-13.

## vs. Success Threshold
Success requires 10% relative improvement over baseline → target NDCG@10 ≥ 0.0520.

---
*Run on: 2026-04-14*
