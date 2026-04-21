# Experiment 001: ALS — Results

## Approach
Implicit ALS matrix factorization via the `implicit` library. factors=128, regularization=0.01, alpha=1.0, iterations=15. Binary interaction matrix (any rating = positive).

## Results
**Primary metric (NDCG@10):** 0.0870
**Secondary metrics:**
- Recall@10: 0.0423
- Hit Rate@10: 0.3366

## Validation Details
- Global time cutoff: 2023-07-13
- Train: 31,713,721 interactions
- Test: 286,483 interactions
- Eval users: 3,535
- Training time: 42.7s

## vs. Baseline
NDCG@10: 0.0473 → 0.0870 (+0.0397, **+84.0% relative**)
Recall@10: 0.0206 → 0.0423 (+0.0217)
Hit Rate@10: 0.2077 → 0.3366 (+0.1289)

## vs. Best Previous
First experiment after baseline — this is the new best.

## vs. Success Threshold
Target NDCG@10 ≥ 0.0520 (10% relative improvement over baseline) → **MET** (0.0870 is +84% relative improvement)

## Observations
ALS with default-ish hyperparameters already crushes the success threshold by a wide margin — 84% relative improvement over popularity. There's likely more headroom from tuning regularization and alpha together, and from increasing embedding dimension. The model trained in under a minute on 32M interactions, confirming it meets the "not too compute-heavy" constraint.

---
*Run on: 2026-04-14*
