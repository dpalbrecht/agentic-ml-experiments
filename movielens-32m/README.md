# MovieLens 32M

## What We Tested

MovieLens 32M: ~32 million user-movie ratings from 200,000 users across 87,000 movies (Jan 2006 – Oct 2023). A large-scale implicit feedback recommendation problem — ratings are treated as positive interactions, with no explicit negatives.

**Problem setup:** Top-10 movie recommendation per user, NDCG@10 as primary metric (binary relevance), global time cutoff validation (cutoff 2023-07-13, ~last 3 months in test). Secondary metrics: Recall@10, Hit Rate@10. Success threshold: 10% relative improvement over popularity baseline (NDCG@10 ≥ 0.0520).

## What Happened

| # | Approach | NDCG@10 | Recall@10 | Hit Rate@10 | Delta vs Baseline |
|---|----------|---------|-----------|-------------|-------------------|
| 000 | Global popularity (top-10 unseen) | 0.0473 | 0.0206 | 20.8% | — (baseline) |
| 001 | ALS matrix factorization (factors=128, α=1.0, 15 iters) | 0.0870 | 0.0423 | 33.7% | **+84% relative** ✓ |

The success threshold was met in experiment 001. No further experiments were run.

## Key Observations

**ALS crushed the threshold on the first try.** The +84% relative improvement over popularity (0.0473 → 0.0870 NDCG@10) is a large margin on a dataset of this scale. The 10% relative threshold was conservative — a reasonable floor given that popularity baselines are unusually strong in recommender systems, but not a challenging target for any personalization method.

**Training was fast.** 42.7 seconds on 32M interactions confirms ALS meets the compute constraint. The `implicit` library's ALS implementation scales well and the iteration count (15) left room to increase if needed.

**Time-based validation was correctly implemented.** The global cutoff split ensures the model is evaluated on future interactions, not random held-out ratings — the right approach for temporal data. 3,535–3,591 eval users (those present in both train and test) is a reasonable evaluation population for a dataset of this size.

**The workflow stopped too early.** Meeting the success threshold in experiment 001 is a good outcome, but the baseline ALS result leaves obvious headroom: factors (128 is modest for 32M interactions), regularization, and alpha are all at near-default values. The 10% threshold was the exit criterion, but a more informative question — "how much is left on the table?" — was never asked.

## Assessment

The workflow handled the scale and temporal structure correctly. The time cutoff validation strategy was sound, the metric choice (NDCG@10) was appropriate for ranked recommendation, and ALS was a reasonable first personalization method to try.

The main limitation is that the experiment loop exited as soon as the threshold was met, leaving the ALS result essentially uncharacterized. For a dataset this size, the experiment log should answer: what does ALS performance look like across different user activity levels (cold vs. warm users)? How sensitive is NDCG@10 to the number of factors? Is the 0.0870 result stable across different time cutoffs, or is it sensitive to which 3-month window is chosen?

The low absolute NDCG@10 values (0.05–0.09) are also worth noting. With 87,000 movies in the catalog and a 10-item recommendation list, these numbers are expected — but a knowledgeable user would want to sanity-check them against published baselines on MovieLens-20M or MovieLens-25M before concluding the model is performing well in any meaningful sense.
