# Experiment 001: ALS

## Approach
Implicit ALS matrix factorization via the `implicit` library. Learns latent factor vectors for users and items from the binary interaction matrix. Recommends by scoring unseen items via user-item dot products.

Hyperparameters:
- factors=128
- regularization=0.01
- alpha=1.0
- iterations=15

## Rationale
The baseline (popularity) has no personalization. ALS is the standard first personalization step for implicit feedback — fast, scalable, and proven competitive with proper tuning. Starting with mid-sized embeddings (128) per research guidance.

## What changed from previous experiments
- Adds personalization via learned user and item latent factors
- Baseline recommended the same popular movies to all users; ALS gives each user a unique ranking

## Evaluation
- **Primary metric:** NDCG@10 (binary relevance)
- **Secondary metrics:** Recall@10, Hit Rate@10
- **Validation strategy:** Global time cutoff (2023-07-13). Same split as baseline.
