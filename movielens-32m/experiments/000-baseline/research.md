# Experiment 000: Baseline — Research

## Approach Researched
Global popularity baseline for implicit-feedback movie recommendation, evaluated with NDCG@10, Recall@10, and Hit Rate@10 using a global time cutoff split.

## Key Findings
- The standard MostPop baseline ranks items by total interaction count in training data. Research shows this can be significantly improved (70%+) by considering time-aware popularity, but for a baseline floor the simple global count version is appropriate.
- For implicit feedback, any rating counts as a positive interaction — no need to threshold on rating value.
- NDCG@10 for implicit feedback uses binary relevance: 1 if the user interacted with the item in the test set, 0 otherwise. DCG = sum(1/log2(rank+1)) for hits in top-10, normalized by IDCG (perfect ranking).
- Hit Rate@10: 1 if at least one test item appears in top-10 recommendations, 0 otherwise. Average across users.
- Recall@10: |relevant items in top-10| / |total relevant items for user|. Average across users.
- Global time cutoff: all interactions before cutoff → train, on/after → test. ~3 months for test set. Exclude users with no training or no test interactions.
- Evaluation at scale (200K+ users, 84K items) requires efficient computation — avoid generating full score matrices. Popularity ranking is the same for all users, so just filter per-user seen items.

## Sources
- [A Re-visit of the Popularity Baseline in Recommender Systems](https://dr.ntu.edu.sg/server/api/core/bitstreams/ab483e99-12a9-4491-867b-79e787971177/content)
- [Evaluating recommendation systems - Shaped](https://www.shaped.ai/blog/evaluating-recommendation-systems-map-mmr-ndcg)
- [recometrics: Library-agnostic evaluation for implicit-feedback recsys](https://github.com/david-cortes/recometrics)
- [Quality Metrics in Recommender Systems](https://arxiv.org/pdf/2206.12858)
- [Are We Really Making Much Progress?](https://arxiv.org/pdf/1907.06902)
