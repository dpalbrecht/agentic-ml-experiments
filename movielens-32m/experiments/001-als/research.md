# Experiment 001: ALS — Research

## Approach Researched
Implicit ALS (Alternating Least Squares) matrix factorization for implicit feedback recommendation, using the `implicit` Python library.

## Key Findings
- iALS is one of the most computationally efficient and scalable collaborative filtering methods. With proper tuning, it outperforms many neural methods on standard benchmarks (Rendle et al., "Revisiting the Performance of iALS", RecSys 2022).
- Start with factors=128 (mid-sized embedding), then double until improvement plateaus.
- The two most critical hyperparameters are **unobserved weight (alpha)** and **regularization** — they should be tuned together as both control the trade-off between fitting observed vs. unobserved interactions.
- Frequency-scaled regularization (ν=1) becomes more important at larger embedding dimensions.
- The `implicit` library API: `AlternatingLeastSquares(factors=128, regularization=0.01, alpha=1.0, iterations=15)`. The input is a sparse user-item CSR matrix where nonzero entries represent interactions.
- Confidence is computed as `1 + alpha * value`. For binary interactions (all 1s), alpha=1.0 gives confidence=2 for positive pairs.
- The library uses optimized C++ with multithreading, making it fast even on 32M interactions.
- For scoring, use `model.recommend(userid, user_items, N=10, filter_already_liked_items=True)` to get top-K recommendations excluding already-seen items.

## Sources
- [Revisiting the Performance of iALS on Item Recommendation Benchmarks](https://arxiv.org/abs/2110.14037)
- [implicit library documentation](https://benfred.github.io/implicit/)
- [AlternatingLeastSquares API](https://benfred.github.io/implicit/api/models/cpu/als.html)
- [implicit GitHub](https://github.com/benfred/implicit)
