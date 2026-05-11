# Agentic-ML Experiments

Experiment runs using the [agentic-ml](https://github.com/dpalbrecht/agentic-ml) workflow. Each directory is a self-contained project with its own data, experiments, and results.

## Runs

| Directory | Problem | Dataset | Outcome |
|-----------|---------|---------|---------|
| [synthetic/test-1b](synthetic/test-1b/README.md) | Binary classification, known ground truth | Synthetic (5K rows, 50 features, 3 real) | Found true decision boundary; plateaued at 87.8% vs 90% ceiling |
| [synthetic/test-2](synthetic/test-2/README.md) | Churn prediction, temporal leakage trap | Synthetic (8K rows, 24 months, 2 leakage features) | Caught leakage; met F2 ≥ 0.90 threshold (experiment 006, tuned RF) |
| [iris](iris/README.md) | 3-class classification | Iris (150 rows, 4 features) | Met 97% threshold (experiment 004, linear SVM) |
| [movielens-32m](movielens-32m/README.md) | Top-N recommendation | MovieLens 32M (32M interactions) | Met NDCG@10 threshold at +84% over popularity baseline (ALS) |

## Purpose

These runs serve two purposes: validating that the workflow produces reasonable results on known benchmarks, and stress-testing whether the workflow's structure surfaces the right information for a human to make good decisions at each phase. The synthetic tests have a known ground truth ceiling, making it possible to diagnose exactly where the workflow helped or failed. The benchmark datasets (iris, movielens) test the workflow on real problems without a known ceiling.

Findings feed back into the workflow templates — not as new automation, but as structural improvements to the questions asked at each phase.
