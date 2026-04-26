# Experiment 004: XGBoost Random Search — Research

## Approach Researched
RandomizedSearchCV over XGBoost hyperparameter space for binary classification.

## Key Findings
- **50 iterations** is reasonable for 5k rows; covers the space well without excessive runtime.
- **5-fold inner CV** inside RandomizedSearchCV (not 10-fold) — sufficient at this sample size and much faster.
- **Search on 'roc_auc'**, not accuracy — threshold-independent, less noisy signal for guiding the search. Report final accuracy separately via 10-fold CV with best params.
- **loguniform distributions** for continuous params spanning multiple orders of magnitude (learning_rate, reg terms). Integer ranges for max_depth, min_child_weight.
- **n_jobs=1 in XGBClassifier, n_jobs=-1 in RandomizedSearchCV** — avoids thread explosion from nested parallelism.
- Final performance reported by re-running 10-fold stratified CV with the best found params, for consistency with prior experiments.

## Sources
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
- https://xgboosting.com/xgboost-configure-n_jobs-for-random-search/
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.loguniform.html
