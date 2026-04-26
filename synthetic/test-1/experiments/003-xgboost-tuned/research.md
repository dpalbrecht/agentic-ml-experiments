# Experiment 003: XGBoost Tuned — Research

## Approach Researched
XGBoost slow-learner pattern: lower learning rate + more trees, with light regularization additions.

## Key Findings
- **n_estimators=500 + learning_rate=0.05 is well-supported.** The rule of thumb is ~2-3x more trees when halving the learning rate. 500 trees at 0.05 is appropriate and not inherently risky at 5k rows.
- **Keep max_depth=6.** Dropping to 4 is a secondary lever; tune learning rate and n_estimators first. High fold variance in exp 002 is more likely noise at this sample size than overfitting.
- **Add min_child_weight=3** (default is 1) — reduces variance by requiring more samples per leaf before splitting.
- **Add gamma=0.1** (default is 0) — enforces minimum loss reduction per split, useful with more trees.
- **Keep subsample=0.8, colsample_bytree=0.8** from exp 002 — already well-regularized.

## Sources
- https://machinelearningmastery.com/tune-learning-rate-for-gradient-boosting-with-xgboost-in-python/
- https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html
- https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
