# Experiment 002: XGBoost — Research

## Approach Researched
XGBClassifier for binary classification on 5k rows, 50 standard-normal features, 72/28 split.

## Key Findings
- **Use XGBClassifier** (xgboost package) over sklearn GradientBoostingClassifier — faster, better regularization, industry standard.
- **Starting hyperparameters:** n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8.
- **No scaling needed** — tree-based, scale-invariant.
- **Set verbosity=0** to suppress XGBoost's per-tree console output inside cross_validate.
- **random_state=42** for reproducibility.
- Do not use eval_set or early_stopping_rounds inside CV loops — keep it simple.

## Sources
- https://xgboosting.com/xgbclassifier-faster-than-gradientboostingclassifier/
- https://forecastegy.com/posts/does-xgboost-need-feature-scaling-or-normalization/
- https://xgboosting.com/configure-xgboost-random_state-parameter/
