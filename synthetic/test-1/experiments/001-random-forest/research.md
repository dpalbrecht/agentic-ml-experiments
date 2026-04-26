# Experiment 001: Random Forest — Research

## Approach Researched
RandomForestClassifier for binary classification on 5k rows, 50 standard-normal features, 72/28 split.

## Key Findings
- **n_estimators=200** preferred over default 100 for better stability at this dataset size.
- **max_features='sqrt'** (≈7 features per split) is well-suited for 50 features — good regularization.
- **min_samples_leaf=5** recommended for 5k rows to prevent overfitting via isolated leaves.
- **No scaler needed** — RandomForest is scale-invariant; raw features are fine.
- **n_jobs=-1** in both the classifier and cross_validate for full parallelism.
- max_depth: leave unbounded initially; only constrain if overfitting appears in train vs. CV gap.
- Absence of linear signal is *ideal* for RandomForest — it thrives on non-linear splits.

## Sources
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- https://machinelearningmastery.com/bagging-and-random-forest-for-imbalanced-classification/
- https://andrewpwheeler.com/2022/10/10/hyperparameter-tuning-for-random-forests/
