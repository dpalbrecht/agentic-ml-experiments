# Experiment 006: RF Hyperparameter Tuning — Research

## Approach Researched
Tuning Random Forest hyperparameters (max_depth, min_samples_leaf, max_features) using OOB F2 score for selection.

## Key Findings
- **max_depth**: Controls tree depth. Default is None (unlimited, trees grow to purity). Constraining depth (e.g., 8–20) can reduce overfitting to noise features. Most impactful single hyperparameter.
- **min_samples_leaf**: Minimum samples at a leaf. Default is 1 (pure leaves). Increasing to 5–20 regularizes by preventing tiny leaves that memorize noise.
- **max_features**: Features considered per split. Default for classification is 'sqrt' (~3.3 for 11 features). Setting to None (all features) or a higher fraction increases per-tree performance but decreases diversity. With many noise features, 'sqrt' may be near-optimal since it forces the model to frequently consider signal features.
- **OOB for hyperparameter selection**: Fit each combo with `oob_score=True`, use OOB predictions to compute F2 (with threshold tuning). This avoids an inner CV loop and doesn't sacrifice training data.
- **n_estimators**: 500 trees is sufficient — diminishing returns beyond this. Use 200 trees during grid search for speed, 500 for the final model.

## Sources
- [RandomForestClassifier — scikit-learn 1.8.0](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Random Forest Hyperparameter Tuning in Python — GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/random-forest-hyperparameter-tuning-in-python/)
- [Hyperparameter Tuning the Random Forest — Towards Data Science](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74/)
