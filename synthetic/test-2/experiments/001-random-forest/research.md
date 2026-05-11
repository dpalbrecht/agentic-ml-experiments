# Experiment 001: Random Forest — Research

## Approach Researched
Random Forest classifier for imbalanced binary churn classification.

## Key Findings
- **`class_weight='balanced_subsample'`** adjusts weights inversely proportional to class frequencies in each bootstrap sample (not the full dataset). Preferred over `'balanced'` for Random Forest since each tree sees a different bootstrap sample.
- **`n_estimators` default is 100** in current sklearn. Increasing to 500 gives more stable predictions with diminishing returns beyond that.
- **Impurity-based feature importances are biased** toward high-cardinality numerical features — random numerical variables can appear more important than they are. For reliable feature selection, permutation importance should also be computed.
- **Default tree depth (None)** grows trees to single-item leaves, which can overfit. However, for an initial RF experiment we keep defaults and change one thing (model) relative to baseline.
- **Collinearity spreads permutation importance** across correlated features. Less of a concern here since no feature pairs exceed r=0.95.

## Sources
- [RandomForestClassifier — scikit-learn 1.8.0](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Bagging and Random Forest for Imbalanced Classification](https://machinelearningmastery.com/bagging-and-random-forest-for-imbalanced-classification/)
- [Best Practice to Calculate and Interpret Model Feature Importance](https://towardsdatascience.com/best-practice-to-calculate-and-interpret-model-feature-importance-14f0e11ee660/)
- [Random Forest Hyperparameters & Tuning Strategies](https://baekholab.com/2025/03/22/part-6-random-forest-hyperparameters-tuning-strategies/)
