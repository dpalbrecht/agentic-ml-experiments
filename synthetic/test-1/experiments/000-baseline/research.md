# Experiment 000: Baseline — Research

## Approach Researched
Logistic regression (L2, lbfgs) for binary classification on standard-normal features.

## Key Findings
- **Increase max_iter to 1000.** Default of 100 frequently causes convergence warnings with lbfgs on moderately sized datasets. 1000 is safe without being excessive.
- **Wrap in a Pipeline with StandardScaler.** Even though features are already ~standard-normal, fitting the scaler inside the pipeline ensures no leakage across CV folds and protects against subtle fold-level distribution drift.
- **Use `cross_validate()` with a scoring dict** to get accuracy, ROC-AUC, and macro F1 in a single pass.
- **Set `random_state=42` on StratifiedKFold** for reproducibility. `cross_validate` uses stratified splits automatically for classifiers, but explicit StratifiedKFold gives full control.
- No other gotchas: lbfgs is stable for dense float features and works cleanly with stratified CV.

## Sources
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- https://scikit-learn.org/stable/modules/preprocessing.html
- https://scikit-learn.org/stable/modules/cross_validation.html
- https://scikit-learn.org/stable/modules/model_evaluation.html
