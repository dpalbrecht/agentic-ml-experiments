# Experiment 000: Baseline — Research

## Approach Researched
Logistic regression for imbalanced binary classification with time-based cross-validation.

## Key Findings
- **class_weight='balanced'** adjusts weights inversely proportional to class frequencies: `n_samples / (n_classes * np.bincount(y))`. Essential for 6.68:1 imbalance — without it, the model predicts nearly all-negative and F2 collapses to ~0.
- **Feature scaling is required** for logistic regression since it uses L2 regularization by default. Features like tenure_days (range 30–3828) and noise columns (range ~-4 to +4) are on very different scales. StandardScaler is standard practice.
- **TimeSeriesSplit** from sklearn implements expanding window CV natively. Data must be sorted chronologically first. Each fold trains on all prior data and tests on the next window. Successive training sets are supersets of earlier ones.
- **Default solver** is 'lbfgs', max_iter=100. May need to increase max_iter if convergence warnings appear.
- **Threshold tuning** is an alternative to class weighting for imbalanced problems, but class_weight='balanced' is simpler and appropriate for a baseline.

## Sources
- [LogisticRegression — scikit-learn 1.8.0](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [TimeSeriesSplit — scikit-learn 1.8.0](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [Post-tuning the decision threshold for cost-sensitive learning — scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cost_sensitive_learning.html)
- [How to improve logistic regression in imbalanced data with class weights](https://medium.com/@data.science.enthusiast/how-to-improve-logistic-regression-in-imbalanced-data-with-class-weights-1693719136aa)
