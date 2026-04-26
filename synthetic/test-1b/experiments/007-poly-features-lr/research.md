# Experiment 007: Polynomial Features + LR — Research

## Approach Researched
Degree-2 polynomial feature expansion of top-3 features followed by L2 logistic regression.

## Key Findings
- `sklearn.preprocessing.PolynomialFeatures(degree=2, include_bias=False)` generates exactly: 3 original + 3 squares + 3 cross-products = 9 features from 3 inputs.
- Pipeline order: select top-3 → PolynomialFeatures(degree=2) → StandardScaler → LogisticRegressionCV. Scaler must come after polynomial expansion (so interactions are scaled, not raw features).
- `LogisticRegressionCV` with inner stratified 5-fold and C grid logspace(-3, 3, 20) handles regularization selection cleanly within each outer fold.
- No external research needed — this is diagnostic sklearn plumbing.

## Sources
- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
