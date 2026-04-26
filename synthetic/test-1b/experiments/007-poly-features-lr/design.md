# Experiment 007: Polynomial Features + Logistic Regression

## Approach
Select top-3 features (feature_0, feature_1, feature_2) → PolynomialFeatures(degree=2, include_bias=False) → StandardScaler → LogisticRegressionCV (L2, inner 5-fold, C grid logspace(-3,3,20)). This produces 9 features: 3 originals + 3 squares + 3 cross-products.

## Rationale
HGBT has plateaued at 84% across all tuning attempts and feature subsets. The signal lives in feature_0, feature_1, feature_2 (consistent across all permutation importance runs). If the true decision boundary is a degree-2 polynomial surface (e.g., f0² + f2² > c, or f0·f2 > c), logistic regression on these 9 features can represent it exactly. This is a direct test of that hypothesis. If it breaks through 84%, the boundary is quadratic and identifiable. If not, 84% is very likely the irreducible noise floor.

## What changed from previous experiments
- Feature engineering: raw features → degree-2 polynomial expansion of top-3
- Model: HGBT → logistic regression on 9 engineered features
- Combines the feature selection insight (top-3) with explicit non-linear feature construction

## Evaluation
- **Primary metric**: Accuracy (mean ± std, outer 10-fold CV)
- **Secondary metrics**: ROC-AUC, Macro F1
- **Validation**: Stratified 10-fold outer CV (same as all experiments)
