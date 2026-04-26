# Experiment 001: Lasso Logistic Regression

## Approach
L1-regularized logistic regression with cross-validated C selection (`LogisticRegressionCV`, penalty='l1', solver='liblinear', Cs=20 values from 1e-4 to 10). Wrapped in a StandardScaler pipeline. Outer evaluation via stratified 10-fold CV.

## Rationale
The baseline (L2, C=1.0) scored at the majority-class floor — 48 noise features drowned out the 2 informative ones. L1 regularization drives noise feature coefficients to exactly zero, performing automatic feature selection inside the model. The optimal C is chosen by inner CV on each training fold, so the sparsity level adapts to the data.

## What changed from previous experiments
- Penalty: L2 → L1
- Solver: lbfgs → liblinear (required for L1)
- C: fixed default (1.0) → cross-validated over logspace(-4, 1, 20)

## Evaluation
- **Primary metric**: Accuracy (mean ± std across folds)
- **Secondary metrics**: ROC-AUC, Macro F1
- **Validation**: Stratified 10-fold cross-validation (same as all experiments)
