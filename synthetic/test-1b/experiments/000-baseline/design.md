# Experiment 000: Baseline

## Approach
Logistic regression with scikit-learn defaults (`LogisticRegression()`, solver=lbfgs, C=1.0, L2 penalty) wrapped in a `Pipeline` with `StandardScaler`. Trained on all 50 features.

## Rationale
Establishes the floor. All future experiments must beat this.

## Evaluation
- **Primary metric**: Accuracy (mean ± std across folds)
- **Secondary metrics**: ROC-AUC, Macro F1
- **Validation**: Stratified 10-fold cross-validation
