# Experiment 000: Baseline

## Approach
Logistic regression with scikit-learn defaults (L2 penalty, C=1.0, solver='lbfgs', max_iter=1000), wrapped in a StandardScaler pipeline. Trained on all 50 features.

## Rationale
Establishes the floor. All future experiments must beat this.

## Evaluation
- **Primary metric:** Accuracy (mean ± std across folds)
- **Secondary metrics:** ROC-AUC, Macro F1
- **Validation:** Stratified 10-fold cross-validation, random_state=42
