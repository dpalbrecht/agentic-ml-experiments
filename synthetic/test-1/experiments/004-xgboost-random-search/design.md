# Experiment 004: XGBoost Random Search

## Approach
RandomizedSearchCV (50 iterations, 5-fold inner CV, scoring='roc_auc') over XGBoost hyperparameter space:
- max_depth: [3, 4, 5, 6, 7, 8]
- learning_rate: loguniform(0.01, 0.3)
- n_estimators: [100, 200, 300, 400, 500]
- subsample: uniform(0.6, 0.4)
- colsample_bytree: uniform(0.5, 0.5)
- min_child_weight: [1, 2, 3, 5, 7, 10]
- gamma: loguniform(0.001, 1.0)

Best params then evaluated with 10-fold stratified CV for a fair comparison with prior experiments.

## Rationale
Two manual XGBoost runs landed in the same 84–85% band. Folds 9 and 10 consistently hit 87%+, indicating the signal is extractable — manual tuning just hasn't found the right combination. Systematic random search covers the space properly.

## What changed from previous experiments
- Search strategy: manual → RandomizedSearchCV (50 iterations)
- Inner CV: 5-fold for search efficiency
- Outer CV: same 10-fold for final estimate (consistent with all prior experiments)

## Evaluation
- **Primary metric:** Accuracy (mean ± std across folds)
- **Secondary metrics:** ROC-AUC, Macro F1
- **Validation:** Stratified 10-fold cross-validation, random_state=42
