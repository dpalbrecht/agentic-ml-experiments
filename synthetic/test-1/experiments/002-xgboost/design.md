# Experiment 002: XGBoost

## Approach
XGBClassifier (n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8) on all 50 features, no scaler.

## Rationale
Random forest confirmed strong non-linear signal (ROC-AUC 0.81) but accuracy stalled at 75.3%. Gradient boosting corrects residuals sequentially and consistently outperforms random forest on tabular data.

## What changed from previous experiments
- Model: random forest → XGBoost gradient boosting
- Sequential tree building instead of parallel bagging
- Additional regularization via subsample and colsample_bytree

## Evaluation
- **Primary metric:** Accuracy (mean ± std across folds)
- **Secondary metrics:** ROC-AUC, Macro F1
- **Validation:** Stratified 10-fold cross-validation, random_state=42
