# Experiment 001: Random Forest

## Approach
RandomForestClassifier (n_estimators=200, max_features='sqrt', min_samples_leaf=5, n_jobs=-1) on all 50 features, no scaler.

## Rationale
Logistic regression found no linear signal. Random forest handles non-linear feature interactions natively and works well with defaults on this dataset size.

## What changed from previous experiments
- Model: logistic regression → random forest
- No StandardScaler (RF is scale-invariant)
- n_estimators=200 for stability over default 100

## Evaluation
- **Primary metric:** Accuracy (mean ± std across folds)
- **Secondary metrics:** ROC-AUC, Macro F1
- **Validation:** Stratified 10-fold cross-validation, random_state=42
