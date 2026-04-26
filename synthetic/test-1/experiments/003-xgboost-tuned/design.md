# Experiment 003: XGBoost Tuned

## Approach
XGBClassifier with slow-learner settings: n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, gamma=0.1.

## Rationale
Experiment 002 reached 84.3% with default XGBoost — 5.7 pp from the goal with notable fold variance (82–87%). The slow-learner pattern (more trees, lower rate) plus light regularization (min_child_weight, gamma) is the standard next step when default gradient boosting is close but not there.

## What changed from previous experiments
- learning_rate: 0.1 → 0.05
- n_estimators: 100 → 500
- min_child_weight: 1 (default) → 3
- gamma: 0 (default) → 0.1

## Evaluation
- **Primary metric:** Accuracy (mean ± std across folds)
- **Secondary metrics:** ROC-AUC, Macro F1
- **Validation:** Stratified 10-fold cross-validation, random_state=42
