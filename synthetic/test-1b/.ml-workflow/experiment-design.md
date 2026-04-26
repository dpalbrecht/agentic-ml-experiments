# Experiment Design

## Primary Metric
**Accuracy** — errors are symmetric and we want a single interpretable number. Note: majority-class dummy scores 71.9%, so the effective floor is already high.

## Secondary Metrics
- **ROC-AUC** — threshold-free ranking quality
- **Macro F1** — monitors per-class balance

## Validation Strategy
**Stratified 10-fold cross-validation** — preserves the 72/28 class ratio in each fold. Report mean ± std across folds. All experiments must use this same split strategy for results to be comparable.

## Baseline Approach
**Logistic regression with scikit-learn defaults** (`LogisticRegression()`) trained on all 50 features. No hyperparameter tuning, no feature selection. This is the floor all future experiments must beat.

## Success Threshold
**90% accuracy** (mean CV score). Models below this threshold are not worth deploying.

---
*Designed on: 2026-04-23*
