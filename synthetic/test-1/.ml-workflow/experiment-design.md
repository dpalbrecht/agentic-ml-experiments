# Experiment Design

## Primary Metric
**Accuracy** — symmetric error costs, so raw accuracy is the right single-number summary.

## Secondary Metrics
- **ROC-AUC** — threshold-free ranking quality
- **Macro F1** — equal weight to both classes, guards against class-imbalance drift

## Validation Strategy
**Stratified 10-fold cross-validation** on the full 5,000-row dataset. Stratification preserves the 72/28 class split in every fold. Report mean ± std across folds for all metrics.

## Baseline Approach
**Logistic regression with scikit-learn defaults** (L2 penalty, C=1.0, `solver='lbfgs'`), trained on all 50 features with no preprocessing beyond what's already standard-normal. This sets the floor all future experiments must beat.

## Success Threshold
**90% accuracy** (mean CV score). Models below this threshold are not considered deployment-ready.

---
*Designed on: 2026-04-23*
