# Experiment 005: HGBT Hyperparameter Search — Results

## Approach
Nested CV: GridSearchCV (inner 5-fold) over max_depth=[3,4,5] × learning_rate=[0.01,0.05,0.1], evaluated with outer stratified 10-fold CV. All 50 features, max_iter=2000, early stopping unchanged.

## Results
**Primary metric (Accuracy):** 0.8440 ± 0.0108
**Secondary metrics:**
- ROC-AUC: 0.8333 ± 0.0261
- Macro F1: 0.7862 ± 0.0144

## Validation Details
Stratified 10-fold CV:
| Fold | Accuracy | ROC-AUC | Macro F1 | Best depth | Best lr |
|------|----------|---------|----------|------------|---------|
| 1  | 0.8520 | 0.8899 | 0.7971 | 5 | 0.01 |
| 2  | 0.8400 | 0.8257 | 0.7832 | 5 | 0.10 |
| 3  | 0.8320 | 0.8031 | 0.7657 | 5 | 0.01 |
| 4  | 0.8340 | 0.8134 | 0.7781 | 5 | 0.10 |
| 5  | 0.8340 | 0.8038 | 0.7731 | 5 | 0.10 |
| 6  | 0.8500 | 0.8170 | 0.7950 | 5 | 0.10 |
| 7  | 0.8420 | 0.8592 | 0.7865 | 5 | 0.01 |
| 8  | 0.8360 | 0.8260 | 0.7699 | 4 | 0.01 |
| 9  | 0.8680 | 0.8457 | 0.8137 | 5 | 0.01 |
| 10 | 0.8520 | 0.8492 | 0.7994 | 5 | 0.10 |

Best params: max_depth=5 (9/10 folds), learning_rate split 5×0.01 / 5×0.1 — no clear winner on lr.

## vs. Baseline
0.7186 → 0.8440 (+12.5 pp)

## vs. Best Previous
003-hgbt (0.8436) → 0.8440 (**+0.04 pp** — negligible)

## vs. Success Threshold
90% required → 84.40% achieved → **NOT MET (gap: -5.60 pp)**

## Feature Importance
Not computed (GridSearchCV wrapper). See experiments 002/003 for consistent permutation importance results (feature_0, feature_2, feature_1 dominate).

## Observations
The grid search found max_depth=5 almost unanimously (9/10 folds) — confirming depth=7 was mildly over-specified but depth=3/4 are too shallow. However, the accuracy gain over fixed depth=7 is essentially zero (+0.04 pp). We have now tried 4 HGBT configurations across different depths, iteration counts, feature sets, and learning rates, all converging to 84.0–84.4%. This strongly suggests ~84% is the practical ceiling for this model family on this data — either the Bayes error rate for the dataset, or a fundamental limit of the HGBT inductive bias. The next experiment should try a completely different model family (e.g., SVM with RBF kernel) to test whether the ceiling is data-specific or model-specific.

---
*Run on: 2026-04-25*
