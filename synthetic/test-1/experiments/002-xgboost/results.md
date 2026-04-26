# Experiment 002: XGBoost — Results

## Approach
XGBClassifier (n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8) on all 50 features, no scaler, stratified 10-fold CV.

## Results
**Primary metric (Accuracy):** 0.8432 ± 0.0140
**Secondary metrics:**
- ROC-AUC: 0.8377 ± 0.0230
- Macro F1: 0.7833 ± 0.0203

## Validation Details
| Fold | Accuracy |
|------|----------|
| 01   | 0.8460   |
| 02   | 0.8420   |
| 03   | 0.8320   |
| 04   | 0.8200   |
| 05   | 0.8340   |
| 06   | 0.8400   |
| 07   | 0.8440   |
| 08   | 0.8400   |
| 09   | 0.8720   |
| 10   | 0.8620   |

## vs. Baseline
0.7186 → 0.8432 (+12.46 pp)

## vs. Best Previous
0.7530 (experiment 001) → 0.8432 (+9.02 pp) — new best

## vs. Success Threshold
90% accuracy → **NOT MET** (gap: −5.68 pp)

## Observations
XGBoost made a large jump (+9 pp over random forest), confirming gradient boosting is the right model family. At 5.68 pp from the goal with notable fold variance (0.82–0.87), hyperparameter tuning — particularly more estimators, lower learning rate, or a grid search over depth/subsample — is the logical next step.

---
*Run on: 2026-04-23*
