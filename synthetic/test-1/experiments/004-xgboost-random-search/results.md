# Experiment 004: XGBoost Random Search — Results

## Approach
RandomizedSearchCV (50 iterations, 5-fold inner CV, scoring='roc_auc') over XGBoost hyperparameter space. Best params evaluated with 10-fold stratified CV for comparison with prior experiments.

## Best Params Found
- colsample_bytree: 0.806, gamma: 0.00176, learning_rate: 0.0102
- max_depth: 6, min_child_weight: 7, n_estimators: 400, subsample: 0.860
- Best inner CV ROC-AUC: 0.8428

## Results
**Primary metric (Accuracy):** 0.8498 ± 0.0135
**Secondary metrics:**
- ROC-AUC: 0.8447 ± 0.0263
- Macro F1: 0.7888 ± 0.0198

## Validation Details
| Fold | Accuracy |
|------|----------|
| 01   | 0.8360   |
| 02   | 0.8580   |
| 03   | 0.8400   |
| 04   | 0.8320   |
| 05   | 0.8480   |
| 06   | 0.8580   |
| 07   | 0.8420   |
| 08   | 0.8420   |
| 09   | 0.8760   |
| 10   | 0.8660   |

## vs. Baseline
0.7186 → 0.8498 (+13.12 pp)

## vs. Best Previous
0.8448 (experiment 003) → 0.8498 (+0.50 pp) — marginal new best

## vs. Success Threshold
90% accuracy → **NOT MET** (gap: −5.02 pp)

## Observations
Systematic search over 50 random combinations yielded only +0.5 pp over manual tuning — XGBoost appears to have a hard ceiling around 84–85% on this dataset. The consistent fold 9/10 outliers (87–88%) persist across all XGBoost runs, suggesting this isn't a hyperparameter problem but a model-family limitation. A different model family (LightGBM, MLP, or stacking) is the logical next step.

---
*Run on: 2026-04-23*
