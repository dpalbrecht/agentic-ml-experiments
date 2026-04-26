# Experiment 003: XGBoost Tuned — Results

## Approach
XGBClassifier (n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, gamma=0.1) on all 50 features, no scaler, stratified 10-fold CV.

## Results
**Primary metric (Accuracy):** 0.8448 ± 0.0118
**Secondary metrics:**
- ROC-AUC: 0.8346 ± 0.0241
- Macro F1: 0.7857 ± 0.0169

## Validation Details
| Fold | Accuracy |
|------|----------|
| 01   | 0.8460   |
| 02   | 0.8360   |
| 03   | 0.8320   |
| 04   | 0.8360   |
| 05   | 0.8400   |
| 06   | 0.8440   |
| 07   | 0.8460   |
| 08   | 0.8360   |
| 09   | 0.8720   |
| 10   | 0.8600   |

## vs. Baseline
0.7186 → 0.8448 (+12.62 pp)

## vs. Best Previous
0.8432 (experiment 002) → 0.8448 (+0.16 pp) — marginal new best

## vs. Success Threshold
90% accuracy → **NOT MET** (gap: −5.52 pp)

## Observations
The slow-learner pattern yielded virtually no improvement (+0.16 pp) over default XGBoost, and fold variance only reduced slightly (±0.014 → ±0.012). The accuracy ceiling around 84–85% suggests XGBoost with these features may be saturating — either the signal ceiling for tree-based models is here, or a systematic hyperparameter search (grid/random search over depth, subsample, colsample) could unlock more. A Bayesian or random search over a wider hyperparameter space is the logical next step.

---
*Run on: 2026-04-23*
