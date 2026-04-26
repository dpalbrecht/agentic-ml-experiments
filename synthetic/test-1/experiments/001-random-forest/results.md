# Experiment 001: Random Forest — Results

## Approach
RandomForestClassifier (n_estimators=200, max_features='sqrt', min_samples_leaf=5) on all 50 features, no scaler, stratified 10-fold CV.

## Results
**Primary metric (Accuracy):** 0.7530 ± 0.0080
**Secondary metrics:**
- ROC-AUC: 0.8137 ± 0.0268
- Macro F1: 0.5427 ± 0.0223

## Validation Details
| Fold | Accuracy |
|------|----------|
| 01   | 0.7640   |
| 02   | 0.7520   |
| 03   | 0.7440   |
| 04   | 0.7460   |
| 05   | 0.7420   |
| 06   | 0.7440   |
| 07   | 0.7600   |
| 08   | 0.7560   |
| 09   | 0.7600   |
| 10   | 0.7620   |

## vs. Baseline
0.7186 → 0.7530 (+3.44 pp)

## vs. Best Previous
0.7186 (experiment 000) → 0.7530 (+3.44 pp) — new best

## vs. Success Threshold
90% accuracy → **NOT MET** (gap: −14.70 pp)

## Observations
Random forest found meaningful non-linear signal (ROC-AUC jumped from 0.60 to 0.81), confirming the target is driven by feature interactions. However, accuracy only reached 75.3% — still 14.7 pp from the goal. Gradient boosting (XGBoost or LightGBM) typically outperforms random forest on tabular data and is the logical next step.

---
*Run on: 2026-04-23*
