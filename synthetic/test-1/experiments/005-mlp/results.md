# Experiment 005: MLP — Results

## Approach
MLPClassifier (hidden_layer_sizes=(128, 64), relu, adam, alpha=0.001, early_stopping=True, max_iter=2000) in a StandardScaler pipeline, stratified 10-fold CV.

## Results
**Primary metric (Accuracy):** 0.7546 ± 0.0197
**Secondary metrics:**
- ROC-AUC: 0.7282 ± 0.0349
- Macro F1: 0.6399 ± 0.0509

## Validation Details
| Fold | Accuracy |
|------|----------|
| 01   | 0.7240   |
| 02   | 0.7720   |
| 03   | 0.7440   |
| 04   | 0.7540   |
| 05   | 0.7480   |
| 06   | 0.7580   |
| 07   | 0.7640   |
| 08   | 0.7760   |
| 09   | 0.7220   |
| 10   | 0.7840   |

## vs. Baseline
0.7186 → 0.7546 (+3.60 pp)

## vs. Best Previous
0.8498 (experiment 004) → 0.7546 (−9.52 pp) — worse than XGBoost

## vs. Success Threshold
90% accuracy → **NOT MET** (gap: −14.54 pp)

## Observations
MLP substantially underperformed XGBoost (75.5% vs 85%), suggesting the interaction structure in this dataset is better captured by tree-based splits than continuous representations. High fold variance (±0.0197 vs ±0.014 for XGBoost) indicates the MLP is also less stable. The XGBoost family remains the best approach; a stacking ensemble combining XGBoost with other models, or a more exhaustive hyperparameter search with a larger MLP, could be explored next.

---
*Run on: 2026-04-23*
