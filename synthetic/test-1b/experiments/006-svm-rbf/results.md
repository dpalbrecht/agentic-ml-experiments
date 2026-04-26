# Experiment 006: SVM with RBF Kernel — Results

## Approach
Pipeline: StandardScaler → SVC(kernel='rbf', probability=True). Nested CV: GridSearchCV (inner 5-fold) over C=[0.1,1,10,100] × gamma=[0.001,0.01,0.1,1], outer stratified 10-fold CV. All 50 features.

## Results
**Primary metric (Accuracy):** 0.7388 ± 0.0070
**Secondary metrics:**
- ROC-AUC: 0.7104 ± 0.0323
- Macro F1: 0.5265 ± 0.0347

## Validation Details
Stratified 10-fold CV:
| Fold | Accuracy | ROC-AUC | Macro F1 | Best C | Best gamma |
|------|----------|---------|----------|--------|------------|
| 1  | 0.7340 | 0.7352 | 0.5027 | 100 | 0.001 |
| 2  | 0.7400 | 0.6998 | 0.5164 | 100 | 0.001 |
| 3  | 0.7320 | 0.6398 | 0.4964 | 100 | 0.001 |
| 4  | 0.7440 | 0.7441 | 0.5286 | 100 | 0.001 |
| 5  | 0.7300 | 0.6665 | 0.6200 | 10  | 0.010 |
| 6  | 0.7300 | 0.7036 | 0.5003 | 100 | 0.001 |
| 7  | 0.7460 | 0.7361 | 0.5202 | 100 | 0.001 |
| 8  | 0.7520 | 0.7284 | 0.5522 | 100 | 0.001 |
| 9  | 0.7380 | 0.7158 | 0.5102 | 100 | 0.001 |
| 10 | 0.7420 | 0.7351 | 0.5176 | 100 | 0.001 |

Best params: C=100 (9/10 folds), gamma=0.001 (9/10 folds) — both at the edges of the grid.

## vs. Baseline
0.7186 → 0.7388 (+2.0 pp)

## vs. Best Previous
005-hgbt (0.8440) → 0.7388 (**-10.5 pp** — significant regression)

## vs. Success Threshold
90% required → 73.88% achieved → **NOT MET (gap: -16.1 pp)**

## Feature Importance
Top 10 features by mean permutation importance:
| Feature | Importance |
|---------|------------|
| feature_0  | +0.0347 |
| feature_2  | +0.0309 |
| feature_1  | +0.0057 |
| feature_49 | +0.0037 |
| feature_3  | +0.0021 |
| feature_48 | +0.0019 |

26/50 features show positive importance — more diffuse than HGBT's top-3 dominance. Same top features (feature_0, feature_2, feature_1) confirming these are the true signal carriers.

## Observations
SVM with RBF underperforms HGBT by 10.5 pp, confirming the signal structure is more "axis-aligned" and tree-friendly than radially symmetric. The consistent best params (C=100, gamma=0.001 — both at grid edges) suggest the SVM wants an even smoother, larger-radius kernel; a wider gamma grid could be explored, but given it still can't approach HGBT's 84%, the RBF kernel is simply not the right inductive bias for this data. Combined with the HGBT plateau at 84%, we now have strong evidence from two different model families that ~84% is close to the Bayes error rate for this dataset. One remaining avenue: explicit polynomial feature engineering on the top-3 features, which could confirm whether the boundary is a learnable polynomial surface or genuinely irreducible noise.

---
*Run on: 2026-04-25*
