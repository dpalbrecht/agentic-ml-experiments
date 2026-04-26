# Experiment 000: Baseline — Results

## Approach
Logistic regression (L2, C=1.0, lbfgs, max_iter=1000) in a StandardScaler pipeline, trained on all 50 features, evaluated with stratified 10-fold CV.

## Results
**Primary metric (Accuracy):** 0.7186 ± 0.0054
**Secondary metrics:**
- ROC-AUC: 0.5990 ± 0.0224
- Macro F1: 0.4503 ± 0.0107

## Validation Details
| Fold | Accuracy |
|------|----------|
| 01   | 0.7260   |
| 02   | 0.7200   |
| 03   | 0.7180   |
| 04   | 0.7160   |
| 05   | 0.7160   |
| 06   | 0.7140   |
| 07   | 0.7260   |
| 08   | 0.7180   |
| 09   | 0.7080   |
| 10   | 0.7240   |

## Interpretation
Logistic regression scored 71.86% — essentially the majority-class floor (71.9%). ROC-AUC of 0.599 is barely above chance (0.5), and macro F1 of 0.450 confirms the model is failing to learn class 1 meaningfully. The features have near-zero linear signal: the target is not linearly separable in this feature space.

## vs. Success Threshold
90% accuracy → **NOT MET** (gap: −18.14 pp). This is expected for the baseline; the gap will need to be closed by non-linear models.

---
*Run on: 2026-04-23*
