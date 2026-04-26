# Experiment 009: HGBT + Full Degree-2 Polynomial Features

## Approach
Append all 6 degree-2 terms from {feature_0, feature_1, feature_2} to the original 50 features (56 total): f0², f0·f1, f0·f2, f1², f1·f2, f2². HGBT with best settings (max_depth=5, lr=0.1, max_iter=2000, min_samples_leaf=20, early stopping).

## Rationale
Experiment 008 added only the top-2 polynomial terms (f0·f2, f1²) and gained +3.08 pp, breaking the 84% plateau and reaching 87.48%. Experiment 007 showed the remaining terms (f0², f2², f0·f1, f1·f2) also carry non-trivial signal (coefficients 0.09–0.29 vs 1.06–1.37 for top-2). Adding all 6 completes the degree-2 expansion and closes the remaining gap to 90%.

## What changed from experiment 008
- Added 4 more polynomial features: f0², f0·f1, f1·f2, f2² (completing the full degree-2 expansion)
- Total: 50 original + 6 poly = 56 features (vs 52 in exp 008)
- Everything else identical

## Evaluation
- **Primary metric**: Accuracy (mean ± std, outer 10-fold CV)
- **Secondary metrics**: ROC-AUC, Macro F1
- **Validation**: Stratified 10-fold cross-validation (same as all experiments)
