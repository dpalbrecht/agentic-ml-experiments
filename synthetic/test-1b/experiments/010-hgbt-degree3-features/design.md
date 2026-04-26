# Experiment 010: HGBT + Degree-3 Polynomial Features

## Approach
Append all 10 degree-3 terms from {feature_0, feature_1, feature_2} to the 56-feature set from experiment 009 (66 total). HGBT with identical settings (max_depth=5, lr=0.1, max_iter=2000, min_samples_leaf=20, early stopping).

Degree-3 terms added: f0³, f0²·f1, f0²·f2, f0·f1², f0·f1·f2, f0·f2², f1³, f1²·f2, f1·f2², f2³.

## Rationale
The degree-2 expansion is exhausted — adding 4 more degree-2 terms in experiment 009 yielded only +0.14 pp. Three folds already exceed 90% (max 91.0%), confirming the Bayes limit is above 90% and the remaining 2.38 pp gap is a model capacity issue. The dominant signal terms (f0·f2 and f1²) suggest a boundary that may involve higher-order interactions. Degree-3 terms — particularly f0·f1·f2 and f1³ — are the natural next candidates.

## What changed from experiment 009
- Added 10 degree-3 terms from {feature_0, feature_1, feature_2}
- Total: 56 (exp 009) + 10 = 66 features
- Everything else identical

## Evaluation
- **Primary metric**: Accuracy (mean ± std, outer 10-fold CV)
- **Secondary metrics**: ROC-AUC, Macro F1
- **Validation**: Stratified 10-fold cross-validation (same as all experiments)
