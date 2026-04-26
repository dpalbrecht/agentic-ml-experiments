# Experiment 006: SVM with RBF Kernel

## Approach
Pipeline: StandardScaler → SVC(kernel='rbf', probability=True). Nested CV: GridSearchCV (inner 5-fold) over C=[0.1, 1, 10, 100] × gamma=[0.001, 0.01, 0.1, 1] (16 combos), evaluated with outer stratified 10-fold CV. All 50 features. Permutation importance on each held-out fold.

## Rationale
Four HGBT variants all plateau at 84.0–84.4%, exhausting what boosted trees can do. SVM with RBF kernel finds a maximum-margin decision boundary in a kernel-induced feature space — a fundamentally different inductive bias. If it also caps at ~84%, that's strong evidence the ceiling is the dataset's Bayes error rate. If it breaks through, it reveals structure HGBT couldn't find.

## What changed from previous experiments
- Entirely different model family: HGBT → SVM with RBF kernel
- StandardScaler re-introduced (required for distance-based kernel)
- Hyperparameter search: C and gamma (not depth/lr)

## Evaluation
- **Primary metric**: Accuracy (mean ± std, outer 10-fold CV)
- **Secondary metrics**: ROC-AUC, Macro F1
- **Validation**: Stratified 10-fold outer CV (same as all experiments)
