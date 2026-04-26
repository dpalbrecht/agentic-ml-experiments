# Experiment 006: SVM RBF — Research

## Approach Researched
SVC with RBF kernel for binary classification, nested CV hyperparameter tuning.

## Key Findings
- **StandardScaler required**: Even though features are ~N(0,1), the scaler must be fit on training folds only to prevent data leakage — distance-based kernels are sensitive to scale. Use Pipeline.
- **C & gamma grid**: `C=[0.1, 1, 10, 100]` × `gamma=[0.001, 0.01, 0.1, 1]` (16 combos). Logarithmic spacing covers the diagonal where good RBF models typically live. If best params sit on edges, extend grid.
- **Speed**: 16 combos × 5 inner folds × 10 outer folds = 800 fits. Practical at 5K rows.
- **Noise robustness**: RBF degrades gracefully with irrelevant features (won't catastrophically fail), but high C risks overfitting to noise — monitor outer CV variance.
- **Feature importance**: Permutation importance is the only option for RBF SVC (no coefficient access). Compute on each outer held-out fold.
- **probability=True** required for ROC-AUC scoring via `predict_proba`.

## Sources
- https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
- https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
