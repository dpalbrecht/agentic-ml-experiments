# Experiment 005: MLP — Research

## Approach Researched
sklearn MLPClassifier for binary classification on 5k rows, 50 standard-normal features.

## Key Findings
- **Hidden layers (128, 64)** preferred over (256, 128) at 5k rows — wider nets overfit without enough data. Rule of thumb: neurons between input size (50) and output (2).
- **relu activation, adam solver** — relu scales better than tanh at this dimensionality; adam converges faster than lbfgs at 5k samples.
- **alpha=0.001** (L2 regularization) as a starting point; MLP is sensitive to this. Range 1e-4 to 1e-2.
- **early_stopping=True with max_iter=2000** — default 200 iterations often insufficient; early stopping prevents overfitting while allowing convergence.
- **StandardScaler inside Pipeline** — neural nets are sensitive to feature scale even when features are ~standard-normal; scaler ensures correct fit per fold.
- **random_state=42** in both MLPClassifier and StratifiedKFold for reproducibility.

## Sources
- https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
- https://scikit-learn.org/stable/modules/neural_networks_supervised.html
- https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html
