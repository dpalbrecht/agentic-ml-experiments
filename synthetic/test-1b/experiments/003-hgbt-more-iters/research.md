# Experiment 003: HGBT More Iterations — Research

## Approach Researched
Continuing from experiment 002: all folds hit the 500-iteration cap and early stopping never fired, indicating the model was still improving. Increasing max_iter to allow full convergence.

## Key Findings
- When early stopping (n_iter_no_change=10) never fires, it means validation loss improved by more than `tol=1e-4` in every window of 10 iterations — the model was actively learning throughout all 500 rounds.
- Increasing max_iter to 2000 gives early stopping room to fire naturally when improvement genuinely stalls, rather than being cut off by an artificial cap.
- All other hyperparameters are unchanged — this is a single-variable change to isolate the effect of more iterations.
- No additional research needed beyond experiment 002 findings.

## Sources
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
