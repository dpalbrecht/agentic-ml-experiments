# Experiment 006: RF Hyperparameter Tuning

## Approach
Random Forest with OOB threshold tuning (same as experiment 002), plus hyperparameter grid search over max_depth, min_samples_leaf, and max_features. Hyperparameters selected per-fold by maximizing OOB F2 score (with threshold tuning). 11 features.

Grid:
- max_depth: [None, 8, 12, 20]
- min_samples_leaf: [1, 5, 10, 20]
- max_features: ['sqrt', 0.5, None]
- Total: 48 combinations per fold

200 trees during search for speed, 500 trees for final model with best params.

## Rationale
Experiment 002 (RF + OOB threshold, defaults) is the best model at F2=0.9219. It uses default hyperparameters — constraining tree depth or leaf size could reduce overfitting to the 8 noise/weak features and improve the weakest fold (Jul 2023, F2=0.8734).

## What changed from previous experiments
- Experiment 002 → 006: Hyperparameter tuning added. Same OOB threshold approach, same features, same CV.

## Evaluation
- **Primary metric:** F2 score
- **Secondary metrics:** Recall, Precision
- **Validation:** Sliding window temporal CV — 12-month train, 1-month test, 6 folds (test months Jul–Dec 2023)
