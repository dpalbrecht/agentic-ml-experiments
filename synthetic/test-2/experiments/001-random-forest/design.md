# Experiment 001: Random Forest

## Approach
Random Forest classifier with `class_weight='balanced_subsample'`, `n_estimators=500`, otherwise default hyperparameters. No feature scaling (trees don't need it). Same 11 features as baseline.

## Rationale
Baseline logistic regression scores F2=0.9091 but barely clears the 0.90 threshold, with fold 1 dipping to 0.8798. Signal is concentrated in 3 features, and a linear model can only capture linear relationships. Random Forest can model non-linear feature-target relationships and feature interactions (e.g., low product_usage_score combined with high days_since_last_login) that logistic regression misses.

## What changed from previous experiments
- Model: LogisticRegression → RandomForestClassifier
- class_weight: 'balanced' → 'balanced_subsample' (per-bootstrap adjustment)
- Removed StandardScaler (unnecessary for trees)
- Added permutation importance alongside impurity-based importance for reliable feature selection

## Evaluation
- **Primary metric:** F2 score
- **Secondary metrics:** Recall, Precision
- **Validation:** Sliding window temporal CV — 12-month train, 1-month test, 6 folds (test months Jul–Dec 2023)
