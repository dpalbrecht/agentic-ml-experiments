# Experiment 005: MLP

## Approach
sklearn MLPClassifier: hidden_layer_sizes=(128, 64), activation='relu', solver='adam', alpha=0.001, early_stopping=True, max_iter=2000. Wrapped in StandardScaler pipeline.

## Rationale
Three XGBoost variants (002–004) converged to a hard ceiling at ~85% regardless of hyperparameters. Decision-tree ensembles express interactions through axis-aligned splits; an MLP learns continuous representations that can express different interaction structures. A different model family is the logical next step.

## What changed from previous experiments
- Entirely new model family: gradient-boosted trees → 2-layer neural network
- StandardScaler pipeline re-introduced (MLP is not scale-invariant)
- Early stopping used instead of fixed n_estimators

## Evaluation
- **Primary metric:** Accuracy (mean ± std across folds)
- **Secondary metrics:** ROC-AUC, Macro F1
- **Validation:** Stratified 10-fold cross-validation, random_state=42
