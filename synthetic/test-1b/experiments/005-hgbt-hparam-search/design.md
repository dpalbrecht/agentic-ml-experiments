# Experiment 005: HGBT Hyperparameter Search

## Approach
Nested CV: GridSearchCV (inner 5-fold) over max_depth × learning_rate, evaluated with outer stratified 10-fold CV. All 50 features. Grid:
- `max_depth`: [3, 4, 5]
- `learning_rate`: [0.01, 0.05, 0.1]
- `max_iter`: 2000 (with early stopping throughout)
- All other params unchanged (min_samples_leaf=20, n_iter_no_change=10, validation_fraction=0.1)

## Rationale
All four prior HGBT experiments plateau at ~84% with early stopping never firing. max_depth=7 is likely too deep — the model keeps splitting on noise rather than converging. A grid search over depth and learning rate with proper nested CV will find the combination that generalises best, while inner CV ensures hyperparameter selection never sees the outer test fold.

## What changed from previous experiments
- Hyperparameter tuning via nested GridSearchCV instead of fixed values
- max_depth candidates: [3, 4, 5] (shallower than prior 7)
- learning_rate candidates: [0.01, 0.05, 0.1]

## Evaluation
- **Primary metric**: Accuracy (mean ± std, outer 10-fold CV)
- **Secondary metrics**: ROC-AUC, Macro F1
- **Validation**: Stratified 10-fold outer CV (same as all experiments)
