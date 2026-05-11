# Experiment 006: RF Hyperparameter Tuning — Results

## Approach
Random Forest with OOB-based hyperparameter selection and PR curve threshold tuning. Per-fold grid search over max_depth, min_samples_leaf, max_features (48 combos). 200 trees during search, 500 trees for final model. 11 features.

## Results
**Primary metric (F2): 0.9271 ± 0.0229**
**Secondary metrics:** Recall: 0.9602 ± 0.0239 | Precision: 0.8175 ± 0.0530

## Validation Details

| Fold | Test Month | Threshold | F2 | Recall | Precision | Best Params |
|---|---|---|---|---|---|---|
| 1 | 2023-07 | 0.26 | 0.8811 | 0.9302 | 0.7273 | depth=20, leaf=1, feat=0.5 |
| 2 | 2023-08 | 0.20 | 0.9506 | 1.0000 | 0.7937 | depth=20, leaf=1, feat=sqrt |
| 3 | 2023-09 | 0.23 | 0.9430 | 0.9773 | 0.8269 | depth=None, leaf=1, feat=sqrt |
| 4 | 2023-10 | 0.37 | 0.9386 | 0.9483 | 0.9016 | depth=8, leaf=1, feat=sqrt |
| 5 | 2023-11 | 0.42 | 0.9306 | 0.9672 | 0.8082 | depth=12, leaf=10, feat=sqrt |
| 6 | 2023-12 | 0.23 | 0.9187 | 0.9385 | 0.8472 | depth=20, leaf=1, feat=sqrt |

## vs. Baseline
0.9091 → 0.9271 (**+0.0180**) — strong improvement

## vs. Best Previous
Best was experiment 002 (RF + OOB threshold, defaults) at 0.9219. This is **+0.0052** above — **new best**.

## vs. Success Threshold
F2 ≥ 0.90 → **MET** (margin: +0.0271)

## Per-fold improvement over Experiment 002

| Fold | Test Month | Exp 002 | Exp 006 | Delta |
|---|---|---|---|---|
| 1 | 2023-07 | 0.8734 | 0.8811 | +0.0077 |
| 2 | 2023-08 | 0.9470 | 0.9506 | +0.0036 |
| 3 | 2023-09 | 0.9430 | 0.9430 | +0.0000 |
| 4 | 2023-10 | 0.9107 | 0.9386 | +0.0279 |
| 5 | 2023-11 | 0.9295 | 0.9306 | +0.0011 |
| 6 | 2023-12 | 0.9281 | 0.9187 | -0.0094 |

Improvement on 4/6 folds. Biggest gain on fold 4 (+0.028). Fold 1 (weakest) improved slightly but remains below 0.90.

## Feature Importance (Permutation, F2-based)

| Rank | Feature | Importance |
|---|---|---|
| 1 | product_usage_score | 0.6096 |
| 2 | days_since_last_login | 0.3289 |
| 3 | monthly_spend | 0.0379 |
| 4 | satisfaction_survey | 0.0025 |
| 5 | tenure_days | 0.0022 |
| 6–11 | noise features, support_tickets | ≤ 0.0012 |

## Selected Hyperparameters
Most folds selected deep trees (max_depth=20 or None), min_samples_leaf=1, and max_features='sqrt'. The defaults were near-optimal — the grid search primarily helped by adapting per-fold (fold 4 benefited from depth=8, fold 5 from leaf=10). max_features='sqrt' was selected in 5/6 folds.

## Observations
Hyperparameter tuning yielded a modest but real improvement (+0.005 F2 over best previous). The model is largely robust to hyperparameter choices — 'sqrt' features and deep trees are consistently near-optimal. The biggest remaining weakness is fold 1 (Jul 2023, F2=0.88), which has resisted improvement across all experiments. PR curve threshold tuning (replacing the grid sweep used in exp 002) also contributed to the improvement.

---
*Run on: 2026-05-10*
