# Experiment 003: Feature Selection (Top 3)

## Approach
Same threshold-tuned Random Forest as experiment 002 (500 trees, `class_weight='balanced_subsample'`, OOB threshold tuning), but with only 3 features: product_usage_score, days_since_last_login, monthly_spend.

## Rationale
All 3 prior experiments consistently show only 3 features carry meaningful signal. The other 8 features show near-zero or negative permutation importance — they are noise the model must split around. Removing them should reduce overfitting and improve generalization, particularly on the weakest fold (Jul 2023, F2=0.8734 in exp 002).

## What changed from previous experiments
- Experiment 002 → 003: Feature set reduced from 11 to 3. Everything else identical.

## Evaluation
- **Primary metric:** F2 score
- **Secondary metrics:** Recall, Precision
- **Validation:** Sliding window temporal CV — 12-month train, 1-month test, 6 folds (test months Jul–Dec 2023)
