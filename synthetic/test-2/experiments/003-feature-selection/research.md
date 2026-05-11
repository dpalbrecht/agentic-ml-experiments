# Experiment 003: Feature Selection — Research

## Approach Researched
Reducing the feature set from 11 to 3 based on permutation importance evidence from experiments 000–002.

## Key Findings
- No external research needed — this experiment is motivated entirely by internal evidence.
- Across all 3 prior experiments and 2 model families (LR and RF), only 3 features show meaningful importance:
  - product_usage_score (permutation importance 0.60)
  - days_since_last_login (0.33)
  - monthly_spend (0.04)
- The remaining 8 features (noise_0–4, satisfaction_survey, tenure_days, support_tickets_30d) all show permutation importance ≤ 0.001 or negative in experiment 002, meaning they contribute nothing or actively hurt.
- Impurity-based importance inflates the noise features (they appear ~0.017 each) because MDI is biased toward continuous numerical features — this is a known pitfall confirmed by research in experiment 001.
- With only 3 features, the RF has fewer dimensions to split on, which should reduce overfitting to noise and improve out-of-sample stability.

## Sources
- Internal experiments 000, 001, 002 (permutation importance tables)
