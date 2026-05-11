# Stress Test 2: Temporal Leakage Trap

## What We Tested

Does the workflow surface leakage risks and temporal structure clearly enough that a knowledgeable user would catch them? Does the design phase create a natural intervention point for validation strategy? The dataset simulates customer churn prediction with 8,000 rows over 24 months, two leakage features (`cancellation_request` at r=0.88, `account_status_code` at r=0.88), 5 explicitly-labeled noise features, and an increasing churn rate over time (8% → 20%). Class imbalance is 6.68:1.

The test evaluates whether each phase gives a knowledgeable user the information and intervention points needed to avoid temporal leakage and select appropriate evaluation.

## What Happened

### Problem Frame
Captured the false-negative priority and deployment constraint ("deployed monthly on new customer data") correctly. The deployment constraint was added by the user mid-design and carried through to all subsequent decisions.

### Data Exploration
The explore phase correctly identified both leakage features with strong supporting evidence:
- `cancellation_request`: flagged at r=+0.88 with a crosstab showing 1,010 of 1,041 churned customers had a cancellation request — clearly a consequence rather than a predictor
- `account_status_code`: flagged at r=+0.88, with code=3 having exactly 1,041 entries (same count as churned customers)

It also correctly reported:
- 6.68:1 class imbalance
- Signal concentration in 3 features (product_usage_score, days_since_last_login, monthly_spend)
- Noise features at near-zero correlation
- Temporal range (729 unique dates over 2 years)

### Experiment Design
**This is where the user had to intervene.** The workflow initially recommended stratified 5-fold CV — the standard default. The user pushed back: "We're predicting churn, which has inherently temporal properties. Use a time-based temporal split." The workflow adapted and documented why stratified CV was rejected.

A second intervention was needed on the sliding window specifics. The workflow initially proposed an expanding window; the user specified a fixed 12-month sliding window with 6 folds on the most recent months. The rationale (recency bias, consistent training size) was incorporated after the user explained it.

Other design decisions were collaborative:
- F2 metric: workflow proposed, user agreed (correctly reflects FN-priority)
- Baseline: logistic regression with balanced class weights (user selected from options)
- Feature set: user overrode the workflow's recommendation to drop noise features, saying to keep them unless proven unhelpful
- Success threshold: user set F2 ≥ 0.90 (ambitious)

### Experiments

| # | Approach | F2 | Recall | Precision | Status |
|---|----------|-----|--------|-----------|--------|
| 000 | Logistic regression (baseline) | 0.9091 ± 0.017 | 0.9535 | 0.7702 | ✓ Barely |
| 001 | Random forest (default threshold) | 0.8666 ± 0.032 | 0.8589 | 0.9045 | ✗ |
| 002 | RF + OOB threshold tuning | 0.9219 ± 0.025 | 0.9543 | 0.8165 | ✓ |
| 003 | RF + threshold + 3 features only | 0.9114 ± 0.023 | 0.9419 | 0.8121 | ✓ (worse) |
| 004 | LightGBM + 1-month val threshold | 0.8965 ± 0.029 | 0.9169 | 0.8297 | ✗ |
| 005 | LightGBM + 3-month val threshold | 0.9104 ± 0.016 | 0.9422 | 0.8061 | ✓ |
| 006 | RF tuned + OOB PR-curve threshold | 0.9271 ± 0.023 | 0.9602 | 0.8175 | **✓ Best** |

Key experimental findings:
- **Threshold tuning was the most impactful single change** (+0.055 F2 for RF, going from 0.8666 to 0.9219)
- **OOB predictions >> temporal validation split for threshold tuning** — OOB uses ~4,000 samples and produces stable thresholds (spread 0.10), while a 1-month validation split (~300 samples) was wildly unstable (spread 0.87)
- **Feature selection hurt** — dropping from 11 to 3 features slightly decreased performance, despite 8 features having near-zero permutation importance
- **Feature importance was consistent across all experiments** — product_usage_score (dominant), days_since_last_login (strong), monthly_spend (moderate), everything else near zero

## Assessment

**The workflow correctly identified leakage and class imbalance, but required user intervention for temporal validation.** The explore phase's "Leakage Risk" section was excellent — it flagged both leakage features with enough evidence (crosstabs, exact count matches) that any knowledgeable user would immediately exclude them. The metric choice (F2) correctly traced from the problem frame's "FN is costly" through to a recall-weighted metric. Class imbalance was handled explicitly in every experiment.

The gap was in temporal validation: the design phase defaulted to stratified random CV despite the data having an obvious `record_date` column and the problem frame stating monthly deployment. This required the user to override.

### What the workflow surfaced well
- **Leakage detection was strong.** Both features flagged clearly with correlation values, crosstabs, and plain-language explanations of why they constitute leakage. The data assessment's "Leakage Risk" section worked exactly as intended.
- **Feature importance consistency** across experiments made it clear that signal is concentrated — this was reported from experiment 000 onward, giving the user immediate visibility.
- **Threshold tuning as a lever** was identified naturally when experiment 001 showed the RF had high precision but low recall — the workflow correctly diagnosed the default 0.5 threshold as the cause and proposed tuning.
- **Per-fold breakdowns** made it visible that fold 1 (Jul 2023) was consistently the weakest, giving the user information to investigate distributional shift if desired.
- **The "vs. baseline / vs. best previous / vs. threshold" comparisons** made diminishing returns and relative performance immediately visible at each step.
- **PR curve for threshold tuning** was introduced as a principled improvement over grid sweeping when the user asked about it — the workflow explained the trade-offs clearly.

### Where the workflow left the user without enough information
- **Temporal validation was not the default recommendation.** The design phase proposed stratified 5-fold CV despite the data having a date column and the problem frame mentioning monthly deployment. A user who didn't know about temporal leakage would have sleepwalked past this.
- **Expanding vs. sliding window required user input.** The workflow's initial "time-based" proposal was an expanding window — which doesn't match the fixed-size retraining pattern described in the problem frame. The user had to specify the sliding window and check that enough data existed.
- **The LightGBM threshold instability (exp 004) could have been predicted.** The workflow didn't warn that a single-month validation split (~300 samples) might be too noisy for threshold tuning — it only discovered this after the fact.
- **No early discussion of OOB vs. validation split trade-offs.** The RF's OOB approach was used without noting that it gives the model a significant structural advantage (free threshold tuning on full training data). This only became apparent when LightGBM couldn't match it.

### Suggested Structural Changes
- **The design phase should check for date/time columns** and, if found, default to temporal validation (or at minimum present it as the recommended option rather than an alternative). The intervention point exists but the default is wrong.
- **When proposing threshold tuning, note the sample size requirement.** A rule of thumb like "threshold tuning needs ≥1,000 samples in the tuning set to be stable" would have prevented the 1-month validation mistake.
- **The design phase should ask about the retraining pattern** ("Will you retrain on a fixed window or expanding history?") — this directly determines whether expanding or sliding window CV is appropriate.
- **When multiple model families are being compared, the results template should note structural advantages** — e.g., "RF has free OOB predictions for threshold tuning; GBMs don't. This affects threshold stability independently of model quality."
