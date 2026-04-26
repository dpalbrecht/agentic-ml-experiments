# Stress Test 1: Known Ground Truth (Monte Carlo)

## What We Tested

Can the workflow surface enough information for a user to identify sparse signal structure and act on it? The dataset has 50 features, but only 3 matter — the true decision boundary is `feature_0 * feature_2 > feature_1²`, with 10% label noise giving a theoretical accuracy ceiling of ~90%. The other 47 features are pure noise drawn from the same distribution as the real ones.

The test evaluates the workflow as a decision-support tool, not an autonomous system. The question is whether each phase gives a knowledgeable user what they need to notice the problem and intervene.

## What Happened

### Problem Frame
Worked as expected. Captured the symmetric error priority and no constraints — straightforward for this problem.

### Data Exploration
The explore phase reported that all 50 features have near-zero individual correlation with the target (max |r| = 0.134) and correctly interpreted this as "complex feature interactions rather than individual features." It recommended using all 50 features and flagged no issues.

**The miss:** "Weak individual correlations" has two possible interpretations — (a) features interact nonlinearly, or (b) most features are irrelevant. The explore output didn't distinguish between these, and the "Recommended Feature Set" section said "use all 50" with no hedging. That recommendation was never revisited.

### Experiment Design
Reasonable choices: accuracy as primary metric (symmetric errors), stratified 10-fold CV (appropriate for 5,000 rows), logistic regression baseline, 90% success threshold.

### Experiments

| # | Approach | Accuracy | Delta vs Previous |
|---|----------|----------|-------------------|
| 000 | Logistic regression | 71.9% | — (baseline, essentially majority-class) |
| 001 | Random forest | 75.3% | +3.4 pp |
| 002 | XGBoost (defaults) | 84.3% | +9.0 pp |
| 003 | XGBoost (manual tuning) | 84.5% | +0.2 pp |
| 004 | XGBoost (random search, 50 iter) | 85.0% | +0.5 pp |
| 005 | MLP (128, 64) | 75.5% | −9.5 pp vs best |

The workflow correctly identified that the problem is nonlinear (experiment 000→001), that gradient boosting is the right model family (001→002), and that hyperparameter tuning has diminishing returns (003→004). The results template's "vs. Best Previous" comparisons made the plateau visible.

### What Was Never Tried
**Feature selection.** Every experiment used all 50 features. The 85% ceiling is almost certainly caused by 47 noise features diluting the signal — XGBoost is good at ignoring noise, but 47:3 noise-to-signal with 5,000 rows is a lot to overcome. A feature importance analysis would have shown that XGBoost concentrates weight on features 0, 1, and 2, and dropping the rest would likely push accuracy toward the 90% ceiling.

The MLP regression (005 → 75.5%) was blamed on the model family, but a 2-layer MLP can easily learn `x0 * x2 > x1²`. The real problem was 47 noise dimensions with insufficient samples — another case where feature selection was the right diagnosis but the workflow didn't prompt it.

## Assessment

**The workflow made reasonable model-selection decisions but never revisited its initial feature set recommendation.** The data assessment's "use all 50" became an unquestioned assumption for all six experiments. The experiment loop's "change one thing at a time" guideline implicitly pointed toward models and hyperparameters — it never prompted the user to consider feature engineering or selection as an experiment axis.

### What the workflow surfaced well
- The nonlinear signal structure (logistic regression failed, trees succeeded)
- Diminishing returns from hyperparameter tuning (clearly visible in results comparisons)
- Model family differences (XGBoost vs MLP)

### Where the workflow left the user without enough information
- No feature importance reporting in any experiment — the user had no visibility into *which features the model relied on*
- The "Recommended Feature Set" was set once in explore and never challenged
- The experiment loop's examples and guidelines focused entirely on model/hyperparameter changes, not feature-level interventions

### Suggested Structural Changes
- Add a "Feature Importance" or "What the model relied on" section to the experiment results template — if every experiment reported its top features, the 3-feature concentration would be obvious by experiment 002
- Add a prompt in the explore template for "Signal concentration: are a few features dominant, or is signal spread across many?" to distinguish sparse-signal from dense-interaction problems
- Include "feature engineering / selection" as an explicit experiment axis in the `/ml-experiment` command's guidelines, alongside model changes and hyperparameter tuning
