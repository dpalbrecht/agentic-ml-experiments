# Experiment 002: RF Threshold Tuning — Research

## Approach Researched
Optimizing the decision threshold of a Random Forest classifier to maximize F2 score.

## Key Findings
- **scikit-learn's TunedThresholdClassifierCV** exists but uses internal stratified CV, which doesn't respect temporal ordering. We'll implement threshold tuning manually.
- **OOB (out-of-bag) predictions** provide unbiased probability estimates on training data without needing a held-out set. Each sample is predicted only by trees that didn't include it in their bootstrap sample. Available via `oob_score=True` and `oob_decision_function_`.
- **Threshold sweep**: scan thresholds from 0.05 to 0.95, compute F2 on OOB predictions for each, pick the threshold that maximizes F2.
- **Never tune threshold on test data** — this leaks information. OOB predictions on training data are the cleanest approach for Random Forest since they don't sacrifice training samples.
- **F2 with make_scorer**: fbeta_score requires beta parameter, so it must be wrapped with make_scorer for sklearn tools. For manual threshold tuning, we call fbeta_score directly.

## Sources
- [Tuning the decision threshold — scikit-learn 1.8.0](https://scikit-learn.org/stable/modules/classification_threshold.html)
- [Post-tuning the decision threshold for cost-sensitive learning — scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cost_sensitive_learning.html)
- [TunedThresholdClassifierCV — GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/how-to-use-scikit-learns-tunedthresholdclassifiercv-for-threshold-optimization/)
