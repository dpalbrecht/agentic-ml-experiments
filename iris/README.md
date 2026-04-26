# Iris

## What We Tested

Standard Iris classification dataset: 150 samples, 4 features (sepal length/width, petal length/width), 3 balanced classes (setosa, versicolor, virginica). A clean, small, well-understood benchmark with no missing data, no noise features, and a known near-linear decision boundary between versicolor and virginica (setosa is trivially separable).

**Problem setup:** 3-class accuracy, stratified 5-fold CV, 97% success threshold.

## What Happened

| # | Approach | Accuracy | Delta vs Previous |
|---|----------|----------|-------------------|
| 000 | Logistic regression (default) | 96.67% | — (baseline) |
| 001 | Random forest (100 trees, default) | 94.67% | −2.0 pp |
| 002 | SVM RBF (default, unscaled) | 96.67% | +0.0 pp |
| 003 | SVM RBF + scaling + grid search (C, gamma) | 96.67% | +0.0 pp |
| 004 | SVM + scaling + full grid search (kernel, C, gamma) | **97.33%** | **+0.67 pp** ✓ |
| 005 | Linear SVM + scaling + fine C grid | 97.33% | +0.0 pp |
| 006 | k-NN + scaling + tuned k | 97.33% | +0.0 pp |

The success threshold was met in experiment 004. Experiments 005 and 006 confirmed the ceiling but couldn't break through it.

## Key Observations

**The baseline was already near the ceiling.** Logistic regression at 96.67% left only 0.33 pp to close — roughly half a sample per fold. The dataset is clean enough that a linear model captures most of the signal immediately. The experiment loop spent most of its budget confirming this rather than making meaningful gains.

**Random forest underperformed the linear baseline.** With only 150 samples and 4 features, ensemble variance hurts more than flexibility helps. This was correctly diagnosed in experiment 001's observations and the workflow moved on quickly.

**Scaling was necessary but not sufficient for SVM.** Unscaled SVM (002) matched logistic regression exactly; scaled SVM (003) still didn't improve. The gain came from expanding the kernel search to include linear (004) — the optimal kernel turned out to be linear with C=10, not RBF.

**97.33% appears to be a hard ceiling for single models on this CV setup.** Three different model families (SVM, k-NN, and implicitly logistic regression with further tuning) all converged to the same number. The bottleneck is fold 3, which consistently lands at 90–93% across all experiments. A small number of borderline versicolor/virginica samples in that fold are the limiting factor, not model choice. k-NN (006) rearranged the error distribution (fold 3 improved, fold 2 dropped) but the mean didn't change.

## Assessment

The workflow handled a clean, small dataset competently. The experiment log is a reasonable record of the standard progression: logistic → tree → SVM → k-NN. The success threshold was met.

What the workflow didn't surface: whether 97.33% is a true ceiling or a CV artifact. Fold 3's consistent underperformance across all models suggests the same borderline samples appear in its test set each time. An ensemble combining SVM and k-NN predictions could potentially break through by handling those specific boundary cases differently — but the workflow had no natural prompt to ask "is this ceiling a data problem or a model problem?"

The iris dataset is probably too easy to be a useful stress test of the workflow. There's no feature selection challenge, no noise, no class imbalance, and no nonlinear structure beyond the versicolor/virginica boundary. The experiment loop terminates naturally but without much to show about the workflow's ability to navigate harder problems.
