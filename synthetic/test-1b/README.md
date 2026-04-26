# Stress Test 1b: Known Ground Truth (Monte Carlo) — Repeat Run

## What We Tested

Same dataset as Test 1: 5,000 rows, 50 features, true decision boundary `feature_0 * feature_2 > feature_1²`, 10% label noise giving a theoretical accuracy ceiling of ~90%. The other 47 features are pure noise drawn from the same distribution as the real ones.

This run was a second pass on the same stress test to see whether the workflow, given another run, would reach the 90% ceiling — and if so, how.

## What Happened

### Problem Frame
Worked cleanly. Symmetric error priority and no constraints captured correctly.

### Data Exploration
The explore phase correctly identified that signal is concentrated in two features — feature_49 (|r| = 0.134) and feature_48 (|r| = 0.122) — while the remaining 48 features show near-zero individual correlation. Classes were reported as 71.9% / 28.1%.

**Same miss as Test 1:** The recommended feature set was "use all 50." The data assessment still did not distinguish between "weak correlations because features interact nonlinearly" and "weak correlations because most features are irrelevant." That ambiguity went unresolved into the experiment loop.

### Experiment Design
Reasonable choices: accuracy as primary metric, stratified 10-fold CV, logistic regression baseline, 90% success threshold. No changes from Test 1.

### Experiments

| # | Approach | Accuracy | Delta vs Previous |
|---|----------|----------|-------------------|
| 000 | Logistic regression | 71.9% | — (baseline) |
| 001 | Lasso logistic regression | 72.0% | +0.1 pp |
| 002 | HistGradientBoosting (500 iters) | 84.1% | +12.1 pp |
| 003 | HGBT (2000 iters) | 84.4% | +0.2 pp |
| 004 | HGBT, top-3 features only | 84.0% | −0.4 pp |
| 005 | HGBT hyperparameter search | 84.4% | +0.0 pp |
| 006 | SVM RBF | 73.9% | −10.5 pp vs best |
| 007 | Polynomial features + logistic regression | 82.2% | — |
| 008 | HGBT + 2 engineered features (f0×f2, f1²) | 87.5% | **+3.1 pp** |
| 009 | HGBT + full degree-2 expansion (6 terms) | 87.6% | +0.1 pp |
| 010 | HGBT + degree-3 expansion (10 terms) | 87.8% | +0.1 pp |
| 011 | HGBT, engineered features only (9 features) | 86.9% | −0.9 pp |

### What Changed Compared to Test 1

This run went further in two ways Test 1 did not.

**Feature importance was surfaced.** Permutation importance from experiment 003 named feature_0, feature_2, and feature_1 as the top three — giving the user the same signal that Test 1 never produced. Experiment 004 acted on this by dropping the other 47 features, confirming the sparse structure.

**Feature engineering was tried.** Experiment 007 (polynomial features + logistic regression) identified the exact interaction terms: `f0×f2` (coefficient 1.37) and `f1²` (coefficient 1.06) — the true boundary in interpretable form. Experiment 008 added these two engineered features to HGBT and produced the largest single gain in the run (+3.1 pp), pushing accuracy to 87.5%.

### Why the Ceiling Wasn't Reached

Despite identifying the correct interaction structure, the workflow plateaued at 87.8% — still ~2 pp short of the 90% ceiling. Several factors:

**The gap is real, not sampling variance.** An oracle model (applying the true boundary directly to the CSV) achieves 90.08% under 10-fold CV, with a 95% CI of [89.2%, 91.0%]. The best result here (87.8%) sits 1.4 pp below the lower bound of that CI — this is a genuine modeling shortfall, not fold-level noise.

**Wrong model for the engineered feature space.** Once `f0_x_f2` and `f1_sq` are available, the decision boundary is `f0_x_f2 − f1_sq > 0` — linear in those two features. HGBT is the wrong tool at that point. A logistic regression on just `[f0_x_f2, f1_sq]` would find the boundary cleanly with far less variance. The workflow continued adding polynomial terms and tuning HGBT after experiment 008, when the right move was to go simpler: drop the 50 noise features, drop the tree, fit a linear model on the two oracle features. The experiment that was never run — `LR([f0_x_f2, f1_sq])` — is the natural conclusion of what experiment 007 already revealed. That said, this is easier to see with hindsight (see below).

**Experiment 007 underperformed for the wrong reasons, and its results.md drew the wrong conclusion.** The L2 regularization used in 007 shrinks all 9 polynomial features together — including the 7 near-zero-coefficient ones. The penalty needed to control those 7 irrelevant features also pulled the coefficients on f0×f2 and f1² away from their true values. The results.md attributed the 2.2 pp gap below HGBT to model expressiveness ("logistic regression can only model a quadratic boundary") — but that is wrong. The boundary IS linear in the polynomial feature space; LR has sufficient expressiveness. The gap came from regularization, not capacity.

**Early stopping never fired in any experiment.** Every run hit its iteration cap, meaning HGBT was still finding marginal splits on the 50 noise features long after it found the real signal. The noise features are not neutral here — they add variance that pulls the CV mean down. Experiments 004 and 011 both showed that dropping original features slightly hurt HGBT performance, which looks like evidence for keeping them — but it's actually evidence that HGBT is the wrong model once the engineered features are available. A logistic regression wouldn't have this problem.

## Assessment

**This run identified the right features and the right interaction structure — and still couldn't reliably reach the 90% ceiling.** The key structural difference from Test 1 was that permutation importance from experiment 003 gave the user concrete feature names to act on, and the polynomial features experiment (007) revealed the boundary in algebraic form. Both of these came from the user continuing to experiment rather than from the workflow templates prompting them. The remaining gap (87.8% vs 90% ceiling) is not a noise-floor problem — it is a model-selection problem: once the oracle features are known, the right model is a logistic regression, not HGBT.

### What the workflow surfaced well
- Nonlinear signal structure (logistic regression fails, trees succeed — confirmed again)
- Permutation importance in experiment 003 named the three real features explicitly
- Polynomial logistic regression decoded the exact decision boundary (`f0×f2 > f1²`)
- The engineered-feature experiments made it clear that degree-3 terms add nothing (+0.1 pp over degree-2)
- The results comparisons made diminishing returns visible across the HGBT tuning experiments (003–005)

### Where the workflow left the user without enough information
- **Early stopping never firing** was not flagged in any results summary — a useful signal that the model is fitting noise rather than finding signal
- **The explore phase still didn't distinguish sparse-signal from dense-interaction problems.** "Weak individual correlations" was again left ambiguous; the user had to infer that feature selection was warranted from the permutation importance output in experiment 003, not from the data assessment
- **The explore output ranked feature_49 and feature_48 as most correlated with the target** — but those are noise features (the dataset includes two weakly-correlated distractors at r ≈ 0.15). The assessment did not flag this as suspicious or note that the highest-ranked features were only marginally above zero
- **No results summary noted that individual folds reached 90%+** — the mean CV number was reported, but the fold-level variance was not surfaced in a way that would tell the user "you've found the signal; this is now a noise-floor problem, not a modeling problem"
- **The workflow never prompted a model-simplification step after feature engineering.** Once the right features were known, the experiment loop kept tuning HGBT rather than asking whether a simpler model now fits the problem. A natural template prompt after a feature engineering win: "Given these engineered features, is the decision boundary now simpler? Should we try a linear model?"
- **The coefficient table from experiment 007 pointed at the right features but didn't create a clear threshold.** f0×f2 (1.37) and f1² (1.06) dominated, but f0² (0.29) and f2² (0.18) weren't obviously zero — feature_0 IS a real feature, so a small coefficient on f0² isn't obviously an artifact. Without hindsight, the experimenter couldn't confidently call those coefficients noise rather than genuine signal. L1 regularization (Lasso) is the principled in-the-moment choice for this situation: it auto-zeros irrelevant polynomial features rather than shrinking all of them together, and would have isolated [f0×f2, f1_sq] without requiring the experimenter to choose a cutoff by eye.
- **The wildly varying best-C across folds in experiment 007 (0.336 to 54.56 — two orders of magnitude) was a visible signal that the feature set was overloaded**, but nothing in the results template prompted the user to notice it or act on it.

### Suggested Structural Changes
- Add an "Iteration budget" note to the experiment results template: if early stopping never fired and the model hit the iteration cap, flag it — it's a sign the improvement is noise, not signal
- Add fold-level variance to the results summary alongside the mean — when individual folds hit the success threshold but the mean doesn't, the user needs that information to decide whether to stop or keep going
- The explore template's signal-concentration question (flagged in Test 1) would still have helped here: distinguishing sparse-signal from dense-interaction earlier would have sent the user toward feature engineering sooner
- The two distractor features (weakly correlated noise at r ≈ 0.15) ranked highest in the explore output's correlation table — a prompt like "are the highest-correlated features plausibly causal, or could this be coincidental?" would create a natural intervention point
- When a polynomial LR experiment is proposed, the template should default to L1 (Lasso) rather than L2 — when the goal is to identify which polynomial terms matter, Lasso's sparsity is the right inductive bias. L2 is appropriate when all features are expected to contribute; L1 is appropriate when only a subset are. The design template has no prompt to surface this choice.
