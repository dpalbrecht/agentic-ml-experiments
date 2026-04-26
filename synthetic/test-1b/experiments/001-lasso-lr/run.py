import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, f1_score

DATA_PATH = "data/data.csv"
TARGET = "target"
N_FOLDS = 10
RANDOM_STATE = 42
CS = np.logspace(-4, 1, 20)

# --- Load data ---
print("=== Loading data ===")
df = pd.read_csv(DATA_PATH)
features = [c for c in df.columns if c != TARGET]
X = df[features].values
y = df[TARGET].values
print(f"Shape: {X.shape}, target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# --- Build pipeline ---
print("\n=== Building pipeline ===")
print(f"Pipeline: StandardScaler → LogisticRegressionCV(penalty=l1, solver=liblinear)")
print(f"C search grid: {CS[0]:.2e} to {CS[-1]:.2e} ({len(CS)} values)")

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegressionCV(
        Cs=CS,
        penalty="l1",
        solver="liblinear",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="accuracy",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )),
])

# --- Cross-validation ---
print(f"\n=== Running stratified {N_FOLDS}-fold CV ===")
outer_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

scoring = {
    "accuracy": make_scorer(accuracy_score),
    "roc_auc":  "roc_auc",
    "macro_f1": make_scorer(f1_score, average="macro"),
}

results = cross_validate(pipeline, X, y, cv=outer_cv, scoring=scoring,
                         return_estimator=True, n_jobs=-1)

acc_scores = results["test_accuracy"]
auc_scores = results["test_roc_auc"]
f1_scores  = results["test_macro_f1"]

print("\n--- Per-fold accuracy ---")
for i, s in enumerate(acc_scores, 1):
    print(f"  Fold {i:2d}: {s:.4f}")

print(f"\n=== Results ===")
print(f"Accuracy  (primary):   {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
print(f"ROC-AUC   (secondary): {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
print(f"Macro F1  (secondary): {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")

# --- Best C and feature sparsity per fold ---
print("\n=== Selected C and sparsity per fold ===")
best_cs = []
n_nonzero_list = []
for i, est in enumerate(results["estimator"], 1):
    clf = est.named_steps["clf"]
    best_c = clf.C_[0]
    best_cs.append(best_c)
    n_nonzero = (clf.coef_[0] != 0).sum()
    n_nonzero_list.append(n_nonzero)
    print(f"  Fold {i:2d}: best C={best_c:.4f}, non-zero features={n_nonzero}/{len(features)}")

print(f"\n  Mean best C: {np.mean(best_cs):.4f}")
print(f"  Mean non-zero features: {np.mean(n_nonzero_list):.1f} / {len(features)}")

# --- Feature importance (mean absolute coefficients across folds) ---
print("\n=== Feature importance (mean |coef| across folds) ===")
coef_matrix = np.vstack([est.named_steps["clf"].coef_[0] for est in results["estimator"]])
mean_abs_coef = np.abs(coef_matrix).mean(axis=0)
importance = pd.Series(mean_abs_coef, index=features).sort_values(ascending=False)

print("Top 10 features by mean |coefficient|:")
for feat, val in importance.head(10).items():
    print(f"  {feat}: {val:.4f}")

always_nonzero = [features[i] for i in range(len(features))
                  if all(est.named_steps["clf"].coef_[0][i] != 0 for est in results["estimator"])]
print(f"\nFeatures selected in ALL folds ({len(always_nonzero)}): {always_nonzero}")

# --- vs. baseline and success threshold ---
print(f"\n=== vs. Baseline & Success Threshold ===")
baseline_acc = 0.7186
print(f"Baseline accuracy:  {baseline_acc:.4f}")
print(f"This experiment:    {acc_scores.mean():.4f}  (delta: {acc_scores.mean()-baseline_acc:+.4f})")
threshold = 0.90
gap = acc_scores.mean() - threshold
status = "MET" if gap >= 0 else f"NOT MET (gap: {gap:+.4f})"
print(f"Success threshold:  {threshold:.0%} → {status}")

print("\n=== Done ===")
