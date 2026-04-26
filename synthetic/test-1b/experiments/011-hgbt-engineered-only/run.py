import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, f1_score

DATA_PATH = "data/data.csv"
TARGET = "target"
N_FOLDS = 10
RANDOM_STATE = 42

# Top-9 features by permutation importance from experiment 010
# (3 originals + 6 polynomial terms with positive importance)
KEEP_ORIGINALS = ["feature_0", "feature_1", "feature_2"]

# --- Load data ---
print("=== Loading data ===")
df = pd.read_csv(DATA_PATH)
y = df[TARGET].values
f0, f1, f2 = df["feature_0"].values, df["feature_1"].values, df["feature_2"].values

# --- Build feature matrix ---
print("\n=== Building engineered-only feature set ===")
data = {
    # Original top-3
    "feature_0":   f0,
    "feature_1":   f1,
    "feature_2":   f2,
    # Degree-2
    "f0_sq":       f0 ** 2,
    "f0_x_f2":     f0 * f2,
    "f1_sq":       f1 ** 2,
    # Degree-3 (positive importance from exp 010)
    "f0sq_x_f1":   f0 ** 2 * f1,
    "f0sq_x_f2":   f0 ** 2 * f2,
    "f1sq_x_f2":   f1 ** 2 * f2,
}
feature_names = list(data.keys())
X = np.column_stack(list(data.values()))

print(f"Features used ({len(feature_names)}): {feature_names}")
print(f"Dropped: 47 original noise features + 7 low-importance engineered terms")
print(f"Shape: {X.shape}")
print(f"Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# --- Build model ---
print("\n=== Building model ===")
model = HistGradientBoostingClassifier(
    max_iter=2000,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=20,
    n_iter_no_change=10,
    validation_fraction=0.1,
    tol=1e-4,
    random_state=RANDOM_STATE,
)
print("HistGradientBoostingClassifier(max_iter=2000, max_depth=5, lr=0.1, min_samples_leaf=20, early_stopping)")
print("9 features only — no original noise features")

# --- Cross-validation ---
print(f"\n=== Running stratified {N_FOLDS}-fold CV ===")
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

scoring = {
    "accuracy": make_scorer(accuracy_score),
    "roc_auc":  "roc_auc",
    "macro_f1": make_scorer(f1_score, average="macro"),
}

results = cross_validate(model, X, y, cv=cv, scoring=scoring,
                         return_estimator=True, return_indices=True, n_jobs=-1)

acc_scores = results["test_accuracy"]
auc_scores = results["test_roc_auc"]
f1_scores  = results["test_macro_f1"]

print("\n--- Per-fold results ---")
for i, (acc, auc, f1) in enumerate(zip(acc_scores, auc_scores, f1_scores), 1):
    n_iter = results["estimator"][i - 1].n_iter_
    stopped = "early-stop" if n_iter < 2000 else "hit cap"
    print(f"  Fold {i:2d}: acc={acc:.4f}  auc={auc:.4f}  f1={f1:.4f}  iters={n_iter} ({stopped})")

print(f"\n=== Results ===")
print(f"Accuracy  (primary):   {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
print(f"ROC-AUC   (secondary): {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
print(f"Macro F1  (secondary): {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")

iter_counts = [est.n_iter_ for est in results["estimator"]]
n_early = sum(1 for n in iter_counts if n < 2000)
print(f"Iterations: min={min(iter_counts)}, max={max(iter_counts)}, mean={np.mean(iter_counts):.1f}")
print(f"Folds with early stopping: {n_early}/{N_FOLDS}")

# --- Permutation importance ---
print("\n=== Permutation importance (mean across folds) ===")
perm_imp_matrix = np.zeros((N_FOLDS, len(feature_names)))
for i, (est, fold_indices) in enumerate(zip(results["estimator"], results["indices"]["test"])):
    perm = permutation_importance(est, X[fold_indices], y[fold_indices],
                                  n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
                                  scoring="accuracy")
    perm_imp_matrix[i] = perm.importances_mean
    print(f"  Fold {i+1:2d} done")

mean_perm_imp = perm_imp_matrix.mean(axis=0)
importance = pd.Series(mean_perm_imp, index=feature_names).sort_values(ascending=False)
print("\nAll features by mean permutation importance:")
for feat, val in importance.items():
    print(f"  {feat}: {val:+.4f}")

# --- vs. prior experiments ---
print(f"\n=== vs. Prior Experiments ===")
prior = {
    "005-hgbt (best, 50 feats)":          0.8440,
    "008-hgbt (52 feats, top-2 poly)":    0.8748,
    "009-hgbt (56 feats, full deg-2)":    0.8762,
    "010-hgbt (66 feats, deg-2+deg-3)":   0.8776,
}
best_prev = max(prior.values())
for name, score in prior.items():
    print(f"  {name}: {score:.4f}")
this = acc_scores.mean()
print(f"  011-hgbt-eng-only (this): {this:.4f}  (vs best prev: {this - best_prev:+.4f})")

threshold = 0.90
gap = this - threshold
status = "MET ✓" if gap >= 0 else f"NOT MET (gap: {gap:+.4f})"
print(f"\nSuccess threshold: {threshold:.0%} → {status}")

print("\n=== Done ===")
