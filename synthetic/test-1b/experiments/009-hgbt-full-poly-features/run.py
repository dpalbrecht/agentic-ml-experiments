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

# --- Load data ---
print("=== Loading data ===")
df = pd.read_csv(DATA_PATH)
features = [c for c in df.columns if c != TARGET]
y = df[TARGET].values

# --- Feature engineering: full degree-2 expansion of top-3 ---
print("\n=== Engineering features ===")
df_eng = df[features].copy()
f0, f1, f2 = df["feature_0"], df["feature_1"], df["feature_2"]

df_eng["f0_sq"]    = f0 ** 2
df_eng["f0_x_f1"]  = f0 * f1
df_eng["f0_x_f2"]  = f0 * f2
df_eng["f1_sq"]    = f1 ** 2
df_eng["f1_x_f2"]  = f1 * f2
df_eng["f2_sq"]    = f2 ** 2

poly_cols = ["f0_sq", "f0_x_f1", "f0_x_f2", "f1_sq", "f1_x_f2", "f2_sq"]
all_features = features + poly_cols
X = df_eng[all_features].values

print(f"Original features: {len(features)}")
print(f"Polynomial features added: {poly_cols}")
print(f"Total features: {X.shape[1]}")
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
print("Change from 008: full degree-2 expansion (6 terms) instead of top-2 only")

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
print("Computing on each held-out fold...")
perm_imp_matrix = np.zeros((N_FOLDS, len(all_features)))

for i, (est, fold_indices) in enumerate(zip(results["estimator"], results["indices"]["test"])):
    perm = permutation_importance(est, X[fold_indices], y[fold_indices],
                                  n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
                                  scoring="accuracy")
    perm_imp_matrix[i] = perm.importances_mean
    print(f"  Fold {i+1:2d} done")

mean_perm_imp = perm_imp_matrix.mean(axis=0)
importance = pd.Series(mean_perm_imp, index=all_features).sort_values(ascending=False)

print("\nTop 12 features by mean permutation importance:")
for feat, val in importance.head(12).items():
    marker = " ← engineered" if feat in poly_cols else ""
    print(f"  {feat}: {val:+.4f}{marker}")

print("\nAll engineered feature importances:")
for feat in poly_cols:
    rank = list(importance.index).index(feat) + 1
    print(f"  {feat}: {importance[feat]:+.4f}  (rank #{rank})")

# --- vs. prior experiments ---
print(f"\n=== vs. Prior Experiments ===")
prior = {
    "000-baseline (L2 logistic)":    0.7186,
    "005-hgbt (best no-eng)":        0.8440,
    "007-poly-features-lr":          0.8224,
    "008-hgbt (top-2 poly)":         0.8748,
}
best_prev = max(prior.values())
for name, score in prior.items():
    print(f"  {name}: {score:.4f}")
this = acc_scores.mean()
print(f"  009-hgbt-full-poly (this): {this:.4f}  (vs best prev: {this - best_prev:+.4f})")

threshold = 0.90
gap = this - threshold
status = "MET ✓" if gap >= 0 else f"NOT MET (gap: {gap:+.4f})"
print(f"\nSuccess threshold: {threshold:.0%} → {status}")

print("\n=== Done ===")
