import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, f1_score
from collections import Counter

DATA_PATH = "data/data.csv"
TARGET = "target"
N_FOLDS = 10
RANDOM_STATE = 42

PARAM_GRID = {
    "clf__C":     [0.1, 1, 10, 100],
    "clf__gamma": [0.001, 0.01, 0.1, 1],
}

# --- Load data ---
print("=== Loading data ===")
df = pd.read_csv(DATA_PATH)
features = [c for c in df.columns if c != TARGET]
X = df[features].values
y = df[TARGET].values
print(f"Shape: {X.shape}, target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
n_combos = len(PARAM_GRID["clf__C"]) * len(PARAM_GRID["clf__gamma"])
print(f"Grid: {n_combos} combos × 5 inner folds × {N_FOLDS} outer folds = {n_combos * 5 * N_FOLDS} fits")

# --- Pipeline ---
print("\n=== Building pipeline ===")
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
])
print("Pipeline: StandardScaler → SVC(kernel=rbf, probability=True)")

# --- Nested CV ---
print(f"\n=== Nested CV: inner 5-fold GridSearch, outer {N_FOLDS}-fold eval ===")
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
outer_cv  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=PARAM_GRID,
    cv=inner_cv,
    scoring="accuracy",
    n_jobs=-1,
    refit=True,
)

scoring = {
    "accuracy": make_scorer(accuracy_score),
    "roc_auc":  "roc_auc",
    "macro_f1": make_scorer(f1_score, average="macro"),
}

print("Running outer CV folds (n_jobs=1 outer, -1 inner)...")
results = cross_validate(grid_search, X, y, cv=outer_cv, scoring=scoring,
                         return_estimator=True, return_indices=True, n_jobs=1)

acc_scores = results["test_accuracy"]
auc_scores = results["test_roc_auc"]
f1_scores  = results["test_macro_f1"]

# --- Per-fold report ---
print("\n--- Per-fold results ---")
for i, (acc, auc, f1) in enumerate(zip(acc_scores, auc_scores, f1_scores), 1):
    best = results["estimator"][i - 1].best_params_
    print(f"  Fold {i:2d}: acc={acc:.4f}  auc={auc:.4f}  f1={f1:.4f}  "
          f"C={best['clf__C']}, gamma={best['clf__gamma']}")

print(f"\n=== Results ===")
print(f"Accuracy  (primary):   {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
print(f"ROC-AUC   (secondary): {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
print(f"Macro F1  (secondary): {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")

# --- Best params summary ---
print("\n=== Best params per fold ===")
c_votes     = [est.best_params_["clf__C"]     for est in results["estimator"]]
gamma_votes = [est.best_params_["clf__gamma"] for est in results["estimator"]]
print(f"  C votes:     {dict(Counter(c_votes))}")
print(f"  gamma votes: {dict(Counter(gamma_votes))}")

# --- Permutation importance (averaged across folds) ---
print("\n=== Permutation importance (mean across folds) ===")
print("Computing permutation importance on each held-out fold...")
perm_imp_matrix = np.zeros((N_FOLDS, len(features)))

for i, (est, fold_indices) in enumerate(zip(results["estimator"], results["indices"]["test"])):
    X_test_fold = X[fold_indices]
    y_test_fold = y[fold_indices]
    perm = permutation_importance(
        est, X_test_fold, y_test_fold,
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
        scoring="accuracy",
    )
    perm_imp_matrix[i] = perm.importances_mean
    print(f"  Fold {i+1:2d} done")

mean_perm_imp = perm_imp_matrix.mean(axis=0)
importance = pd.Series(mean_perm_imp, index=features).sort_values(ascending=False)
print("\nTop 10 features by mean permutation importance:")
for feat, val in importance.head(10).items():
    print(f"  {feat}: {val:+.4f}")
n_positive = (importance > 0).sum()
print(f"\nFeatures with positive importance: {n_positive} / {len(features)}")

# --- vs. prior experiments ---
print(f"\n=== vs. Prior Experiments ===")
prior = {
    "000-baseline (L2 logistic)":   0.7186,
    "001-lasso-lr (L1 logistic)":   0.7200,
    "002-hgbt (500 iters)":         0.8414,
    "003-hgbt (2000 iters)":        0.8436,
    "004-hgbt (top-3 feats)":       0.8396,
    "005-hgbt (hparam search)":     0.8440,
}
best_prev = max(prior.values())
for name, score in prior.items():
    print(f"  {name}: {score:.4f}")
this = acc_scores.mean()
print(f"  006-svm-rbf (this): {this:.4f}  (vs best prev: {this - best_prev:+.4f})")

threshold = 0.90
gap = this - threshold
status = "MET ✓" if gap >= 0 else f"NOT MET (gap: {gap:+.4f})"
print(f"\nSuccess threshold: {threshold:.0%} → {status}")

print("\n=== Done ===")
