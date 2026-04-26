import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, f1_score
from tqdm import tqdm

DATA_PATH = "data/data.csv"
TARGET = "target"
N_FOLDS = 10
RANDOM_STATE = 42

PARAM_GRID = {
    "max_depth":     [3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1],
}

# --- Load data ---
print("=== Loading data ===")
df = pd.read_csv(DATA_PATH)
features = [c for c in df.columns if c != TARGET]
X = df[features].values
y = df[TARGET].values
print(f"Shape: {X.shape}, target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
print(f"Grid: {PARAM_GRID}  ({len(PARAM_GRID['max_depth']) * len(PARAM_GRID['learning_rate'])} combos)")

# --- Base estimator ---
base_model = HistGradientBoostingClassifier(
    max_iter=2000,
    min_samples_leaf=20,
    n_iter_no_change=10,
    validation_fraction=0.1,
    tol=1e-4,
    random_state=RANDOM_STATE,
)

# --- Nested CV ---
print(f"\n=== Nested CV: inner 5-fold GridSearch, outer {N_FOLDS}-fold eval ===")
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
outer_cv  = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

grid_search = GridSearchCV(
    estimator=base_model,
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

print("Running outer CV folds...")
results = cross_validate(grid_search, X, y, cv=outer_cv, scoring=scoring,
                         return_estimator=True, n_jobs=1)  # n_jobs=1 outer (inner uses -1)

acc_scores = results["test_accuracy"]
auc_scores = results["test_roc_auc"]
f1_scores  = results["test_macro_f1"]

# --- Per-fold report ---
print("\n--- Per-fold results ---")
for i, (acc, auc, f1) in enumerate(zip(acc_scores, auc_scores, f1_scores), 1):
    best = results["estimator"][i - 1].best_params_
    print(f"  Fold {i:2d}: acc={acc:.4f}  auc={auc:.4f}  f1={f1:.4f}  "
          f"best_params=depth={best['max_depth']}, lr={best['learning_rate']}")

print(f"\n=== Results ===")
print(f"Accuracy  (primary):   {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
print(f"ROC-AUC   (secondary): {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
print(f"Macro F1  (secondary): {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")

# --- Best params summary ---
print("\n=== Best params per fold ===")
depth_votes = [est.best_params_["max_depth"]     for est in results["estimator"]]
lr_votes    = [est.best_params_["learning_rate"]  for est in results["estimator"]]
from collections import Counter
print(f"  max_depth votes:     {dict(Counter(depth_votes))}")
print(f"  learning_rate votes: {dict(Counter(lr_votes))}")

# --- vs. prior experiments and success threshold ---
print(f"\n=== vs. Prior Experiments ===")
prior = {
    "000-baseline (L2 logistic)":   0.7186,
    "001-lasso-lr (L1 logistic)":   0.7200,
    "002-hgbt (500 iters)":         0.8414,
    "003-hgbt (2000 iters)":        0.8436,
    "004-hgbt (top-3 feats)":       0.8396,
}
for name, score in prior.items():
    print(f"  {name}: {score:.4f}")
this = acc_scores.mean()
best_prev = max(prior.values())
print(f"  005-hgbt-hparam-search: {this:.4f}  (vs best prev: {this-best_prev:+.4f})")

threshold = 0.90
gap = this - threshold
status = "MET ✓" if gap >= 0 else f"NOT MET (gap: {gap:+.4f})"
print(f"\nSuccess threshold: {threshold:.0%} → {status}")

print("\n=== Done ===")
