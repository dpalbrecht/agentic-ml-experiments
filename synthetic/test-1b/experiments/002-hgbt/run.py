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
X = df[features].values
y = df[TARGET].values
print(f"Shape: {X.shape}, target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# --- Build model ---
print("\n=== Building model ===")
model = HistGradientBoostingClassifier(
    max_iter=500,
    max_depth=7,
    learning_rate=0.1,
    min_samples_leaf=20,
    n_iter_no_change=10,
    validation_fraction=0.1,
    tol=1e-4,
    random_state=RANDOM_STATE,
)
print("HistGradientBoostingClassifier(max_iter=500, max_depth=7, lr=0.1, min_samples_leaf=20, early_stopping)")

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
    est = results["estimator"][i - 1]
    n_iter = est.n_iter_
    print(f"  Fold {i:2d}: acc={acc:.4f}  auc={auc:.4f}  f1={f1:.4f}  iters={n_iter}")

print(f"\n=== Results ===")
print(f"Accuracy  (primary):   {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
print(f"ROC-AUC   (secondary): {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
print(f"Macro F1  (secondary): {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")

mean_iters = np.mean([est.n_iter_ for est in results["estimator"]])
print(f"Mean early-stop iterations: {mean_iters:.1f} / 500")

# --- Permutation importance (averaged across all folds) ---
print("\n=== Permutation importance (mean across folds) ===")
print("Computing permutation importance on each held-out fold...")
perm_imp_matrix = np.zeros((N_FOLDS, len(features)))

for i, (est, fold_indices) in enumerate(zip(results["estimator"], results["indices"]["test"])):
    X_test_fold = X[fold_indices]
    y_test_fold = y[fold_indices]
    perm = permutation_importance(est, X_test_fold, y_test_fold,
                                  n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
                                  scoring="accuracy")
    perm_imp_matrix[i] = perm.importances_mean
    print(f"  Fold {i+1:2d} done")

mean_perm_imp = perm_imp_matrix.mean(axis=0)
importance = pd.Series(mean_perm_imp, index=features).sort_values(ascending=False)

print("\nTop 10 features by mean permutation importance:")
for feat, val in importance.head(10).items():
    print(f"  {feat}: {val:+.4f}")

positive_features = importance[importance > 0]
print(f"\nFeatures with positive importance: {len(positive_features)} / {len(features)}")

# --- vs. baseline and success threshold ---
print(f"\n=== vs. Baseline & Prior Experiments ===")
baseline_acc = 0.7186
lasso_acc    = 0.7200
print(f"000-baseline (L2 logistic):  {baseline_acc:.4f}")
print(f"001-lasso-lr (L1 logistic):  {lasso_acc:.4f}")
print(f"002-hgbt     (this):         {acc_scores.mean():.4f}  (vs baseline: {acc_scores.mean()-baseline_acc:+.4f})")

threshold = 0.90
gap = acc_scores.mean() - threshold
status = "MET" if gap >= 0 else f"NOT MET (gap: {gap:+.4f})"
print(f"\nSuccess threshold: {threshold:.0%} → {status}")

print("\n=== Done ===")
