import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

# --- Build pipeline ---
print("\n=== Building pipeline ===")
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)),
])
print(f"Pipeline: StandardScaler → LogisticRegression(C=1.0, solver=lbfgs, penalty=l2)")

# --- Cross-validation ---
print(f"\n=== Running stratified {N_FOLDS}-fold CV ===")
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

scoring = {
    "accuracy": make_scorer(accuracy_score),
    "roc_auc":  "roc_auc",
    "macro_f1": make_scorer(f1_score, average="macro"),
}

results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_estimator=True, n_jobs=-1)

acc_scores   = results["test_accuracy"]
auc_scores   = results["test_roc_auc"]
f1_scores    = results["test_macro_f1"]

print("\n--- Per-fold accuracy ---")
for i, s in enumerate(acc_scores, 1):
    print(f"  Fold {i:2d}: {s:.4f}")

print(f"\n=== Results ===")
print(f"Accuracy  (primary):  {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
print(f"ROC-AUC   (secondary): {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
print(f"Macro F1  (secondary): {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")

# --- Feature importance (mean absolute coefficients across folds) ---
print("\n=== Feature importance (mean |coef| across folds) ===")
coef_matrix = np.vstack([est.named_steps["clf"].coef_[0] for est in results["estimator"]])
mean_abs_coef = np.abs(coef_matrix).mean(axis=0)
importance = pd.Series(mean_abs_coef, index=features).sort_values(ascending=False)
print("Top 10 features by |coefficient|:")
for feat, val in importance.head(10).items():
    print(f"  {feat}: {val:.4f}")
spread = "concentrated" if importance.head(5).sum() / importance.sum() > 0.5 else "spread"
print(f"\nImportance is {spread} (top-5 share: {importance.head(5).sum()/importance.sum()*100:.1f}%)")

# --- vs. success threshold ---
print(f"\n=== vs. Success Threshold ===")
threshold = 0.90
gap = acc_scores.mean() - threshold
status = "MET" if gap >= 0 else f"NOT MET (gap: {gap:+.4f})"
print(f"Target: {threshold:.0%} | Achieved: {acc_scores.mean():.4f} | {status}")

print("\n=== Done ===")
