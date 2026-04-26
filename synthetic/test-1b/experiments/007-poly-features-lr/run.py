import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, f1_score

DATA_PATH = "data/data.csv"
TARGET = "target"
TOP_FEATURES = ["feature_0", "feature_1", "feature_2"]
N_FOLDS = 10
RANDOM_STATE = 42
CS = np.logspace(-3, 3, 20)

# --- Load data ---
print("=== Loading data ===")
df = pd.read_csv(DATA_PATH)
X_raw = df[TOP_FEATURES].values
y = df[TARGET].values
print(f"Input features: {TOP_FEATURES}")
print(f"Shape before expansion: {X_raw.shape}")

# Show what the expansion produces
from sklearn.preprocessing import PolynomialFeatures as PF
pf_demo = PF(degree=2, include_bias=False)
pf_demo.fit(X_raw[:1])
poly_names = pf_demo.get_feature_names_out(TOP_FEATURES)
print(f"Expanded features ({len(poly_names)}): {list(poly_names)}")
print(f"Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# --- Pipeline ---
print("\n=== Building pipeline ===")
pipeline = Pipeline([
    ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegressionCV(
                   Cs=CS,
                   penalty="l2",
                   solver="lbfgs",
                   cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
                   scoring="accuracy",
                   max_iter=1000,
                   random_state=RANDOM_STATE,
               )),
])
print("Pipeline: PolynomialFeatures(deg=2) → StandardScaler → LogisticRegressionCV(L2)")
print(f"C search: {CS[0]:.2e} to {CS[-1]:.2e} ({len(CS)} values, inner 5-fold)")

# --- Outer CV ---
print(f"\n=== Running stratified {N_FOLDS}-fold CV ===")
outer_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

scoring = {
    "accuracy": make_scorer(accuracy_score),
    "roc_auc":  "roc_auc",
    "macro_f1": make_scorer(f1_score, average="macro"),
}

results = cross_validate(pipeline, X_raw, y, cv=outer_cv, scoring=scoring,
                         return_estimator=True, n_jobs=-1)

acc_scores = results["test_accuracy"]
auc_scores = results["test_roc_auc"]
f1_scores  = results["test_macro_f1"]

# --- Per-fold report ---
print("\n--- Per-fold results ---")
for i, (acc, auc, f1) in enumerate(zip(acc_scores, auc_scores, f1_scores), 1):
    clf = results["estimator"][i - 1].named_steps["clf"]
    best_c = clf.C_[0]
    print(f"  Fold {i:2d}: acc={acc:.4f}  auc={auc:.4f}  f1={f1:.4f}  best_C={best_c:.4f}")

print(f"\n=== Results ===")
print(f"Accuracy  (primary):   {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
print(f"ROC-AUC   (secondary): {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
print(f"Macro F1  (secondary): {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")

# --- Feature importance (mean |coef| across folds) ---
print("\n=== Feature importance (mean |coef| across folds) ===")
coef_matrix = np.vstack([est.named_steps["clf"].coef_[0] for est in results["estimator"]])
mean_abs_coef = np.abs(coef_matrix).mean(axis=0)
importance = pd.Series(mean_abs_coef, index=poly_names).sort_values(ascending=False)
print("All 9 features by mean |coefficient|:")
for feat, val in importance.items():
    print(f"  {feat}: {val:.4f}")

# --- vs. prior experiments ---
print(f"\n=== vs. Prior Experiments ===")
prior = {
    "000-baseline (L2 logistic, 50 feats)": 0.7186,
    "001-lasso-lr (L1 logistic, 50 feats)": 0.7200,
    "002-hgbt (500 iters)":                 0.8414,
    "003-hgbt (2000 iters)":                0.8436,
    "004-hgbt (top-3 feats)":               0.8396,
    "005-hgbt (hparam search)":             0.8440,
    "006-svm-rbf":                          0.7388,
}
best_prev = max(prior.values())
for name, score in prior.items():
    print(f"  {name}: {score:.4f}")
this = acc_scores.mean()
print(f"  007-poly-features-lr (this): {this:.4f}  (vs best prev: {this - best_prev:+.4f})")

threshold = 0.90
gap = this - threshold
status = "MET ✓" if gap >= 0 else f"NOT MET (gap: {gap:+.4f})"
print(f"\nSuccess threshold: {threshold:.0%} → {status}")

print("\n=== Done ===")
