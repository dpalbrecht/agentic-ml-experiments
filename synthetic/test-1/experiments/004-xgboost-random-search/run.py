import pandas as pd
from scipy.stats import loguniform, uniform
from xgboost import XGBClassifier
from sklearn.model_selection import (
    RandomizedSearchCV, cross_validate, StratifiedKFold
)

DATA_PATH = "data/data.csv"

print("=" * 60)
print("EXPERIMENT 004: XGBOOST RANDOM SEARCH")
print("=" * 60)

print("\nLoading data...")
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["target"])
y = df["target"]
print(f"  X: {X.shape}, y: {y.shape} (classes: {y.value_counts().to_dict()})")

param_dist = {
    "max_depth": [3, 4, 5, 6, 7, 8],
    "learning_rate": loguniform(0.01, 0.3),
    "n_estimators": [100, 200, 300, 400, 500],
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.5, 0.5),
    "min_child_weight": [1, 2, 3, 5, 7, 10],
    "gamma": loguniform(0.001, 1.0),
}

base_model = XGBClassifier(
    random_state=42,
    verbosity=0,
    eval_metric="logloss",
    n_jobs=1,
)

inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nRunning RandomizedSearchCV (50 iterations, 5-fold inner CV)...")
search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_dist,
    n_iter=50,
    scoring="roc_auc",
    cv=inner_cv,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
search.fit(X, y)

print(f"\nBest ROC-AUC (inner CV): {search.best_score_:.4f}")
print(f"Best params: {search.best_params_}")

print("\nEvaluating best params with 10-fold stratified CV...")
best_model = XGBClassifier(
    **search.best_params_,
    random_state=42,
    verbosity=0,
    eval_metric="logloss",
    n_jobs=1,
)

outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring = {"accuracy": "accuracy", "roc_auc": "roc_auc", "f1_macro": "f1_macro"}
results = cross_validate(best_model, X, y, cv=outer_cv, scoring=scoring, n_jobs=-1)

acc_mean = results["test_accuracy"].mean()
acc_std  = results["test_accuracy"].std()
auc_mean = results["test_roc_auc"].mean()
auc_std  = results["test_roc_auc"].std()
f1_mean  = results["test_f1_macro"].mean()
f1_std   = results["test_f1_macro"].std()

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"  Accuracy  (primary):   {acc_mean:.4f} ± {acc_std:.4f}")
print(f"  ROC-AUC   (secondary): {auc_mean:.4f} ± {auc_std:.4f}")
print(f"  Macro F1  (secondary): {f1_mean:.4f} ± {f1_std:.4f}")

BASELINE_ACC  = 0.7186
BEST_PREV_ACC = 0.8448
print(f"\nBaseline (000):     {BASELINE_ACC:.4f} → this run: {acc_mean:.4f} ({acc_mean - BASELINE_ACC:+.4f})")
print(f"Best previous (003): {BEST_PREV_ACC:.4f} → this run: {acc_mean:.4f} ({acc_mean - BEST_PREV_ACC:+.4f})")

print("\nPer-fold accuracy:")
for i, score in enumerate(results["test_accuracy"], 1):
    print(f"  Fold {i:02d}: {score:.4f}")

SUCCESS_THRESHOLD = 0.90
gap = acc_mean - SUCCESS_THRESHOLD
status = "MET" if acc_mean >= SUCCESS_THRESHOLD else f"NOT MET (gap: {gap:+.4f})"
print(f"\nSuccess threshold (90% accuracy): {status}")
print("=" * 60)
