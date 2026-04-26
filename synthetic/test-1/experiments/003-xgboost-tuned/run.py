import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold

DATA_PATH = "data/data.csv"

print("=" * 60)
print("EXPERIMENT 003: XGBOOST TUNED (slow learner)")
print("=" * 60)

print("\nLoading data...")
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["target"])
y = df["target"]
print(f"  X: {X.shape}, y: {y.shape} (classes: {y.value_counts().to_dict()})")

print("\nBuilding model...")
print("  n_estimators=500, learning_rate=0.05, max_depth=6")
print("  subsample=0.8, colsample_bytree=0.8, min_child_weight=3, gamma=0.1")
model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    random_state=42,
    verbosity=0,
    eval_metric="logloss",
)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring = {"accuracy": "accuracy", "roc_auc": "roc_auc", "f1_macro": "f1_macro"}

print("\nRunning 10-fold stratified cross-validation...")
results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

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
BEST_PREV_ACC = 0.8432
print(f"\nBaseline (000):     {BASELINE_ACC:.4f} → this run: {acc_mean:.4f} ({acc_mean - BASELINE_ACC:+.4f})")
print(f"Best previous (002): {BEST_PREV_ACC:.4f} → this run: {acc_mean:.4f} ({acc_mean - BEST_PREV_ACC:+.4f})")

print("\nPer-fold accuracy:")
for i, score in enumerate(results["test_accuracy"], 1):
    print(f"  Fold {i:02d}: {score:.4f}")

SUCCESS_THRESHOLD = 0.90
gap = acc_mean - SUCCESS_THRESHOLD
status = "MET" if acc_mean >= SUCCESS_THRESHOLD else f"NOT MET (gap: {gap:+.4f})"
print(f"\nSuccess threshold (90% accuracy): {status}")
print("=" * 60)
