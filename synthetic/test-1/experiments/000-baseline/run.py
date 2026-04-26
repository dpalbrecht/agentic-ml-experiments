import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, StratifiedKFold

DATA_PATH = "data/data.csv"

print("=" * 60)
print("EXPERIMENT 000: BASELINE — Logistic Regression")
print("=" * 60)

print("\nLoading data...")
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["target"])
y = df["target"]
print(f"  X: {X.shape}, y: {y.shape} (classes: {y.value_counts().to_dict()})")

print("\nBuilding pipeline...")
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)),
])

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring = {"accuracy": "accuracy", "roc_auc": "roc_auc", "f1_macro": "f1_macro"}

print("\nRunning 10-fold stratified cross-validation...")
results = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)

acc_mean = results["test_accuracy"].mean()
acc_std  = results["test_accuracy"].std()
auc_mean = results["test_roc_auc"].mean()
auc_std  = results["test_roc_auc"].std()
f1_mean  = results["test_f1_macro"].mean()
f1_std   = results["test_f1_macro"].std()

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"  Accuracy  (primary):  {acc_mean:.4f} ± {acc_std:.4f}")
print(f"  ROC-AUC   (secondary): {auc_mean:.4f} ± {auc_std:.4f}")
print(f"  Macro F1  (secondary): {f1_mean:.4f} ± {f1_std:.4f}")

print("\nPer-fold accuracy:")
for i, score in enumerate(results["test_accuracy"], 1):
    print(f"  Fold {i:02d}: {score:.4f}")

SUCCESS_THRESHOLD = 0.90
gap = acc_mean - SUCCESS_THRESHOLD
status = "MET" if acc_mean >= SUCCESS_THRESHOLD else f"NOT MET (gap: {gap:+.4f})"
print(f"\nSuccess threshold (90% accuracy): {status}")
print("=" * 60)
