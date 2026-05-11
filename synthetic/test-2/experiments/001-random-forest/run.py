import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, recall_score, precision_score
from sklearn.inspection import permutation_importance
from tqdm import tqdm

print("=" * 60)
print("EXPERIMENT 001: Random Forest")
print("=" * 60)

# --- Load and prepare data ---
print("\nLoading data...")
df = pd.read_csv("data/customer_churn.csv")
df["record_date"] = pd.to_datetime(df["record_date"])
df["year_month"] = df["record_date"].dt.to_period("M")
df = df.sort_values("record_date").reset_index(drop=True)
print(f"Loaded {len(df)} rows, date range: {df['record_date'].min().date()} to {df['record_date'].max().date()}")

# Feature set: drop leakage columns, date, and target
DROP_COLS = ["record_date", "year_month", "cancellation_request", "account_status_code", "churned"]
FEATURES = [c for c in df.columns if c not in DROP_COLS]
print(f"Features ({len(FEATURES)}): {FEATURES}")

# --- Sliding window temporal CV ---
TRAIN_MONTHS = 12
N_FOLDS = 6

all_months = sorted(df["year_month"].unique())
test_months = all_months[-N_FOLDS:]
print(f"\nTest months: {[str(m) for m in test_months]}")
print(f"Running sliding window CV ({TRAIN_MONTHS}-month train, 1-month test, {N_FOLDS} folds)...")

fold_results = []
all_impurity_importances = []
all_perm_importances = []

for fold_i, test_month in enumerate(tqdm(test_months, desc="Folds")):
    test_month_idx = list(all_months).index(test_month)
    train_start_idx = test_month_idx - TRAIN_MONTHS
    train_months_range = all_months[train_start_idx:test_month_idx]

    train_mask = df["year_month"].isin(train_months_range)
    test_mask = df["year_month"] == test_month

    X_train = df.loc[train_mask, FEATURES].values
    y_train = df.loc[train_mask, "churned"].values
    X_test = df.loc[test_mask, FEATURES].values
    y_test = df.loc[test_mask, "churned"].values

    train_dates = df.loc[train_mask, "year_month"]
    train_range = f"{train_dates.iloc[0]} to {train_dates.iloc[-1]}"

    # Fit Random Forest
    model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    f2 = fbeta_score(y_test, y_pred, beta=2)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)

    fold_results.append({
        "fold": fold_i + 1,
        "train_range": train_range,
        "test_month": str(test_month),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "test_pos": int(y_test.sum()),
        "f2": f2,
        "recall": rec,
        "precision": prec,
    })

    # Feature importances
    all_impurity_importances.append(model.feature_importances_)

    # Permutation importance (on test set, using F2 scorer)
    def f2_scorer(estimator, X, y):
        return fbeta_score(y, estimator.predict(X), beta=2)

    perm_imp = permutation_importance(
        model, X_test, y_test,
        scoring=f2_scorer,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )
    all_perm_importances.append(perm_imp.importances_mean)

    print(f"  Fold {fold_i+1}: test={test_month}  F2={f2:.4f}  Recall={rec:.4f}  Precision={prec:.4f}  "
          f"(train={len(X_train)}, test={len(X_test)}, test_pos={int(y_test.sum())})")

# --- Aggregate results ---
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

f2_scores = [r["f2"] for r in fold_results]
rec_scores = [r["recall"] for r in fold_results]
prec_scores = [r["precision"] for r in fold_results]

print(f"\nPrimary Metric (F2):  {np.mean(f2_scores):.4f} ± {np.std(f2_scores):.4f}")
print(f"Recall:               {np.mean(rec_scores):.4f} ± {np.std(rec_scores):.4f}")
print(f"Precision:            {np.mean(prec_scores):.4f} ± {np.std(prec_scores):.4f}")

print("\nPer-fold breakdown:")
print(f"{'Fold':>4s}  {'Test Month':>10s}  {'Train':>5s}  {'Test':>5s}  {'Pos':>4s}  {'F2':>7s}  {'Recall':>7s}  {'Prec':>7s}  {'Train Range':>23s}")
for r in fold_results:
    print(f"{r['fold']:4d}  {r['test_month']:>10s}  {r['train_size']:5d}  {r['test_size']:5d}  {r['test_pos']:4d}  "
          f"{r['f2']:7.4f}  {r['recall']:7.4f}  {r['precision']:7.4f}  {r['train_range']:>23s}")

# --- Feature importance ---
print("\n" + "-" * 60)
print("FEATURE IMPORTANCE")
print("-" * 60)

# Impurity-based (MDI)
print("\nImpurity-based (Mean Decrease Impurity) — mean across folds:")
mean_impurity = np.mean(all_impurity_importances, axis=0)
imp_order = np.argsort(mean_impurity)[::-1]
for rank, idx in enumerate(imp_order):
    print(f"  {rank+1:2d}. {FEATURES[idx]:30s}  importance = {mean_impurity[idx]:.4f}")

# Permutation-based (on F2)
print("\nPermutation importance (F2-based) — mean across folds:")
mean_perm = np.mean(all_perm_importances, axis=0)
perm_order = np.argsort(mean_perm)[::-1]
for rank, idx in enumerate(perm_order):
    flag = " ← near zero or negative" if mean_perm[idx] <= 0.001 else ""
    print(f"  {rank+1:2d}. {FEATURES[idx]:30s}  importance = {mean_perm[idx]:.4f}{flag}")

# --- vs. Baseline and Success Threshold ---
baseline_f2 = 0.9091
mean_f2 = np.mean(f2_scores)
threshold = 0.90
delta_baseline = mean_f2 - baseline_f2
gap = threshold - mean_f2

print(f"\nvs. Baseline (000):    {baseline_f2:.4f} → {mean_f2:.4f} ({delta_baseline:+.4f})")
print(f"vs. Success threshold: F2 >= {threshold}")
if mean_f2 >= threshold:
    print(f"✓ THRESHOLD MET (margin: {mean_f2 - threshold:+.4f})")
else:
    print(f"✗ Gap of {gap:.4f} to close")

print("\n" + "=" * 60)
print("EXPERIMENT 001 COMPLETE")
print("=" * 60)
