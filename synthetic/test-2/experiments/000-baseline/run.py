import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import fbeta_score, recall_score, precision_score
from tqdm import tqdm

print("=" * 60)
print("EXPERIMENT 000: BASELINE — Logistic Regression")
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

y_all = df["churned"].values
print(f"Target distribution: 0={np.sum(y_all == 0)}, 1={np.sum(y_all == 1)}")

# --- Sliding window temporal CV ---
# 12-month train, 1-month test, 6 folds on most recent months
TRAIN_MONTHS = 12
N_FOLDS = 6

all_months = sorted(df["year_month"].unique())
print(f"\nTotal months available: {len(all_months)} ({all_months[0]} to {all_months[-1]})")

# Test months are the last N_FOLDS months
test_months = all_months[-N_FOLDS:]
print(f"Test months: {[str(m) for m in test_months]}")

print(f"\nRunning sliding window CV ({TRAIN_MONTHS}-month train, 1-month test, {N_FOLDS} folds)...")

fold_results = []
all_coefs = []

for fold_i, test_month in enumerate(tqdm(test_months, desc="Folds")):
    # Training window: 12 months before the test month
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

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit logistic regression with balanced class weights
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)

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
        "test_neg": int((1 - y_test).sum()),
        "f2": f2,
        "recall": rec,
        "precision": prec,
    })

    all_coefs.append(model.coef_[0])

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

# --- Feature importance (mean absolute coefficients across folds) ---
print("\nFeature Importance (mean |coefficient| across folds, scaled features):")
mean_coefs = np.mean(np.abs(all_coefs), axis=0)
importance_order = np.argsort(mean_coefs)[::-1]
for rank, idx in enumerate(importance_order):
    print(f"  {rank+1:2d}. {FEATURES[idx]:30s}  |coef| = {mean_coefs[idx]:.4f}")

# --- vs. Success Threshold ---
mean_f2 = np.mean(f2_scores)
threshold = 0.90
gap = threshold - mean_f2
print(f"\nSuccess threshold: F2 >= {threshold}")
print(f"Current F2:        {mean_f2:.4f}")
if mean_f2 >= threshold:
    print("✓ THRESHOLD MET")
else:
    print(f"✗ Gap of {gap:.4f} to close")

print("\n" + "=" * 60)
print("BASELINE COMPLETE")
print("=" * 60)
