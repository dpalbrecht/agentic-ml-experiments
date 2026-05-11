import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import fbeta_score, recall_score, precision_score, precision_recall_curve
from sklearn.inspection import permutation_importance
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

print("=" * 60)
print("EXPERIMENT 005: LightGBM — 3-Month Validation Window")
print("=" * 60)

# --- Load and prepare data ---
print("\nLoading data...")
df = pd.read_csv("data/customer_churn.csv")
df["record_date"] = pd.to_datetime(df["record_date"])
df["year_month"] = df["record_date"].dt.to_period("M")
df = df.sort_values("record_date").reset_index(drop=True)
print(f"Loaded {len(df)} rows")

# Feature set
DROP_COLS = ["record_date", "year_month", "cancellation_request", "account_status_code", "churned"]
FEATURES = [c for c in df.columns if c not in DROP_COLS]
print(f"Features ({len(FEATURES)}): {FEATURES}")

# --- Sliding window temporal CV ---
TRAIN_MONTHS = 12
VAL_MONTHS = 3
N_FOLDS = 6

all_months = sorted(df["year_month"].unique())
test_months = all_months[-N_FOLDS:]
print(f"\nTest months: {[str(m) for m in test_months]}")
print(f"Validation window: last {VAL_MONTHS} months of training")
print(f"Sub-train window: first {TRAIN_MONTHS - VAL_MONTHS} months of training")


def find_best_threshold_pr(y_true, y_proba):
    """Use precision-recall curve to find F2-maximizing threshold."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    precisions = precisions[:-1]
    recalls = recalls[:-1]
    f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-10)
    best_idx = np.argmax(f2_scores)
    return thresholds[best_idx], f2_scores[best_idx]


print(f"\nRunning sliding window CV with 3-month validation threshold tuning...")

fold_results = []
all_gain_importances = []
all_perm_importances = []

for fold_i, test_month in enumerate(tqdm(test_months, desc="Folds")):
    test_month_idx = list(all_months).index(test_month)
    train_start_idx = test_month_idx - TRAIN_MONTHS
    train_months_range = all_months[train_start_idx:test_month_idx]

    train_mask = df["year_month"].isin(train_months_range)
    test_mask = df["year_month"] == test_month

    X_train_full = df.loc[train_mask, FEATURES].values
    y_train_full = df.loc[train_mask, "churned"].values
    X_test = df.loc[test_mask, FEATURES].values
    y_test = df.loc[test_mask, "churned"].values

    train_dates = df.loc[train_mask, "year_month"]
    train_range = f"{train_dates.iloc[0]} to {train_dates.iloc[-1]}"

    # --- Threshold tuning: 3-month temporal validation split ---
    val_months_range = train_months_range[-VAL_MONTHS:]
    sub_train_months = train_months_range[:-VAL_MONTHS]

    sub_train_mask = df["year_month"].isin(sub_train_months)
    val_mask = df["year_month"].isin(val_months_range)

    X_sub_train = df.loc[sub_train_mask, FEATURES].values
    y_sub_train = df.loc[sub_train_mask, "churned"].values
    X_val = df.loc[val_mask, FEATURES].values
    y_val = df.loc[val_mask, "churned"].values

    # Fit on sub-train to find threshold
    model_for_threshold = LGBMClassifier(
        is_unbalance=True,
        importance_type="gain",
        random_state=42,
        verbosity=-1,
    )
    model_for_threshold.fit(X_sub_train, y_sub_train)

    val_proba = model_for_threshold.predict_proba(X_val)[:, 1]
    best_threshold, val_f2 = find_best_threshold_pr(y_val, val_proba)

    # --- Refit on full training window ---
    model = LGBMClassifier(
        is_unbalance=True,
        importance_type="gain",
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train_full, y_train_full)

    # Apply tuned threshold to test set
    y_proba_test = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba_test >= best_threshold).astype(int)

    # Metrics
    f2 = fbeta_score(y_test, y_pred, beta=2)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)

    fold_results.append({
        "fold": fold_i + 1,
        "train_range": train_range,
        "test_month": str(test_month),
        "train_size": len(X_train_full),
        "sub_train_size": len(X_sub_train),
        "val_size": len(X_val),
        "val_months": f"{val_months_range[0]} to {val_months_range[-1]}",
        "val_pos": int(y_val.sum()),
        "test_size": len(X_test),
        "test_pos": int(y_test.sum()),
        "threshold": best_threshold,
        "val_f2": val_f2,
        "f2": f2,
        "recall": rec,
        "precision": prec,
    })

    # Feature importances
    all_gain_importances.append(model.feature_importances_)

    def f2_scorer(estimator, X, y):
        proba = estimator.predict_proba(X)[:, 1]
        pred = (proba >= best_threshold).astype(int)
        return fbeta_score(y, pred, beta=2)

    perm_imp = permutation_importance(
        model, X_test, y_test,
        scoring=f2_scorer,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )
    all_perm_importances.append(perm_imp.importances_mean)

    print(f"  Fold {fold_i+1}: test={test_month}  threshold={best_threshold:.2f}  "
          f"F2={f2:.4f}  Recall={rec:.4f}  Precision={prec:.4f}  "
          f"(val={val_months_range[0]}–{val_months_range[-1]}, "
          f"val_size={len(X_val)}, val_pos={int(y_val.sum())}, val_F2={val_f2:.4f})")

# --- Aggregate results ---
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

f2_scores = [r["f2"] for r in fold_results]
rec_scores = [r["recall"] for r in fold_results]
prec_scores = [r["precision"] for r in fold_results]
thresholds_used = [r["threshold"] for r in fold_results]

print(f"\nPrimary Metric (F2):  {np.mean(f2_scores):.4f} ± {np.std(f2_scores):.4f}")
print(f"Recall:               {np.mean(rec_scores):.4f} ± {np.std(rec_scores):.4f}")
print(f"Precision:            {np.mean(prec_scores):.4f} ± {np.std(prec_scores):.4f}")
print(f"Thresholds used:      {[f'{t:.2f}' for t in thresholds_used]}")
print(f"Threshold range:      {min(thresholds_used):.2f} – {max(thresholds_used):.2f}")
print(f"Mean threshold:       {np.mean(thresholds_used):.3f}")

print("\nPer-fold breakdown:")
print(f"{'Fold':>4s}  {'Test':>10s}  {'Val Window':>20s}  {'ValSz':>5s}  {'Pos':>4s}  {'Thresh':>6s}  {'F2':>7s}  {'Recall':>7s}  {'Prec':>7s}")
for r in fold_results:
    print(f"{r['fold']:4d}  {r['test_month']:>10s}  {r['val_months']:>20s}  {r['val_size']:5d}  {r['val_pos']:4d}  "
          f"{r['threshold']:6.2f}  {r['f2']:7.4f}  {r['recall']:7.4f}  {r['precision']:7.4f}")

# --- Feature importance ---
print("\n" + "-" * 60)
print("FEATURE IMPORTANCE")
print("-" * 60)

print("\nGain-based — mean across folds:")
mean_gain = np.mean(all_gain_importances, axis=0)
gain_order = np.argsort(mean_gain)[::-1]
for rank, idx in enumerate(gain_order):
    print(f"  {rank+1:2d}. {FEATURES[idx]:30s}  gain = {mean_gain[idx]:.1f}")

print("\nPermutation importance (F2-based) — mean across folds:")
mean_perm = np.mean(all_perm_importances, axis=0)
perm_order = np.argsort(mean_perm)[::-1]
for rank, idx in enumerate(perm_order):
    flag = " ← near zero or negative" if mean_perm[idx] <= 0.001 else ""
    print(f"  {rank+1:2d}. {FEATURES[idx]:30s}  importance = {mean_perm[idx]:.4f}{flag}")

# --- Comparisons ---
baseline_f2 = 0.9091
best_prev_f2 = 0.9219  # experiment 002
exp004_f2 = 0.8965
mean_f2 = np.mean(f2_scores)
threshold_target = 0.90

print(f"\nvs. Baseline (000):    {baseline_f2:.4f} → {mean_f2:.4f} ({mean_f2 - baseline_f2:+.4f})")
print(f"vs. Best prev (002):   {best_prev_f2:.4f} → {mean_f2:.4f} ({mean_f2 - best_prev_f2:+.4f})")
print(f"vs. Exp 004 (1-mo val):{exp004_f2:.4f} → {mean_f2:.4f} ({mean_f2 - exp004_f2:+.4f})")
print(f"vs. Success threshold: F2 >= {threshold_target}")
if mean_f2 >= threshold_target:
    print(f"✓ THRESHOLD MET (margin: {mean_f2 - threshold_target:+.4f})")
else:
    print(f"✗ Gap of {threshold_target - mean_f2:.4f} to close")

# Threshold stability comparison
print(f"\nThreshold stability comparison:")
print(f"  Exp 002 (RF OOB):     range 0.17–0.27 (spread 0.10)")
print(f"  Exp 004 (LGBM 1-mo):  range {0.05:.2f}–{0.92:.2f} (spread {0.87:.2f})")
print(f"  Exp 005 (LGBM 3-mo):  range {min(thresholds_used):.2f}–{max(thresholds_used):.2f} (spread {max(thresholds_used)-min(thresholds_used):.2f})")

print("\n" + "=" * 60)
print("EXPERIMENT 005 COMPLETE")
print("=" * 60)
