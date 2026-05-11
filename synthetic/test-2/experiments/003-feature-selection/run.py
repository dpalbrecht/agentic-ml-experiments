import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, recall_score, precision_score
from sklearn.inspection import permutation_importance
from tqdm import tqdm

print("=" * 60)
print("EXPERIMENT 003: Feature Selection (Top 3)")
print("=" * 60)

# --- Load and prepare data ---
print("\nLoading data...")
df = pd.read_csv("data/customer_churn.csv")
df["record_date"] = pd.to_datetime(df["record_date"])
df["year_month"] = df["record_date"].dt.to_period("M")
df = df.sort_values("record_date").reset_index(drop=True)
print(f"Loaded {len(df)} rows")

# Top 3 features only
FEATURES = ["product_usage_score", "days_since_last_login", "monthly_spend"]
print(f"Features ({len(FEATURES)}): {FEATURES}")

# --- Sliding window temporal CV ---
TRAIN_MONTHS = 12
N_FOLDS = 6

all_months = sorted(df["year_month"].unique())
test_months = all_months[-N_FOLDS:]
print(f"\nTest months: {[str(m) for m in test_months]}")

THRESHOLDS = np.arange(0.05, 0.96, 0.01)


def find_best_threshold(y_true, y_proba, thresholds):
    """Sweep thresholds and return the one maximizing F2."""
    best_t, best_f2 = 0.5, 0.0
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        if y_pred.sum() == 0:
            continue
        f2 = fbeta_score(y_true, y_pred, beta=2)
        if f2 > best_f2:
            best_f2 = f2
            best_t = t
    return best_t, best_f2


print(f"Running sliding window CV with OOB threshold tuning...")

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

    # Fit Random Forest with OOB enabled
    model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced_subsample",
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Tune threshold on OOB predictions
    oob_proba = model.oob_decision_function_[:, 1]
    best_threshold, oob_f2 = find_best_threshold(y_train, oob_proba, THRESHOLDS)

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
        "train_size": len(X_train),
        "test_size": len(X_test),
        "test_pos": int(y_test.sum()),
        "threshold": best_threshold,
        "oob_f2": oob_f2,
        "f2": f2,
        "recall": rec,
        "precision": prec,
    })

    # Feature importances
    all_impurity_importances.append(model.feature_importances_)

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
          f"(OOB F2={oob_f2:.4f})")

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
print(f"Mean threshold:       {np.mean(thresholds_used):.3f}")

print("\nPer-fold breakdown:")
print(f"{'Fold':>4s}  {'Test Month':>10s}  {'Thresh':>6s}  {'F2':>7s}  {'Recall':>7s}  {'Prec':>7s}  {'OOB F2':>7s}  {'Train Range':>23s}")
for r in fold_results:
    print(f"{r['fold']:4d}  {r['test_month']:>10s}  {r['threshold']:6.2f}  "
          f"{r['f2']:7.4f}  {r['recall']:7.4f}  {r['precision']:7.4f}  "
          f"{r['oob_f2']:7.4f}  {r['train_range']:>23s}")

# --- Feature importance ---
print("\n" + "-" * 60)
print("FEATURE IMPORTANCE")
print("-" * 60)

print("\nImpurity-based (MDI) — mean across folds:")
mean_impurity = np.mean(all_impurity_importances, axis=0)
imp_order = np.argsort(mean_impurity)[::-1]
for rank, idx in enumerate(imp_order):
    print(f"  {rank+1:2d}. {FEATURES[idx]:30s}  importance = {mean_impurity[idx]:.4f}")

print("\nPermutation importance (F2-based) — mean across folds:")
mean_perm = np.mean(all_perm_importances, axis=0)
perm_order = np.argsort(mean_perm)[::-1]
for rank, idx in enumerate(perm_order):
    print(f"  {rank+1:2d}. {FEATURES[idx]:30s}  importance = {mean_perm[idx]:.4f}")

# --- Comparisons ---
baseline_f2 = 0.9091
best_prev_f2 = 0.9219  # experiment 002
mean_f2 = np.mean(f2_scores)
threshold_target = 0.90

print(f"\nvs. Baseline (000):    {baseline_f2:.4f} → {mean_f2:.4f} ({mean_f2 - baseline_f2:+.4f})")
print(f"vs. Best prev (002):   {best_prev_f2:.4f} → {mean_f2:.4f} ({mean_f2 - best_prev_f2:+.4f})")
print(f"vs. Success threshold: F2 >= {threshold_target}")
if mean_f2 >= threshold_target:
    print(f"✓ THRESHOLD MET (margin: {mean_f2 - threshold_target:+.4f})")
else:
    print(f"✗ Gap of {threshold_target - mean_f2:.4f} to close")

# Per-fold comparison with exp 002
print("\nPer-fold comparison with Experiment 002 (11 features):")
exp002_f2 = [0.8734, 0.9470, 0.9430, 0.9107, 0.9295, 0.9281]
for i, r in enumerate(fold_results):
    delta = r["f2"] - exp002_f2[i]
    print(f"  Fold {r['fold']} ({r['test_month']}): {exp002_f2[i]:.4f} → {r['f2']:.4f} ({delta:+.4f})")

print("\n" + "=" * 60)
print("EXPERIMENT 003 COMPLETE")
print("=" * 60)
