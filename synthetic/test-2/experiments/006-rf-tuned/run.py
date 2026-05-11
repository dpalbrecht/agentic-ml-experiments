import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, recall_score, precision_score, precision_recall_curve
from sklearn.inspection import permutation_importance
from itertools import product
from tqdm import tqdm

print("=" * 60)
print("EXPERIMENT 006: RF Hyperparameter Tuning")
print("=" * 60)

# --- Load and prepare data ---
print("\nLoading data...")
df = pd.read_csv("data/customer_churn.csv")
df["record_date"] = pd.to_datetime(df["record_date"])
df["year_month"] = df["record_date"].dt.to_period("M")
df = df.sort_values("record_date").reset_index(drop=True)
print(f"Loaded {len(df)} rows")

DROP_COLS = ["record_date", "year_month", "cancellation_request", "account_status_code", "churned"]
FEATURES = [c for c in df.columns if c not in DROP_COLS]
print(f"Features ({len(FEATURES)}): {FEATURES}")

# --- Hyperparameter grid ---
PARAM_GRID = {
    "max_depth": [None, 8, 12, 20],
    "min_samples_leaf": [1, 5, 10, 20],
    "max_features": ["sqrt", 0.5, None],
}
grid_combos = list(product(
    PARAM_GRID["max_depth"],
    PARAM_GRID["min_samples_leaf"],
    PARAM_GRID["max_features"],
))
print(f"\nGrid: {len(grid_combos)} combinations")
print(f"  max_depth: {PARAM_GRID['max_depth']}")
print(f"  min_samples_leaf: {PARAM_GRID['min_samples_leaf']}")
print(f"  max_features: {PARAM_GRID['max_features']}")

# --- Sliding window temporal CV ---
TRAIN_MONTHS = 12
N_FOLDS = 6
SEARCH_TREES = 200
FINAL_TREES = 500

all_months = sorted(df["year_month"].unique())
test_months = all_months[-N_FOLDS:]
print(f"\nTest months: {[str(m) for m in test_months]}")
print(f"Search: {SEARCH_TREES} trees | Final: {FINAL_TREES} trees")


def find_best_threshold_pr(y_true, y_proba):
    """Use precision-recall curve to find F2-maximizing threshold."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    precisions = precisions[:-1]
    recalls = recalls[:-1]
    f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-10)
    if len(f2_scores) == 0:
        return 0.5, 0.0
    best_idx = np.argmax(f2_scores)
    return thresholds[best_idx], f2_scores[best_idx]


fold_results = []
all_perm_importances = []
all_best_params = []

for fold_i, test_month in enumerate(test_months):
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

    # --- Grid search using OOB F2 ---
    print(f"\n  Fold {fold_i+1}: test={test_month} — searching {len(grid_combos)} combos...")
    best_oob_f2 = -1
    best_params = None
    best_threshold_search = 0.5

    for max_depth, min_samples_leaf, max_features in tqdm(
        grid_combos, desc=f"  Fold {fold_i+1} grid", leave=False
    ):
        rf = RandomForestClassifier(
            n_estimators=SEARCH_TREES,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight="balanced_subsample",
            oob_score=True,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        oob_proba = rf.oob_decision_function_[:, 1]
        threshold, oob_f2 = find_best_threshold_pr(y_train, oob_proba)

        if oob_f2 > best_oob_f2:
            best_oob_f2 = oob_f2
            best_params = {
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "max_features": max_features,
            }
            best_threshold_search = threshold

    print(f"  Best params: {best_params} (OOB F2={best_oob_f2:.4f})")

    # --- Refit with best params and full trees ---
    final_model = RandomForestClassifier(
        n_estimators=FINAL_TREES,
        max_depth=best_params["max_depth"],
        min_samples_leaf=best_params["min_samples_leaf"],
        max_features=best_params["max_features"],
        class_weight="balanced_subsample",
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X_train, y_train)

    # Re-tune threshold on final model's OOB predictions
    oob_proba = final_model.oob_decision_function_[:, 1]
    best_threshold, oob_f2_final = find_best_threshold_pr(y_train, oob_proba)

    # Apply to test
    y_proba_test = final_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba_test >= best_threshold).astype(int)

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
        "best_params": best_params,
        "threshold": best_threshold,
        "oob_f2": oob_f2_final,
        "f2": f2,
        "recall": rec,
        "precision": prec,
    })

    all_best_params.append(best_params)

    # Permutation importance
    def f2_scorer(estimator, X, y):
        proba = estimator.predict_proba(X)[:, 1]
        pred = (proba >= best_threshold).astype(int)
        return fbeta_score(y, pred, beta=2)

    perm_imp = permutation_importance(
        final_model, X_test, y_test,
        scoring=f2_scorer,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )
    all_perm_importances.append(perm_imp.importances_mean)

    print(f"  → F2={f2:.4f}  Recall={rec:.4f}  Precision={prec:.4f}  threshold={best_threshold:.2f}")

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
print(f"{'Fold':>4s}  {'Test':>10s}  {'Thresh':>6s}  {'F2':>7s}  {'Recall':>7s}  {'Prec':>7s}  {'OOB F2':>7s}  {'max_depth':>10s}  {'min_leaf':>8s}  {'max_feat':>8s}")
for r in fold_results:
    p = r["best_params"]
    print(f"{r['fold']:4d}  {r['test_month']:>10s}  {r['threshold']:6.2f}  "
          f"{r['f2']:7.4f}  {r['recall']:7.4f}  {r['precision']:7.4f}  {r['oob_f2']:7.4f}  "
          f"{str(p['max_depth']):>10s}  {str(p['min_samples_leaf']):>8s}  {str(p['max_features']):>8s}")

# --- Selected params summary ---
print("\nSelected hyperparameters across folds:")
for r in fold_results:
    p = r["best_params"]
    print(f"  Fold {r['fold']}: max_depth={p['max_depth']}, min_samples_leaf={p['min_samples_leaf']}, max_features={p['max_features']}")

# --- Feature importance ---
print("\n" + "-" * 60)
print("FEATURE IMPORTANCE (Permutation, F2-based)")
print("-" * 60)

mean_perm = np.mean(all_perm_importances, axis=0)
perm_order = np.argsort(mean_perm)[::-1]
for rank, idx in enumerate(perm_order):
    flag = " ← near zero or negative" if mean_perm[idx] <= 0.001 else ""
    print(f"  {rank+1:2d}. {FEATURES[idx]:30s}  importance = {mean_perm[idx]:.4f}{flag}")

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
print("\nPer-fold comparison with Experiment 002 (RF defaults + OOB threshold):")
exp002_f2 = [0.8734, 0.9470, 0.9430, 0.9107, 0.9295, 0.9281]
for i, r in enumerate(fold_results):
    delta = r["f2"] - exp002_f2[i]
    print(f"  Fold {r['fold']} ({r['test_month']}): {exp002_f2[i]:.4f} → {r['f2']:.4f} ({delta:+.4f})")

print("\n" + "=" * 60)
print("EXPERIMENT 006 COMPLETE")
print("=" * 60)
