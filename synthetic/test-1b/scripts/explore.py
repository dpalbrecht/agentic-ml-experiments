import pandas as pd
import numpy as np

DATA_PATH = "data/data.csv"
TARGET = "target"

print("=== Loading data ===")
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

features = [c for c in df.columns if c != TARGET]
print(f"Features: {len(features)} | Target: {TARGET}")

# --- Target distribution ---
print("\n=== Target distribution ===")
counts = df[TARGET].value_counts().sort_index()
for cls, cnt in counts.items():
    print(f"  class {cls}: {cnt} ({cnt/len(df)*100:.1f}%)")
ratio = counts.max() / counts.min()
if ratio > 3:
    print(f"  WARNING: imbalanced (ratio {ratio:.1f}:1)")
else:
    print(f"  Balance ratio: {ratio:.2f}:1 — balanced")

# --- Missing values ---
print("\n=== Missing values ===")
missing = df.isnull().sum()
missing_pct = missing / len(df) * 100
flagged_missing = missing_pct[missing_pct > 10]
if flagged_missing.empty:
    print("  No columns with >10% missing")
else:
    for col, pct in flagged_missing.items():
        print(f"  WARNING: {col} has {pct:.1f}% missing")
total_missing = missing.sum()
print(f"  Total missing cells: {total_missing}")

# --- Duplicates ---
print("\n=== Duplicates ===")
dupes = df.duplicated().sum()
print(f"  Duplicate rows: {dupes}")

# --- Feature summary stats ---
print("\n=== Feature summary (numeric) ===")
desc = df[features].describe()
print(desc.T[["mean", "std", "min", "max"]].to_string())

# --- Zero-variance columns ---
print("\n=== Zero-variance columns ===")
zero_var = [c for c in features if df[c].nunique() <= 1]
print(f"  {zero_var if zero_var else 'None'}")

# --- Outliers (>3 std from mean) ---
print("\n=== Outlier counts (>3 std) ===")
outlier_counts = {}
for col in features:
    mean, std = df[col].mean(), df[col].std()
    if std > 0:
        n = ((df[col] - mean).abs() > 3 * std).sum()
        if n > 0:
            outlier_counts[col] = n
if outlier_counts:
    for col, n in sorted(outlier_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {col}: {n} outliers")
    if len(outlier_counts) > 10:
        print(f"  ... and {len(outlier_counts)-10} more columns with outliers")
else:
    print("  None")

# --- High correlations between features ---
print("\n=== Highly correlated feature pairs (|r| > 0.95) ===")
corr_matrix = df[features].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_pairs = [
    (col, row, upper.loc[row, col])
    for col in upper.columns
    for row in upper.index
    if pd.notna(upper.loc[row, col]) and upper.loc[row, col] > 0.95
]
if high_corr_pairs:
    for a, b, r in sorted(high_corr_pairs, key=lambda x: -x[2]):
        print(f"  {a} — {b}: r={r:.3f}")
else:
    print("  None")

# --- Feature-target correlations ---
print("\n=== Feature-target correlations (point-biserial / Pearson) ===")
target_corrs = df[features].corrwith(df[TARGET]).abs().sort_values(ascending=False)
print("  Top 10 correlated features:")
for col, r in target_corrs.head(10).items():
    print(f"    {col}: |r|={r:.4f}")
print(f"\n  Median |r| across all features: {target_corrs.median():.4f}")
print(f"  Features with |r| > 0.1: {(target_corrs > 0.1).sum()}")
print(f"  Features with |r| > 0.3: {(target_corrs > 0.3).sum()}")

# --- Leakage check ---
print("\n=== Leakage check ===")
perfect = target_corrs[target_corrs > 0.99]
if not perfect.empty:
    print(f"  WARNING: suspiciously high correlation with target: {perfect.to_dict()}")
else:
    print("  No features with |r| > 0.99 with target")

print("\n=== Done ===")
