import pandas as pd
import numpy as np
from tqdm import tqdm

DATA_PATH = "data/data.csv"

print("=" * 60)
print("LOADING DATA")
print("=" * 60)
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

print("\n" + "=" * 60)
print("DTYPES")
print("=" * 60)
print(df.dtypes.value_counts())

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)
target_counts = df["target"].value_counts().sort_index()
print(target_counts)
ratio = target_counts.max() / target_counts.min()
print(f"Imbalance ratio (max/min): {ratio:.2f}")
if ratio > 3:
    print("WARNING: Class imbalance >3:1 detected")
else:
    print("OK: Balanced classes")

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
missing_report = missing_report[missing_report["missing_count"] > 0]
if missing_report.empty:
    print("No missing values found.")
else:
    print(missing_report)
    flagged = missing_report[missing_report["missing_pct"] > 10]
    if not flagged.empty:
        print(f"\nWARNING: {len(flagged)} columns with >10% missing:")
        print(flagged)

print("\n" + "=" * 60)
print("FEATURE SUMMARY STATS")
print("=" * 60)
feature_cols = [c for c in df.columns if c != "target"]
stats = df[feature_cols].describe().T
print(stats[["mean", "std", "min", "max"]].to_string())

print("\n" + "=" * 60)
print("OUTLIER CHECK (>3 std from mean)")
print("=" * 60)
outlier_counts = {}
for col in tqdm(feature_cols, desc="Checking outliers"):
    mean, std = df[col].mean(), df[col].std()
    n_outliers = ((df[col] - mean).abs() > 3 * std).sum()
    if n_outliers > 0:
        outlier_counts[col] = n_outliers
if outlier_counts:
    print(f"{len(outlier_counts)} features have outliers >3std:")
    for col, cnt in sorted(outlier_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {col}: {cnt} outliers")
else:
    print("No features with >3std outliers found.")

print("\n" + "=" * 60)
print("ZERO-VARIANCE COLUMNS")
print("=" * 60)
zero_var = [c for c in feature_cols if df[c].std() == 0]
if zero_var:
    print(f"WARNING: Zero-variance columns: {zero_var}")
else:
    print("None found.")

print("\n" + "=" * 60)
print("HIGH CORRELATION PAIRS (>0.95)")
print("=" * 60)
corr_matrix = df[feature_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr = [(col, row, upper.loc[row, col])
             for col in upper.columns
             for row in upper.index
             if pd.notna(upper.loc[row, col]) and upper.loc[row, col] > 0.95]
if high_corr:
    print(f"WARNING: {len(high_corr)} highly correlated pairs (>0.95):")
    for c1, c2, val in high_corr[:10]:
        print(f"  {c1} <-> {c2}: {val:.3f}")
else:
    print("None found.")

print("\n" + "=" * 60)
print("LEAKAGE CHECK — correlation of each feature with target")
print("=" * 60)
target_corr = df[feature_cols].corrwith(df["target"]).abs().sort_values(ascending=False)
print("Top 10 features by |correlation| with target:")
print(target_corr.head(10).round(4))
suspicious = target_corr[target_corr > 0.9]
if not suspicious.empty:
    print(f"\nWARNING: {len(suspicious)} features with >0.9 correlation to target — potential leakage:")
    print(suspicious)
else:
    print("\nNo leakage risk detected (no feature >0.9 corr with target).")

print("\n" + "=" * 60)
print("DUPLICATES")
print("=" * 60)
n_dup = df.duplicated().sum()
print(f"Duplicate rows: {n_dup}")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
