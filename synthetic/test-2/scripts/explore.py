import pandas as pd
import numpy as np

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 120)

print("=" * 60)
print("DATA EXPLORATION")
print("=" * 60)

df = pd.read_csv("data/customer_churn.csv")

# --- Shape & Schema ---
print("\n## Shape & Schema")
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
print("\nColumn dtypes:")
for col in df.columns:
    print(f"  {col:30s} {str(df[col].dtype):10s} (example: {df[col].iloc[0]})")

# --- Target Variable ---
print("\n## Target: churned")
counts = df["churned"].value_counts().sort_index()
print(counts.to_string())
ratio = counts.max() / counts.min()
print(f"Imbalance ratio: {ratio:.2f}:1")
if ratio > 3:
    print("⚠ IMBALANCED (>3:1)")
else:
    print("✓ Balanced enough")

# --- Missing Values ---
print("\n## Missing Values")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
has_missing = missing[missing > 0]
if len(has_missing) == 0:
    print("No missing values found.")
else:
    for col in has_missing.index:
        flag = " ⚠ >10%" if missing_pct[col] > 10 else ""
        print(f"  {col:30s} {has_missing[col]:5d} ({missing_pct[col]:.1f}%){flag}")

# --- Feature Distributions ---
print("\n## Feature Distributions (numeric)")
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop("churned")
stats = df[numeric_cols].describe().T
stats["iqr"] = stats["75%"] - stats["25%"]
stats["range"] = stats["max"] - stats["min"]
print(stats[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(3).to_string())

# Zero-variance columns
zero_var = [c for c in numeric_cols if df[c].std() == 0]
if zero_var:
    print(f"\n⚠ Zero-variance columns: {zero_var}")

# Outliers (>3 std from mean)
print("\n## Outlier Check (>3 std from mean)")
for col in numeric_cols:
    mean, std = df[col].mean(), df[col].std()
    if std > 0:
        outlier_count = ((df[col] - mean).abs() > 3 * std).sum()
        if outlier_count > 0:
            pct = outlier_count / len(df) * 100
            print(f"  {col:30s} {outlier_count:5d} outliers ({pct:.1f}%)")

# Highly correlated pairs
print("\n## Highly Correlated Feature Pairs (|r| > 0.95)")
corr = df[numeric_cols].corr()
pairs_found = False
for i in range(len(numeric_cols)):
    for j in range(i + 1, len(numeric_cols)):
        r = corr.iloc[i, j]
        if abs(r) > 0.95:
            print(f"  {numeric_cols[i]:30s} ↔ {numeric_cols[j]:30s} r={r:.3f}")
            pairs_found = True
if not pairs_found:
    print("  None found.")

# --- Leakage Check ---
print("\n## Leakage Check")
print("Correlations with target (churned):")
target_corr = df[numeric_cols].corrwith(df["churned"]).sort_values(key=abs, ascending=False)
for col, r in target_corr.items():
    flag = " ⚠ POSSIBLE LEAKAGE" if abs(r) > 0.8 else ""
    print(f"  {col:30s} r={r:+.4f}{flag}")

# cancellation_request deserves special attention
if "cancellation_request" in df.columns:
    print("\n  cancellation_request breakdown vs churned:")
    ct = pd.crosstab(df["cancellation_request"], df["churned"], margins=True)
    print(ct.to_string())

# --- Signal Structure ---
print("\n## Signal Structure")
meaningful = target_corr[target_corr.abs() > 0.05]
weak = target_corr[target_corr.abs() <= 0.05]
print(f"Features with |r| > 0.05 to target: {len(meaningful)}")
for col, r in meaningful.items():
    print(f"  {col:30s} r={r:+.4f}")
print(f"\nFeatures with |r| <= 0.05 to target (likely noise): {len(weak)}")
for col, r in weak.items():
    print(f"  {col:30s} r={r:+.4f}")

# --- Data Quality ---
print("\n## Data Quality")
dupes = df.duplicated().sum()
print(f"Duplicate rows: {dupes}")

# Check record_date
if "record_date" in df.columns:
    print(f"record_date range: {df['record_date'].min()} to {df['record_date'].max()}")
    print(f"record_date unique values: {df['record_date'].nunique()}")

# Check categoricals
non_numeric = df.select_dtypes(exclude=[np.number]).columns
for col in non_numeric:
    nuniq = df[col].nunique()
    print(f"  {col:30s} {nuniq} unique values")

# account_status_code
if "account_status_code" in df.columns:
    print(f"\naccount_status_code distribution:")
    print(df["account_status_code"].value_counts().sort_index().to_string())

print("\n" + "=" * 60)
print("EXPLORATION COMPLETE")
print("=" * 60)
