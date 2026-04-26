"""
Stress Test 1: Known Ground Truth (Monte Carlo)
Generates a synthetic classification dataset where only 3 of 50 features matter.
True decision boundary: target = 1 when (feature_0 * feature_2 > feature_1^2), with 10% label noise.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

N = 5000
N_REAL = 3
N_NOISE = 45
N_DISTRACTOR = 2
NOISE_RATE = 0.10

# Real features
real = np.random.randn(N, N_REAL)

# True decision boundary
true_labels = (real[:, 0] * real[:, 2] > real[:, 1] ** 2).astype(int)

# Add 10% label noise
flip_mask = np.random.rand(N) < NOISE_RATE
labels = true_labels.copy()
labels[flip_mask] = 1 - labels[flip_mask]

# Pure noise features
noise = np.random.randn(N, N_NOISE)

# Distractor features: weakly correlated with target (~0.15)
distractors = np.column_stack([
    labels * 0.3 + np.random.randn(N) * 1.0,
    labels * 0.3 + np.random.randn(N) * 1.0,
])

# Assemble
all_features = np.hstack([real, noise, distractors])
columns = [f"feature_{i}" for i in range(all_features.shape[1])]
df = pd.DataFrame(all_features, columns=columns)
df["target"] = labels

df.to_csv("temp/synthetic_ground_truth.csv", index=False)

# Report
print(f"Shape: {df.shape}")
print(f"Target distribution:\n{df['target'].value_counts().to_string()}")
print(f"\nTheoretical accuracy ceiling: ~{1 - NOISE_RATE:.0%}")
print(f"Actual noise applied: {flip_mask.sum()} rows flipped ({flip_mask.mean():.1%})")

print("\n--- Feature-target correlations (top 10) ---")
corrs = df.drop(columns=["target"]).corrwith(df["target"]).abs().sort_values(ascending=False)
for feat, val in corrs.head(10).items():
    real_flag = " ← REAL" if feat in ["feature_0", "feature_1", "feature_2"] else ""
    dist_flag = " ← DISTRACTOR" if feat in ["feature_48", "feature_49"] else ""
    print(f"  {feat}: {val:.4f}{real_flag}{dist_flag}")

print("\nGround truth: features 0, 1, 2 are real. 3-47 are noise. 48-49 are distractors.")
print("Decision boundary: target=1 when feature_0 * feature_2 > feature_1^2")
