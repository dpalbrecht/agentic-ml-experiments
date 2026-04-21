"""
Experiment 001: Implicit ALS Matrix Factorization
Learns personalized user/item latent factors from binary interaction matrix.

Run: docker run --rm -v "$(pwd)":/project agentic-automl python experiments/001-als/run.py
"""
import datetime
import time
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

K = 10

# ── 1. Load data ─────────────────────────────────────────────────────────────

print("Loading ratings...")
t0 = time.time()
ratings = pl.read_csv("data/ratings.csv")
print(f"  Loaded {len(ratings):,} ratings in {time.time()-t0:.1f}s")

# ── 2. Global time cutoff split ──────────────────────────────────────────────

cutoff_dt = datetime.datetime(2023, 7, 13, tzinfo=datetime.timezone.utc)
cutoff_ts = int(cutoff_dt.timestamp())

train_df = ratings.filter(pl.col("timestamp") < cutoff_ts)
test_df = ratings.filter(pl.col("timestamp") >= cutoff_ts)

print(f"\nTime cutoff: 2023-07-13 (unix={cutoff_ts})")
print(f"  Train: {len(train_df):,} interactions")
print(f"  Test:  {len(test_df):,} interactions")

# ── 3. Build user/item ID mappings ──────────────────────────────────────────

# Need contiguous integer IDs for the sparse matrix
train_users = set(train_df["userId"].unique().to_list())
test_users = set(test_df["userId"].unique().to_list())
eval_users = train_users & test_users

# Build mappings from all training data (need all items in the matrix)
all_user_ids = sorted(train_df["userId"].unique().to_list())
all_item_ids = sorted(train_df["movieId"].unique().to_list())

user_to_idx = {uid: i for i, uid in enumerate(all_user_ids)}
item_to_idx = {mid: i for i, mid in enumerate(all_item_ids)}
idx_to_item = {i: mid for mid, i in item_to_idx.items()}

n_users = len(all_user_ids)
n_items = len(all_item_ids)
print(f"\n  Matrix size: {n_users:,} users x {n_items:,} items")

# ── 4. Build sparse interaction matrix (user x item) ────────────────────────

print("Building sparse interaction matrix...")
t0 = time.time()

user_indices = train_df["userId"].to_list()
item_indices = train_df["movieId"].to_list()

rows = [user_to_idx[u] for u in user_indices]
cols = [item_to_idx[m] for m in item_indices]
data = np.ones(len(rows), dtype=np.float32)

user_item = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
print(f"  Built matrix in {time.time()-t0:.1f}s")
print(f"  NNZ: {user_item.nnz:,}")

# ── 5. Train ALS model ──────────────────────────────────────────────────────

print("\nTraining ALS model...")
print("  factors=128, regularization=0.01, alpha=1.0, iterations=15")
t0 = time.time()

model = AlternatingLeastSquares(
    factors=128,
    regularization=0.01,
    alpha=1.0,
    iterations=15,
    random_state=42,
)
model.fit(user_item)
train_time = time.time() - t0
print(f"  Training completed in {train_time:.1f}s")

# ── 6. Build per-user test sets ─────────────────────────────────────────────

print("\nBuilding per-user test sets...")
test_dict = {}
for row in (
    test_df
    .filter(pl.col("userId").is_in(list(eval_users)))
    .group_by("userId")
    .agg(pl.col("movieId").alias("test_items"))
).iter_rows():
    test_dict[row[0]] = set(row[1])

print(f"  Eval users: {len(test_dict):,}")

# ── 7. Evaluate ─────────────────────────────────────────────────────────────

print(f"\nEvaluating on {len(test_dict):,} users...")
t0 = time.time()

ndcg_scores = []
recall_scores = []
hit_scores = []
eval_count = 0

for uid, truth in test_dict.items():
    if uid not in user_to_idx:
        continue

    uidx = user_to_idx[uid]

    # Get top-K recommendations (filter_already_liked_items=True excludes training items)
    item_ids, scores = model.recommend(
        uidx, user_item[uidx], N=K, filter_already_liked_items=True
    )

    # Map back to movieIds
    recs = [idx_to_item[i] for i in item_ids]

    # Filter truth to only include items that exist in our item vocabulary
    truth_in_vocab = truth & set(item_to_idx.keys())
    if not truth_in_vocab:
        continue

    # NDCG@K (binary relevance)
    dcg = 0.0
    hits_in_topk = 0
    for rank, mid in enumerate(recs):
        if mid in truth_in_vocab:
            dcg += 1.0 / np.log2(rank + 2)
            hits_in_topk += 1

    n_relevant = min(len(truth_in_vocab), K)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    # Recall@K
    recall = hits_in_topk / len(truth_in_vocab)

    # Hit Rate@K
    hit = 1.0 if hits_in_topk > 0 else 0.0

    ndcg_scores.append(ndcg)
    recall_scores.append(recall)
    hit_scores.append(hit)
    eval_count += 1

    if eval_count % 1000 == 0:
        print(f"  Evaluated {eval_count:,} users...")

elapsed = time.time() - t0
print(f"  Done. Evaluated {eval_count:,} users in {elapsed:.1f}s")

# ── 8. Results ───────────────────────────────────────────────────────────────

mean_ndcg = np.mean(ndcg_scores)
mean_recall = np.mean(recall_scores)
mean_hit = np.mean(hit_scores)

# Baseline values for comparison
baseline_ndcg = 0.0473
baseline_recall = 0.0206
baseline_hit = 0.2077

print("\n" + "=" * 60)
print("RESULTS: Experiment 001 — ALS (factors=128)")
print("=" * 60)
print(f"  NDCG@{K}:     {mean_ndcg:.6f}  (baseline: {baseline_ndcg:.6f}, delta: {mean_ndcg - baseline_ndcg:+.6f})")
print(f"  Recall@{K}:   {mean_recall:.6f}  (baseline: {baseline_recall:.6f}, delta: {mean_recall - baseline_recall:+.6f})")
print(f"  Hit Rate@{K}: {mean_hit:.6f}  (baseline: {baseline_hit:.6f}, delta: {mean_hit - baseline_hit:+.6f})")
print(f"\n  Relative improvement NDCG@{K}: {(mean_ndcg - baseline_ndcg) / baseline_ndcg * 100:+.1f}%")
print(f"  Success threshold (10% rel. improvement): NDCG@{K} >= {baseline_ndcg * 1.1:.6f}")
success = mean_ndcg >= baseline_ndcg * 1.1
print(f"  Threshold met: {'YES' if success else 'NO'}")
print(f"\n  Training time: {train_time:.1f}s")
print(f"  Eval users:    {eval_count:,}")
print("=" * 60)
