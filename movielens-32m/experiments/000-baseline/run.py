"""
Experiment 000: Popularity Baseline
Global popularity recommender evaluated with NDCG@10, Recall@10, Hit Rate@10.
Global time cutoff split (~last 3 months → test).

Run: docker run --rm -v "$(pwd)":/project agentic-automl python experiments/000-baseline/run.py
"""
import numpy as np
import polars as pl
import time

K = 10

# ── 1. Load data ─────────────────────────────────────────────────────────────

print("Loading ratings...")
t0 = time.time()
ratings = pl.read_csv("data/ratings.csv")
print(f"  Loaded {len(ratings):,} ratings in {time.time()-t0:.1f}s")

# ── 2. Global time cutoff split ──────────────────────────────────────────────

# Data spans 1995-01-09 to 2023-10-13. Last ~3 months → cutoff around 2023-07-13.
import datetime
cutoff_dt = datetime.datetime(2023, 7, 13, tzinfo=datetime.timezone.utc)
cutoff_ts = int(cutoff_dt.timestamp())

train = ratings.filter(pl.col("timestamp") < cutoff_ts)
test = ratings.filter(pl.col("timestamp") >= cutoff_ts)

print(f"\nTime cutoff: 2023-07-13 (unix={cutoff_ts})")
print(f"  Train: {len(train):,} interactions")
print(f"  Test:  {len(test):,} interactions")

# ── 3. Filter users: must have both train and test interactions ──────────────

train_users = set(train["userId"].unique().to_list())
test_users = set(test["userId"].unique().to_list())
eval_users = train_users & test_users

print(f"\n  Users in train: {len(train_users):,}")
print(f"  Users in test:  {len(test_users):,}")
print(f"  Users in both (eval set): {len(eval_users):,}")

# ── 4. Build popularity ranking from training data ──────────────────────────

print("\nBuilding popularity ranking from training data...")
popularity = (
    train
    .group_by("movieId")
    .len()
    .sort("len", descending=True)
)
# Ordered list of movie IDs by popularity (most popular first)
pop_ranking = popularity["movieId"].to_list()
print(f"  {len(pop_ranking):,} movies in training set")

# ── 5. Build per-user training sets (movies already seen) ───────────────────

print("Building per-user seen sets...")
t0 = time.time()
user_train_items = (
    train
    .filter(pl.col("userId").is_in(list(eval_users)))
    .group_by("userId")
    .agg(pl.col("movieId").alias("seen"))
)
# Convert to dict for fast lookup
seen_dict = {}
for row in user_train_items.iter_rows():
    seen_dict[row[0]] = set(row[1])
print(f"  Built seen sets in {time.time()-t0:.1f}s")

# ── 6. Build per-user test sets (ground truth) ──────────────────────────────

print("Building per-user test sets...")
t0 = time.time()
user_test_items = (
    test
    .filter(pl.col("userId").is_in(list(eval_users)))
    .group_by("userId")
    .agg(pl.col("movieId").alias("test_items"))
)
test_dict = {}
for row in user_test_items.iter_rows():
    test_dict[row[0]] = set(row[1])
print(f"  Built test sets in {time.time()-t0:.1f}s")

# ── 7. Evaluate: for each user, recommend top-K popular unseen movies ───────

print(f"\nEvaluating on {len(eval_users):,} users...")
t0 = time.time()

ndcg_scores = []
recall_scores = []
hit_scores = []

eval_count = 0
for uid in eval_users:
    seen = seen_dict.get(uid, set())
    truth = test_dict.get(uid, set())
    if not truth:
        continue

    # Get top-K popular movies not in user's training set
    recs = []
    for mid in pop_ranking:
        if mid not in seen:
            recs.append(mid)
            if len(recs) == K:
                break

    # NDCG@K (binary relevance)
    dcg = 0.0
    hits_in_topk = 0
    for rank, mid in enumerate(recs):
        if mid in truth:
            dcg += 1.0 / np.log2(rank + 2)  # rank is 0-indexed, so +2
            hits_in_topk += 1

    # IDCG: best possible DCG with min(|truth|, K) items
    n_relevant = min(len(truth), K)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    # Recall@K
    recall = hits_in_topk / len(truth) if len(truth) > 0 else 0.0

    # Hit Rate@K
    hit = 1.0 if hits_in_topk > 0 else 0.0

    ndcg_scores.append(ndcg)
    recall_scores.append(recall)
    hit_scores.append(hit)
    eval_count += 1

    if eval_count % 20000 == 0:
        print(f"  Evaluated {eval_count:,} users...")

elapsed = time.time() - t0
print(f"  Done. Evaluated {eval_count:,} users in {elapsed:.1f}s")

# ── 8. Results ───────────────────────────────────────────────────────────────

mean_ndcg = np.mean(ndcg_scores)
mean_recall = np.mean(recall_scores)
mean_hit = np.mean(hit_scores)

print("\n" + "=" * 50)
print("RESULTS: Experiment 000 — Popularity Baseline")
print("=" * 50)
print(f"  NDCG@{K}:     {mean_ndcg:.6f}")
print(f"  Recall@{K}:   {mean_recall:.6f}")
print(f"  Hit Rate@{K}: {mean_hit:.6f}")
print(f"\n  Eval users:   {eval_count:,}")
print(f"  Train size:   {len(train):,}")
print(f"  Test size:    {len(test):,}")
print(f"  Cutoff:       2023-07-13")
print("=" * 50)
