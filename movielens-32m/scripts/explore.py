"""
Data exploration for MovieLens 32M recommender system task.
Uses Polars for speed on the 32M-row ratings file.
Run: docker run --rm -v "$(pwd)":/project agentic-automl python scripts/explore.py
"""
import polars as pl
from datetime import datetime

# ── 1. Load all files ────────────────────────────────────────────────────────

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

ratings = pl.read_csv("data/ratings.csv")
movies = pl.read_csv("data/movies.csv")
tags = pl.read_csv("data/tags.csv")
links = pl.read_csv("data/links.csv")

print(f"ratings.csv: {ratings.shape}")
print(f"movies.csv:  {movies.shape}")
print(f"tags.csv:    {tags.shape}")
print(f"links.csv:   {links.shape}")
print()

# ── 2. Ratings — shape, schema, stats ────────────────────────────────────────

print("=" * 70)
print("RATINGS")
print("=" * 70)
print(f"Schema: {ratings.schema}")
print(f"Head:\n{ratings.head()}\n")

null_counts = ratings.null_count()
print(f"Missing values:\n{null_counts}\n")

n_users = ratings["userId"].n_unique()
n_movies = ratings["movieId"].n_unique()
n_ratings = len(ratings)
sparsity = 1 - n_ratings / (n_users * n_movies)

print(f"Unique users:  {n_users:,}")
print(f"Unique movies: {n_movies:,}")
print(f"Total ratings: {n_ratings:,}")
print(f"Sparsity:      {sparsity:.6f} ({sparsity*100:.2f}%)")
print()

# Rating value distribution
print("Rating distribution:")
rating_dist = ratings.group_by("rating").len().sort("rating")
for row in rating_dist.iter_rows():
    print(f"  {row[0]:.1f}: {row[1]:>10,}")
print()

rating_stats = ratings["rating"].describe()
print(f"Rating stats:\n{rating_stats}\n")

# Ratings per user
user_counts = ratings.group_by("userId").len().rename({"len": "count"})
uc = user_counts["count"]
print("Ratings per user:")
print(f"  min={uc.min()}, median={uc.median():.0f}, mean={uc.mean():.1f}, "
      f"max={uc.max()}, std={uc.std():.1f}")
print(f"  Percentiles: 10%={uc.quantile(0.1):.0f}, 25%={uc.quantile(0.25):.0f}, "
      f"75%={uc.quantile(0.75):.0f}, 90%={uc.quantile(0.9):.0f}, "
      f"99%={uc.quantile(0.99):.0f}")
print()

# Ratings per movie
movie_counts = ratings.group_by("movieId").len().rename({"len": "count"})
mc = movie_counts["count"]
print("Ratings per movie:")
print(f"  min={mc.min()}, median={mc.median():.0f}, mean={mc.mean():.1f}, "
      f"max={mc.max()}, std={mc.std():.1f}")
print(f"  Percentiles: 10%={mc.quantile(0.1):.0f}, 25%={mc.quantile(0.25):.0f}, "
      f"75%={mc.quantile(0.75):.0f}, 90%={mc.quantile(0.9):.0f}, "
      f"99%={mc.quantile(0.99):.0f}")
print()

# Timestamp range
ts_min = datetime.utcfromtimestamp(ratings["timestamp"].min())
ts_max = datetime.utcfromtimestamp(ratings["timestamp"].max())
print(f"Timestamp range: {ts_min} to {ts_max}")
print()

# Duplicate check
dupes = ratings.select(pl.struct("userId", "movieId").is_duplicated().sum()).item()
print(f"Duplicate (userId, movieId) pairs: {dupes}")
print()

# ── 3. Movies ────────────────────────────────────────────────────────────────

print("=" * 70)
print("MOVIES")
print("=" * 70)
print(f"Schema: {movies.schema}")
print(f"Head:\n{movies.head()}\n")
print(f"Missing values:\n{movies.null_count()}\n")

# Genre analysis
genres_exploded = movies.select(
    pl.col("genres").str.split("|").explode().alias("genre")
)
genre_counts = genres_exploded.group_by("genre").len().sort("len", descending=True)
print(f"Unique genres: {genres_exploded['genre'].n_unique()}")
print(f"Genre distribution:\n{genre_counts}\n")

no_genre = movies.filter(pl.col("genres") == "(no genres listed)").height
print(f"Movies with no genres listed: {no_genre}")
print()

# Cross-reference with ratings
rated_ids = set(ratings["movieId"].unique().to_list())
catalog_ids = set(movies["movieId"].unique().to_list())
print(f"Movies in ratings but not in movies.csv: {len(rated_ids - catalog_ids)}")
print(f"Movies in movies.csv but not rated: {len(catalog_ids - rated_ids)}")
print()

# ── 4. Tags ──────────────────────────────────────────────────────────────────

print("=" * 70)
print("TAGS")
print("=" * 70)
print(f"Schema: {tags.schema}")
print(f"Head:\n{tags.head()}\n")
print(f"Missing values:\n{tags.null_count()}\n")

tag_users = tags["userId"].n_unique()
tag_movies = tags["movieId"].n_unique()
tags_per_movie = tags.group_by("movieId").len()["len"]
print(f"Unique users who tagged:  {tag_users:,}")
print(f"Unique movies tagged:     {tag_movies:,}")
print(f"Tags per movie (of tagged): mean={tags_per_movie.mean():.1f}, "
      f"median={tags_per_movie.median():.0f}")
# Coverage: what fraction of rated movies have tags?
tagged_ids = set(tags["movieId"].unique().to_list())
print(f"Tag coverage of rated movies: {len(tagged_ids & rated_ids):,} / {len(rated_ids):,} "
      f"({100*len(tagged_ids & rated_ids)/len(rated_ids):.1f}%)")
print()

# ── 5. Links ─────────────────────────────────────────────────────────────────

print("=" * 70)
print("LINKS")
print("=" * 70)
print(f"Schema: {links.schema}")
print(f"Head:\n{links.head()}\n")
print(f"Missing values:\n{links.null_count()}\n")

# ── 6. Leakage check ────────────────────────────────────────────────────────

print("=" * 70)
print("LEAKAGE CHECK")
print("=" * 70)
print("- timestamp: present in ratings. Must use for temporal train/test split only,")
print("  never as a model feature (would leak future info).")
print("- rating value: task is implicit feedback (predict watch/not-watch), so the")
print("  explicit rating is not the target. Could be used as signal strength, but")
print("  not available at inference time for unseen movies — DROP from features.")
print("- tags: user-generated post-watch. Same issue — not available at prediction")
print("  time for unseen movies. Can use for item-side content features only.")
print()

print("=" * 70)
print("DONE")
print("=" * 70)
