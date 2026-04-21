# Data Assessment

## Overview
- **32,000,204 ratings** from **200,948 users** on **84,432 movies** (MovieLens 32M)
- 4 files: `ratings.csv` (836MB), `movies.csv` (4MB), `tags.csv` (69MB), `links.csv` (1.9MB)
- Time span: 1995-01-09 to 2023-10-13
- Matrix sparsity: 99.81% — extremely sparse

## Target: Implicit feedback (watch/not-watch)
- Ratings exist on a 0.5–5.0 scale (half-star increments), but we treat any rating as a positive interaction
- Rating distribution skews positive: mean 3.54, median 3.5. Most common rating is 4.0 (8.4M)
- No missing values, no duplicate (userId, movieId) pairs

## Features

| Column | Source | Dtype | Missing % | Notes |
|--------|--------|-------|-----------|-------|
| userId | ratings | int64 | 0% | 200,948 unique users. Min 20 ratings/user (by design), median 73, mean 159, max 33,332. Heavy long-tail. |
| movieId | ratings | int64 | 0% | 84,432 unique movies. Median 5 ratings/movie, mean 379, max 102,929. Extreme long-tail — 25th percentile is just 2 ratings. |
| rating | ratings | float64 | 0% | Not usable at inference time (can't know rating for unseen movie). DROP. |
| timestamp | ratings | int64 | 0% | Useful for temporal train/test split only. DROP as feature. |
| title | movies | string | 0% | Contains year in parentheses. Could extract year as feature. |
| genres | movies | string | 0% | Pipe-separated, 20 unique genres. 7,080 movies (8.1%) have "(no genres listed)". |
| tag | tags | string | 0% | 2M tags across 51,323 movies. Only 57.1% of rated movies have tags — sparse coverage. |
| imdbId | links | int64 | 0% | External ID — not a model feature. |
| tmdbId | links | int64 | 0.1% | External ID — not a model feature. 124 missing values. |

## Issues Found
- **Extreme item long-tail:** 25% of movies have ≤2 ratings. Cold-start items will be a challenge for collaborative filtering.
- **7,080 movies with no genre info** (8.1% of catalog). These lack content features.
- **3,153 movies in catalog but never rated** — no interaction data, irrelevant for training.
- **Tag coverage is partial** (57.1%) — tags can augment item features but can't be relied on as a primary signal.

## Leakage Risk
- **rating value:** Must not be used as a feature — it's not available at prediction time for unseen movies.
- **timestamp:** Must only be used for temporal splitting, not as a model input (would leak temporal ordering info).
- **tags:** User-generated post-watch. Safe for item-side content features (aggregate tag profiles), but not for user-item features at prediction time.

## Recommended Feature Set
- **Use:** `userId`, `movieId` (core interaction pair for collaborative filtering), `genres` (item content), `year` extracted from `title` (item metadata), aggregated tag profiles per movie (optional content enrichment)
- **Drop:** `rating` (not available at inference), `timestamp` (split use only), `imdbId`/`tmdbId` (external IDs, no signal), raw `tag` rows (aggregate into item features if used)

---
*Assessed on: 2026-04-14*
