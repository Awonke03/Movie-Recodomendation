# CineMatch

Cinematic movie discovery and content-based recommendations. Replaces the older Dash interface while preserving the original CSV and notebook.

- Weighted sparse TF-IDF on genre, director, cast and year; cosine-style content similarity (not a viewer rating or collaborative filtering). Local CSV is loaded directly; no GitHub HTML read or index drift.
- Search, genre/decade filters, similar films, local browser watchlist, details and IMDb/YouTube search links.
- Optional TMDB integration for posters, backdrops, ratings, descriptions, cast, official YouTube trailers and modern movie search. TMDB related films are labelled separately from the local recommender.
- Without credentials, local recommendations work; posters use title tiles. Individual Apple South Africa storefront artwork can be requested in details where exact title/year match exists. No media files are downloaded or cached.
- Historical catalogue, not a complete current movie library. Original file has metadata, not individual user-rating interactions. No recommendation quality claims without relevance labels.
 
## Run
pip install -r requirements.txt
python movie_recommendation.py

## Render
Create a free Blueprint using render.yaml. Optional: set TMDB_READ_TOKEN (API Read Access Token) or TMDB_API_KEY in the service environment, never commit secrets. Obtain a developer token at https://www.themoviedb.org/settings/api. The app still starts without credentials. TMDB usage is subject to its attribution and noncommercial terms; no live metadata requests have been validated without a token. This product uses the TMDB API but is not endorsed or certified by TMDB.

## Tests
pip install pytest playwright
pytest -q
python -m playwright install chromium
python tests/browser_check.py

With TMDB enabled, CineMatch opens on current weekly trending movies, while Discover opens the original local catalogue. Without credentials, the historical catalogue remains the landing page. Trending results and remote related-film recommendations are supplied by TMDB, not the locally fitted model.

## Learned recommendation engine v2
MovieLens latest-small ratings and linked movie identifiers are downloaded from GroupLens during the training build. Eleven ranking candidates: popularity, genre/decade TF-IDF, shrinkage item-item collaborative filtering, rank-32/64 SVD, BPR item embeddings with profile pooling, and five weighted hybrids. Global temporal train/validation/test splits; select NDCG@10 using validation only; refit then test against full warm catalogue, exclude all previously rated films. Report Precision@10, Recall@10, NDCG@10, Hit Rate@10, coverage and paired bootstrap NDCG difference. Cold items/users excluded from warm evaluation are counted. The full-data portable artifacts are generated only after evaluation. For You uses local likes/dislikes and diversity reranking; it is not trained on browser profiles and is not Netflix's production model. Modern TMDB-only titles are outside the MovieLens model and have separate TMDB recommendations. The original catalogue remains available as a development fallback if artifacts are absent. MovieLens IDs use a new namespace, so old local watchlist entries may require reselecting from the new catalogue. Render builds the model before starting inference; a failed dataset download fails the build rather than silently presenting a trained model. No user interaction history is persisted server-side. See model/report.json and the GitHub Actions training artifact for measured results. Dataset terms: https://grouplens.org/datasets/movielens/latest/ .

The training notebook `CineMatch_Training.ipynb` requires pandas for displaying results (`pip install pandas`).

Extended rankers: EASE regularised linear item autoencoder, two EASE hybrids (14 candidates). Optional larger benchmark `python train_recommender.py --dataset 25m --out model-large` samples all interactions of users whose ID is divisible by 32 from MovieLens 25M, uses a training-only 12,000-item catalogue cap and reports excluded ratings. This is a separate historical benchmark, not guaranteed to be disjoint from latest-small; it does not automatically replace the serving artifacts. No claims of independent new users or live engagement are made.

For You now includes genre-based, non-overlapping recommendation shelves derived from the liked movies, with seen preference items excluded and lightweight diversity re-ranking. This is an additional serving feature, not a claim that the shelves were independently user-tested.

## Live movie discovery and trailer panel
TMDB-enabled Discover opens on the trailing 365 days of releases and offers All time, Recent releases, Coming soon, Top rated, Classics, and the local MovieLens catalogue. Live filters include genre and decade with pagination; search can page across TMDB results. A local seed combines the trained recommendations with TMDB related titles, while modern titles outside MovieLens use provider recommendations. No guarantee every movie is indexed or an official trailer is available. The trailer appears on the right of the movie detail modal on desktop and below on small screens; play uses a privacy-enhanced YouTube embed with an external fallback. API token required for live posters, dates and video IDs. Automated tests mock provider responses and do not prove third-party playback.

The larger historical benchmark completed successfully (Actions 45636506922): 745,308 retained ratings, 5,079 users, 12,000 movies. Validation selected item KNN; test NDCG@10 0.1282 vs 0.0974 popularity across 172 eligible users. It remains a separate experiment; Render currently builds the smaller model to limit build costs. The live TMDB catalogue is not restricted by these training catalogues.
