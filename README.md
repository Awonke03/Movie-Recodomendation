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
