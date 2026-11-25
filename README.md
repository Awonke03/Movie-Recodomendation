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
