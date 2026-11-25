import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import movie_recommendation as m
def test_catalog():
 c=m.app.test_client();assert c.get('/').status_code==200
 d=c.get('/api/catalog').get_json();assert d['total']>1000 and len(d['items'])==24
 assert c.get('/api/catalog?decade=oops').status_code==400
def test_search_and_recommend():
 c=m.app.test_client();s=c.get('/api/search?q=toy%20story').get_json()['local'];assert s
 id=s[0]['id'];r=c.get('/api/recommend/'+str(id)).get_json()['items'];assert r and all(x['id']!=id for x in r)
 assert len({x['id'] for x in r})==len(r)
 assert all(r[i]['similarity']>=r[i+1]['similarity'] for i in range(len(r)-1))
 assert c.get('/api/recommend/999999').status_code==404
def test_genre_filter():
 id=m.MOVIES[0]['id'];g=m.GENRES[0];r=m.recommend(id,genre=g);assert all(g in x['genres'] for x in r)
def test_missing_metadata(monkeypatch):
 monkeypatch.setattr(m,'TOKEN','');monkeypatch.setattr(m,'API_KEY','');m.metadata.cache_clear()
 c=m.app.test_client();assert c.get('/api/media/'+str(m.MOVIES[0]['id'])).get_json()=={}
 assert c.get('/api/tmdb/1').status_code==503
