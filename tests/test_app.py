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

def test_trending_and_metadata_matching(monkeypatch):
 monkeypatch.setattr(m,'TOKEN','test-token')
 movie=m.MOVIES[0]
 item={'id':123,'title':movie['title'],'release_date':str(movie['year'])+'-01-01','poster_path':'/example.jpg','backdrop_path':'/back.jpg','vote_average':7.2,'vote_count':100}
 def fake(path,**params):
  if path.startswith('trending'):return {'results':[item]}
  if path=='search/movie':return {'results':[item,{**item,'id':999,'release_date':'2020-01-01'}]}
  return {**item,'videos':{'results':[{'site':'YouTube','type':'Trailer','official':True,'key':'ABCDEFGHIJK'}]}}
 monkeypatch.setattr(m,'tmdb',fake);m.metadata.cache_clear();c=m.app.test_client()
 assert c.get('/api/trending').get_json()['items'][0]['poster']=='https://image.tmdb.org/t/p/w500/example.jpg'
 assert c.get('/api/media/'+str(movie['id'])).get_json()['tmdb_id']==123
 assert c.get('/api/tmdb/123').get_json()['trailer']=='ABCDEFGHIJK'
 m.metadata.cache_clear()
