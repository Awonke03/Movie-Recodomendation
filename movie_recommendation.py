"""CineMatch: content-based recommendation and optional licensed movie metadata."""
import csv,os,re,math,unicodedata
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode,quote,urlparse
import requests
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
ROOT=Path(__file__).resolve().parent
app=Flask(__name__,static_folder='static',static_url_path='/static')
app.config['MAX_CONTENT_LENGTH']=16384
TOKEN=os.environ.get('TMDB_READ_TOKEN','').strip()
API_KEY=os.environ.get('TMDB_API_KEY','').strip()
def norm(s):
    return ' '.join(re.findall(r'[a-z0-9]+',unicodedata.normalize('NFKD',str(s)).encode('ascii','ignore').decode().lower()))
def display_title(s):
    return s.strip().title() if s.islower() else s.strip()
def parse_movies(path):
    movies=[];seen=set()
    with open(path,encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            title=(row.get('title') or '').strip()
            try:year=int(float(row.get('year') or 0));id=int(row['movie_id'])
            except (ValueError,TypeError,KeyError):continue
            if not title or not 1870<=year<=2100 or id in seen:continue
            seen.add(id)
            movies.append(dict(id=id,title=display_title(title),year=year,directors=(row.get('directors') or '').strip(),actors=(row.get('actors') or '').strip(),genres=(row.get('genres') or '').split()))
    if not movies:raise RuntimeError('No usable movies')
    return movies
from recommender import load_artifacts,rank,scores,combine
TRAINED=None;MODEL_REPORT=None
if (ROOT/'model'/'report.json').exists():
    TRAINED,MODEL_REPORT=load_artifacts(ROOT/'model')
MOVIES=TRAINED['movies'] if TRAINED is not None else parse_movies(ROOT/'movielens_100k.csv')
BY_ID={x['id']:i for i,x in enumerate(MOVIES)}
TMDB_TO_LOCAL={m['tmdb_id']:m for m in MOVIES if m.get('tmdb_id')}
def matrix():
    blocks=[]
    for field,weight in [('genres',.48),('directors',.24),('actors',.23),('year',.05)]:
        vals=[' '.join(m[field]) if isinstance(m[field],list) else str(m[field]) for m in MOVIES]
        vec=TfidfVectorizer(token_pattern=r'(?u)\b\w+\b',ngram_range=(1,2),sublinear_tf=True)
        blocks.append(vec.fit_transform(vals)*math.sqrt(weight))
    return hstack(blocks).tocsr()
MATRIX=None if TRAINED is not None else matrix()
GENRES=sorted({g for m in MOVIES for g in m['genres']})
def compact(m):
    return {k:m[k] for k in ('id','title','year','genres','directors','tmdb_id','imdb_id') if k in m}
def recommend(id,limit=12,genre='',minyear=0,maxyear=2100):
    idx=BY_ID[id]
    if TRAINED is not None:
        parts=scores(TRAINED,[idx]);s=combine(parts,TRAINED['champion'])
        order=rank(TRAINED,[idx],genre=genre,limit=limit)
        return [{**compact(MOVIES[i]),'shared_genres':sorted(set(MOVIES[idx]['genres'])&set(MOVIES[i]['genres'])),'recommendation_type':'learned ranking'} for i in order if minyear<=MOVIES[i]['year']<=maxyear]
    scores_=(MATRIX@MATRIX[idx].T).toarray().ravel()
    seed=MOVIES[idx];ordered=np.lexsort((np.array([m['id'] for m in MOVIES]),-scores_))
    out=[];seen={(norm(seed['title']),seed['year'])}
    for i in ordered:
        m=MOVIES[i];key=(norm(m['title']),m['year'])
        if i==idx or key in seen or scores_[i]<=0 or genre and genre not in m['genres'] or not minyear<=m['year']<=maxyear:continue
        seen.add(key);out.append({**compact(m),'similarity':round(float(scores_[i]),4),'shared_genres':sorted(set(seed['genres'])&set(m['genres']))})
        if len(out)>=limit:break
    return out
def search(q):
    q=norm(q)
    if not q:return []
    terms=q.split()
    return [compact(m) for m in sorted((m for m in MOVIES if all(t in norm(m['title']+' '+str(m['year'])) for t in terms)),key=lambda m:(norm(m['title'])!=q,not norm(m['title']).startswith(q),len(m['title']),-m['year'],m['id']))[:30]]
def tmdb(path,**params):
    if not (TOKEN or API_KEY):return {}
    if API_KEY and not TOKEN:params['api_key']=API_KEY
    try:
        res=requests.get('https://api.themoviedb.org/3/'+path,params=params,headers={'Authorization':'Bearer '+TOKEN} if TOKEN else {},timeout=6)
        res.raise_for_status();return res.json()
    except (requests.RequestException,ValueError):return {}
def tmdb_public(m):
    title=m.get('title','');date=m.get('release_date','');pid=m.get('id')
    if not isinstance(pid,int) or not title:return None
    path=m.get('poster_path');back=m.get('backdrop_path')
    safe=lambda p:bool(isinstance(p,str) and re.fullmatch(r'/[A-Za-z0-9_.-]+',p))
    local=TMDB_TO_LOCAL.get(pid)
    return {**({'id':local['id'],'genres':local['genres'],'imdb_id':local.get('imdb_id')} if local else {}),'provider':'TMDB','tmdb_id':pid,'title':title,'year':int(date[:4]) if date[:4].isdigit() else None,'poster':'https://image.tmdb.org/t/p/w500'+path if safe(path) else None,'backdrop':'https://image.tmdb.org/t/p/w1280'+back if safe(back) else None,'overview':m.get('overview') or '', 'rating':round(m.get('vote_average') or 0,1) if m.get('vote_count',0)>0 else None,'rating_count':m.get('vote_count',0),'url':'https://www.themoviedb.org/movie/'+str(pid)}
def apple_movie(m):
    # Apple storefront metadata only; no downloading/caching of media files.
    try:
        res=requests.get('https://itunes.apple.com/search',params={'term':m['title'],'media':'movie','entity':'movie','country':'ZA','limit':15},timeout=6);res.raise_for_status()
        matches=[x for x in res.json().get('results',[]) if norm(x.get('trackName',''))==norm(m['title']) and str(x.get('releaseDate',''))[:4]==str(m['year'])]
        if len(matches)!=1:return {}
        x=matches[0];url=x.get('trackViewUrl','');art=x.get('artworkUrl100','')
        if not url.startswith('https://') or urlparse(url).hostname not in ('itunes.apple.com','tv.apple.com'):return {}
        if not isinstance(art,str) or urlparse(art).scheme!='https' or not (urlparse(art).hostname or '').endswith('.mzstatic.com'):return {}
        return {'provider':'Apple','poster':art,'url':url,'overview':x.get('longDescription') or x.get('shortDescription') or ''}
    except (requests.RequestException,ValueError,TypeError):return {}
@lru_cache(maxsize=512)
def metadata(id):
    m=MOVIES[BY_ID[id]]
    if TOKEN or API_KEY:
        if m.get('tmdb_id'):
            linked=tmdb('movie/'+str(m['tmdb_id']))
            if linked.get('id')==m['tmdb_id']:return tmdb_public(linked) or {}
        rs=tmdb('search/movie',query=m['title'],year=m['year'],include_adult='false').get('results',[])
        matched=[x for x in rs if norm(x.get('title',''))==norm(m['title']) and str(x.get('release_date',''))[:4]==str(m['year'])]
        if len(matched)==1:return tmdb_public(matched[0]) or {}
    return {}
@app.after_request
def headers(res):
    res.headers['X-Content-Type-Options']='nosniff'
    res.headers['Referrer-Policy']='strict-origin-when-cross-origin'
    res.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' https://image.tmdb.org https://*.mzstatic.com data:; frame-src https://www.youtube-nocookie.com; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
    return res
@app.get('/')
def index():return send_from_directory(app.static_folder,'index.html')
@app.get('/health')
def health():return jsonify(status='ok',movies=len(MOVIES),tmdb=bool(TOKEN or API_KEY),personalisation=TRAINED is not None)
@app.get('/api/trending')
def trending():
    if not (TOKEN or API_KEY):return jsonify(items=[],available=False)
    data=tmdb('trending/movie/week',language='en-US')
    items=[x for x in (tmdb_public(m) for m in data.get('results',[]) if not m.get('adult',False)) if x]
    return jsonify(items=items,available=bool(items))
@app.get('/api/catalog')
def catalog():
    genre=request.args.get('genre','');decade=request.args.get('decade','',type=str);q=request.args.get('q','')[:100]
    if genre and genre not in GENRES:return jsonify(error='Unknown genre'),400
    if decade and not re.fullmatch(r'\d{4}',decade):return jsonify(error='Invalid decade'),400
    page=max(1,min(1000,request.args.get('page',1,type=int)));n=24
    pool=[m for m in MOVIES if (not genre or genre in m['genres']) and (not decade or int(decade)<=m['year']<int(decade)+10) and (not q or norm(q) in norm(m['title']))]
    pool.sort(key=lambda m:(-m['year'],m['id']))
    return jsonify(items=[compact(m) for m in pool[(page-1)*n:page*n]],total=len(pool),genres=GENRES,decades=sorted({m['year']//10*10 for m in MOVIES},reverse=True),page=page)
@app.get('/api/search')
def search_api():
    q=request.args.get('q','')[:100];local=search(q)
    remote=[]
    if len(q.strip())>=2 and (TOKEN or API_KEY):
        remote=[x for x in (tmdb_public(m) for m in tmdb('search/movie',query=q,include_adult='false').get('results',[])) if x][:15]
        keys={(norm(m['title']),m['year']) for m in local};remote=[m for m in remote if (norm(m['title']),m['year']) not in keys]
    return jsonify(local=local,remote=remote)
@app.get('/api/recommend/<int:id>')
def rec_api(id):
    if id not in BY_ID:return jsonify(error='Movie not found'),404
    genre=request.args.get('genre','')
    if genre and genre not in GENRES:return jsonify(error='Unknown genre'),400
    return jsonify(seed=compact(MOVIES[BY_ID[id]]),items=recommend(id,genre=genre))
@app.post('/api/personalised')
def personalised():
    if TRAINED is None:return jsonify(error='Trained model unavailable'),503
    data=request.get_json(silent=True)
    if not isinstance(data,dict):return jsonify(error='Invalid profile'),400
    liked=data.get('likes',[]);disliked=data.get('dislikes',[])
    if not all(isinstance(x,list) and len(x)<=100 and all(type(i) is int for i in x) for x in (liked,disliked)):return jsonify(error='Invalid profile'),400
    if set(liked)&set(disliked):return jsonify(error='Conflicting preferences'),400
    likes=list(dict.fromkeys(BY_ID[i] for i in liked if i in BY_ID));dislikes=list(dict.fromkeys(BY_ID[i] for i in disliked if i in BY_ID))
    genre=data.get('genre','')
    if not isinstance(genre,str) or (genre and genre not in GENRES):return jsonify(error='Unknown genre'),400
    result=rank(TRAINED,likes,dislikes,genre=genre,limit=24,diversity=True)
    return jsonify(items=[compact(MOVIES[i]) for i in result],liked=len(likes),disliked=len(dislikes),mode='personalised' if likes else 'popular')
@app.get('/api/model')
def model_info():
    if MODEL_REPORT is None:return jsonify(available=False)
    return jsonify(available=True,ratings=MODEL_REPORT['ratings'],users=MODEL_REPORT['users'],movies=MODEL_REPORT['movies'],champion=MODEL_REPORT['champion'])
@app.get('/api/movie/<int:id>')
def detail(id):
    if id not in BY_ID:return jsonify(error='Movie not found'),404
    m=MOVIES[BY_ID[id]]
    return jsonify(**compact(m),actors=m['actors'],**{'media':metadata(id)})
@app.get('/api/media/<int:id>')
def media(id):
    if id not in BY_ID:return jsonify(error='Movie not found'),404
    return jsonify(metadata(id))
@app.get('/api/apple/<int:id>')
def apple(id):
    if id not in BY_ID:return jsonify(error='Movie not found'),404
    return jsonify(apple_movie(MOVIES[BY_ID[id]]))
@app.get('/api/tmdb/<int:id>')
def remote_detail(id):
    if not (TOKEN or API_KEY):return jsonify(error='Movie metadata unavailable'),503
    d=tmdb('movie/'+str(id),append_to_response='videos,credits,recommendations')
    m=tmdb_public(d)
    if not m:return jsonify(error='Movie details unavailable'),404
    videos=[v for v in d.get('videos',{}).get('results',[]) if v.get('site')=='YouTube' and v.get('type')=='Trailer' and v.get('official') is True and re.fullmatch(r'[A-Za-z0-9_-]{11}',str(v.get('key','')))]
    m['trailer']=videos[0]['key'] if videos else None
    m['genres']=[g['name'] for g in d.get('genres',[])];m['runtime']=d.get('runtime');m['directors']=', '.join(x['name'] for x in d.get('credits',{}).get('crew',[]) if x.get('job')=='Director');m['actors']=', '.join(x['name'] for x in d.get('credits',{}).get('cast',[])[:8])
    m['related']=[x for x in (tmdb_public(y) for y in d.get('recommendations',{}).get('results',[])) if x][:12]
    return jsonify(m)
if __name__=='__main__':app.run(host='0.0.0.0',port=int(os.environ.get('PORT',8050)))
