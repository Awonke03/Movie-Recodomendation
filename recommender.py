"""Portable ranking components shared by offline training and Flask inference."""
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer

CANDIDATES={
 'popularity':{'pop':1},
 'content':{'content':1},
 'item_knn':{'knn':1},
 'svd_32':{'svd32':1},
 'svd_64':{'svd64':1},
 'bpr_32':{'bpr':1},
 'hybrid_graph':{'knn':.65,'content':.25,'pop':.10},
 'hybrid_latent':{'svd64':.55,'knn':.25,'content':.15,'pop':.05},
 'hybrid_bpr':{'bpr':.50,'knn':.30,'content':.15,'pop':.05},
 'hybrid_all':{'svd32':.25,'svd64':.15,'bpr':.20,'knn':.25,'content':.10,'pop':.05},
 'hybrid_discovery':{'knn':.35,'svd64':.25,'bpr':.15,'content':.25},
 'ease':{'ease':1},
 'hybrid_ease':{'ease':.55,'knn':.20,'bpr':.15,'content':.10},
 'hybrid_balanced':{'ease':.25,'knn':.30,'bpr':.30,'content':.10,'pop':.05},
}
def make_content(movies):
    text=[' '.join('genre_'+g.replace(' ','_') for g in m['genres'])+' decade_'+str((m['year'] or 0)//10*10) for m in movies]
    return TfidfVectorizer(token_pattern=r'(?u)\b[\w-]+\b',sublinear_tf=True).fit_transform(text).astype(np.float32).tocsr()
def scaled(x):
    # Rank by nonnegative relative score; do not give arbitrary ranks to ties.
    from scipy.stats import rankdata
    x=np.asarray(x).ravel()
    return ((rankdata(x,method='average')-1)/max(1,len(x)-1)).astype(np.float32)
def scores(model,likes,dislikes=()):
    n=len(model['pop'])
    if not likes:return {k:model['pop'] for k in ['pop','content','knn','svd32','svd64','bpr','ease']}
    pos=np.asarray(likes,dtype=int);neg=np.asarray(dislikes,dtype=int)
    def sparse_score(mat):
        # matrices for content and collaborative similarity have distinct shapes
        p=np.asarray(mat[pos].mean(axis=0)).ravel()
        if len(neg):p-=.45*np.asarray(mat[neg].mean(axis=0)).ravel()
        return p
    content=model['content'];cp=sparse_score(content)
    raw={'pop':model['pop'],'content':np.asarray(content@cp).ravel(),'knn':sparse_score(model['knn']),'ease':sparse_score(model['ease'])}
    for k in ['svd32','svd64','bpr']:
        e=model[k];p=e[pos].mean(axis=0)
        if len(neg):p=p-.45*e[neg].mean(axis=0)
        raw[k]=e@p
    return {k:scaled(v) for k,v in raw.items()}
def combine(parts,name):
    return sum(w*parts[k] for k,w in CANDIDATES[name].items())
def rank(model,likes,dislikes=(),exclude=(),genre='',limit=24,name=None,diversity=False):
    p=scores(model,likes,dislikes);s=combine(p,name or model['champion'])
    s=np.array(s,copy=True);blocked=set(likes)|set(dislikes)|set(exclude)
    for i in blocked:s[i]=-np.inf
    if genre:
        for i,m in enumerate(model['movies']):
            if genre not in m['genres']:s[i]=-np.inf
    order=np.argsort(-s,kind='stable');out=[];genre_counts={}
    # Optional light diversity: no more than four movies with same exact genre tuple per first page when alternatives exist.
    for i in order:
        if not np.isfinite(s[i]):break
        g=tuple(sorted(model['movies'][i]['genres']))
        if diversity and genre_counts.get(g,0)>=4:continue
        genre_counts[g]=genre_counts.get(g,0)+1;out.append(int(i))
        if len(out)>=limit:break
    if len(out)<limit:
        selected=set(out);out.extend(int(i) for i in order if np.isfinite(s[i]) and int(i) not in selected)
    return out[:limit]
def load_artifacts(path):
    import json
    from pathlib import Path
    path=Path(path);movies=json.loads((path/'catalog.json').read_text());report=json.loads((path/'report.json').read_text())
    with np.load(path/'factors.npz',allow_pickle=False) as z:model={k:z[k] for k in z.files}
    model.update(movies=movies,champion=report['champion'],content=sparse.load_npz(path/'content.npz').tocsr(),knn=sparse.load_npz(path/'knn.npz').tocsr(),ease=sparse.load_npz(path/'ease.npz').tocsr())
    n=len(movies)
    assert model['ease'].shape==(n,n) and model['knn'].shape==(n,n) and model['content'].shape[0]==n and all(model[k].shape[0]==n for k in ('pop','svd32','svd64','bpr'))
    assert all(np.isfinite(model[k]).all() for k in ('pop','svd32','svd64','bpr')) and len({m['id'] for m in movies})==n and model['champion'] in CANDIDATES
    return model,report
