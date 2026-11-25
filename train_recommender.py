"""MovieLens interaction training: global chronological holdout, validation-selected ranker, portable artifacts."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,csv,hashlib,io,json,re,zipfile
from pathlib import Path
import numpy as np
import requests
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from recommender import CANDIDATES,make_content,scores,combine,load_artifacts
ROOT=Path(__file__).resolve().parent
URL='https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
LARGE_URL='https://files.grouplens.org/datasets/movielens/ml-25m.zip'
def acquire(path):
    path=Path(path)
    if not path.exists():
        r=requests.get(URL,timeout=90);r.raise_for_status()
        if len(r.content)>20_000_000:raise ValueError('Unexpected dataset size')
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(r.content)
    raw=path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        def rows(name):return list(csv.DictReader(io.StringIO(z.read('ml-latest-small/'+name).decode('utf-8-sig'))))
        films=rows('movies.csv');links={int(x['movieId']):x for x in rows('links.csv')};ratings=rows('ratings.csv')
        license=z.read('ml-latest-small/README.txt').decode('utf-8')
    movies=[]
    for x in films:
        mid=int(x['movieId']);t=x['title'];year=re.search(r'\((\d{4})\)\s*$',t);y=int(year[1]) if year else 0;title=t[:year.start()].strip() if year else t
        title=re.sub(r'^(.*), (The|A|An)(\s*\(.*\))?$',lambda m:m[2]+' '+m[1]+(m[3] or ''),title)
        link=links.get(mid,{})
        movies.append({'id':mid+1000000,'movielens_id':mid,'title':title,'year':y,'genres':[g for g in x['genres'].split('|') if g!='(no genres listed)'],'directors':'','actors':'','tmdb_id':int(link['tmdbId']) if link.get('tmdbId') else None,'imdb_id':'tt'+link['imdbId'].zfill(7) if link.get('imdbId') else None})
    ids={m['movielens_id']:i for i,m in enumerate(movies)}
    users=sorted({int(x['userId']) for x in ratings});uix={u:i for i,u in enumerate(users)}
    a=np.array([(uix[int(x['userId'])],ids[int(x['movieId'])],float(x['rating']),int(x['timestamp'])) for x in ratings],dtype=np.float64)
    a=a[np.argsort(a[:,3],kind='stable')]
    assert len(movies)>9000 and len(a)>100000 and len(users)>=600 and len({(int(x[0]),int(x[1])) for x in a})==len(a)
    assert np.isfinite(a).all() and ((a[:,2]>=.5)&(a[:,2]<=5)).all()
    return movies,a,len(users),hashlib.sha256(raw).hexdigest(),license
def acquire_large(path):
    path=Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix('.part')
        try:
            with requests.get(LARGE_URL,stream=True,timeout=(20,120)) as r:
                r.raise_for_status()
                size=0
                with temp.open('wb') as f:
                    for chunk in r.iter_content(1024*1024):
                        size+=len(chunk)
                        if size>300_000_000:raise ValueError('Unexpected archive size')
                        f.write(chunk)
            temp.replace(path)
        finally:
            if temp.exists():temp.unlink()
    with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    with zipfile.ZipFile(path) as z:
        def rows(name):
            with z.open('ml-25m/'+name) as stream:
                yield from csv.DictReader(io.TextIOWrapper(stream,encoding='utf-8-sig'))
        films=list(rows('movies.csv'));links={int(x['movieId']):x for x in rows('links.csv')}
        # Deterministic approximately 1/32 user sample. All rating interactions of sampled users.
        interactions=[(int(x['userId']),int(x['movieId']),float(x['rating']),int(x['timestamp'])) for x in rows('ratings.csv') if int(x['userId'])%32==0]
        license=z.read('ml-25m/README.txt').decode('utf-8')
    raw=np.array(interactions,dtype=np.float64);raw=raw[np.argsort(raw[:,3],kind='stable')]
    assert len(raw)>500000 and len(np.unique(raw[:,0]))>=4000
    t1,t2=np.quantile(raw[:,3],[.8,.9],method='nearest')
    pre=raw[raw[:,3]<t1]
    counts=np.bincount(pre[:,1].astype(int))
    # Bound serving memory: most frequent 12,000 items by training counts only.
    ids=np.flatnonzero(counts>0);ids=ids[np.lexsort((ids,-counts[ids]))[:12000]]
    chosen=set(map(int,ids));movies=[]
    for x in films:
        mid=int(x['movieId'])
        if mid not in chosen:continue
        t=x['title'];year=re.search(r'\((\d{4})\)\s*positive,seen,nu,ni,epochs=18,dim=32):
    rng=np.random.default_rng(42);U=rng.normal(0,.08,(nu,dim)).astype(np.float32);V=rng.normal(0,.08,(ni,dim)).astype(np.float32)
    pool=np.flatnonzero(np.bincount(seen[:,1].astype(int),minlength=ni));known=[set() for _ in range(nu)]
    for u,i,_,_ in seen:known[int(u)].add(int(i))
    if not len(positive):return V
    pairs=positive[:,:2].astype(int)
    for _ in range(epochs):
        order=rng.permutation(len(pairs))
        for start in range(0,len(order),512):
            b=pairs[order[start:start+512]];u=b[:,0];i=b[:,1];j=rng.choice(pool,len(b))
            for k in range(len(j)):
                for _try in range(40):
                    if j[k] not in known[u[k]]:break
                    j[k]=rng.choice(pool)
            ok=np.array([v not in known[uu] for uu,v in zip(u,j)]);u,i,j=u[ok],i[ok],j[ok]
            if not len(u):continue
            a=U[u].copy();x=V[i].copy();y=V[j].copy();g=1/(1+np.exp(np.clip(np.sum(a*(x-y),axis=1),-30,30)))
            lr=.035;reg=.015
            np.add.at(U,u,lr*(g[:,None]*(x-y)-reg*a))
            np.add.at(V,i,lr*(g[:,None]*a-reg*x))
            np.add.at(V,j,lr*(-g[:,None]*a-reg*y))
    return normalize(V).astype(np.float32)
def fit(movies,a,nu,content):
    ni=len(movies);pos=a[a[:,2]>=4];u=pos[:,0].astype(int);i=pos[:,1].astype(int)
    R=sparse.csr_matrix((np.ones(len(pos),np.float32),(u,i)),shape=(nu,ni));R.sum_duplicates();R.data[:]=1
    # Co-liked films: cosine normalisation, support shrinkage and sparse top-80 neighbours.
    count=np.asarray(R.sum(axis=0)).ravel();co=(R.T@R).tocsr();co.setdiag(0);co.eliminate_zeros()
    coo=co.tocoo();v=coo.data/(np.sqrt(count[coo.row]*count[coo.col])+1e-8)*coo.data/(coo.data+10)
    co=sparse.csr_matrix((v,(coo.row,coo.col)),shape=(ni,ni));rr=[];cc=[];vv=[]
    for k in range(ni):
        lo,hi=co.indptr[k:k+2];take=np.argsort(-co.data[lo:hi])[:80]+lo;rr.extend([k]*len(take));cc.extend(co.indices[take]);vv.extend(co.data[take])
    knn=sparse.csr_matrix((np.array(vv,dtype=np.float32),(rr,cc)),shape=(ni,ni))
    # EASE: regularised closed-form linear item autoencoder on 2,000 most-supported positive items.
    active=np.argsort(-count,kind='stable')[:min(2000,int((count>0).sum()))]
    G=(R[:,active].T@R[:,active]).toarray().astype(np.float64)
    G.flat[::len(active)+1]+=300
    inv=np.linalg.solve(G,np.eye(len(active)))
    B=-inv/np.diag(inv)[None,:];np.fill_diagonal(B,0)
    # Retain strongest 100 absolute coefficients per item, with signed weights.
    er=[];ec=[];ev=[]
    for ix,k in enumerate(active):
        j=np.argsort(-np.abs(B[ix]),kind='stable')[:100]
        er.extend([int(k)]*len(j));ec.extend(active[j]);ev.extend(B[ix,j])
    ease=sparse.csr_matrix((np.asarray(ev,dtype=np.float32),(er,ec)),shape=(ni,ni))
    # SVD learns item embeddings from users' positive interaction matrix.
    factors={}
    for d in (32,64):
        svd=TruncatedSVD(n_components=min(d,nu-1,ni-1),random_state=42,n_iter=7)
        svd.fit(normalize(R,norm='l2',axis=1));factors['svd'+str(d)]=normalize(svd.components_.T*svd.singular_values_**.5).astype(np.float32)
    factors['bpr']=fit_bpr(pos,a,nu,ni)
    # Unrated movies have no collaborative representation.
    for v in factors.values():v[count==0]=0
    return {'movies':movies,'content':content,'knn':knn,'ease':ease,'pop':np.log1p(count).astype(np.float32),**factors}
def evaluate(model,history,future,nu,k=10):
    warm=np.bincount(history[:,1].astype(int),minlength=len(model['movies']))>0
    accum={name:[] for name in CANDIDATES};hits={name:set() for name in CANDIDATES};skipped=0;cold=0;total_positive=0
    for u in range(nu):
        h=history[history[:,0]==u];f=future[(future[:,0]==u)&(future[:,2]>=4)]
        if not len(f):continue
        seen=set(h[:,1].astype(int));likes=h[h[:,2]>=4][-30:,1].astype(int).tolist();dis=h[h[:,2]<=2][-15:,1].astype(int).tolist()
        relevant=set(f[:,1].astype(int))-seen;total_positive+=len(relevant);cold+=sum(not warm[i] for i in relevant);relevant={i for i in relevant if warm[i]}
        if len(likes)<3 or not relevant:skipped+=1;continue
        parts=scores(model,likes,dis);mask=~warm.copy();mask[list(seen)]=True
        for name in CANDIDATES:
            s=combine(parts,name).copy();s[mask]=-np.inf;order=np.argsort(-s,kind='stable')[:k];rel=np.array([int(i in relevant) for i in order]);n=rel.sum()
            dcg=(rel/np.log2(np.arange(2,len(rel)+2))).sum();ideal=(1/np.log2(np.arange(2,min(k,len(relevant))+2))).sum()
            accum[name].append([n/k,n/len(relevant),dcg/ideal,float(n>0)]);hits[name].update(map(int,order))
    if not accum['popularity']:raise RuntimeError('No eligible users in temporal split')
    report={}
    for name,values in accum.items():
        v=np.mean(values,axis=0);report[name]={'precision_at_10':float(v[0]),'recall_at_10':float(v[1]),'ndcg_at_10':float(v[2]),'hit_rate_at_10':float(v[3]),'coverage':len(hits[name])/int(warm.sum()),'users':len(values)}
    return report,{'skipped_users':skipped,'cold_positive_items':cold,'total_positive_items':total_positive,'warm_candidates':int(warm.sum())},accum
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default=str(ROOT/'data'/'ml-latest-small.zip'));p.add_argument('--out',default=str(ROOT/'model'));p.add_argument('--dataset',choices=['small','25m'],default='small');args=p.parse_args()
    if args.dataset=='25m':
        data=ROOT/'data'/'ml-25m.zip' if args.data==str(ROOT/'data'/'ml-latest-small.zip') else args.data
        movies,a,nu,sha,license,(t1,t2),sampling=acquire_large(data)
    else:
        movies,a,nu,sha,license=acquire(args.data)
        t1,t2=np.quantile(a[:,3],[.8,.9],method='nearest');sampling={'sampling':'all users and films'}
    content=make_content(movies)
    train=a[a[:,3]<t1];val=a[(a[:,3]>=t1)&(a[:,3]<t2)];test=a[a[:,3]>=t2]
    assert train[:,3].max()<val[:,3].min()<=val[:,3].max()<test[:,3].min()
    model=fit(movies,train,nu,content);v,vc,_=evaluate(model,train,val,nu)
    champion=max(v,key=lambda k:(v[k]['ndcg_at_10'],v[k]['recall_at_10']));print('Validation champion:',champion,flush=True)
    # Refit through validation, freeze selection, evaluate test only once.
    hist=np.concatenate([train,val]);m=fit(movies,hist,nu,content);t,tc,rows=evaluate(m,hist,test,nu)
    rng=np.random.default_rng(42);diff=np.array(rows[champion])[:,2]-np.array(rows['popularity'])[:,2]
    boot=[rng.choice(diff,len(diff),replace=True).mean() for _ in range(2000)]
    report={'dataset':'MovieLens 25M deterministic user sample' if args.dataset=='25m' else 'MovieLens latest-small','source':LARGE_URL if args.dataset=='25m' else URL,'sampling':sampling,'sha256':sha,'ratings':len(a),'users':nu,'movies':len(movies),'split':{'train':len(train),'validation':len(val),'test':len(test),'validation_start_timestamp':int(t1),'test_start_timestamp':int(t2)},'champion':champion,'validation':v,'test':t,'validation_population':vc,'test_population':tc,'ndcg_gain_ci_vs_popularity':list(map(float,np.quantile(boot,[.025,.975]))),'note':'Global chronological evaluation; warm items and users with >=3 positive history ratings, full warm-catalogue ranking, all previously rated items excluded. Ratings >=4 relevant. Diversity not applied in evaluation. Historical benchmark, not live user satisfaction. No test-set model selection.'}
    print(json.dumps(report,indent=2),flush=True)
    # Full-data fit for deployment only after untouched test evaluation.
    final=fit(movies,a,nu,content);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'catalog.json').write_text(json.dumps(movies,ensure_ascii=False));(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False));(out/'MOVIELENS_LICENSE.txt').write_text(license)
    np.savez_compressed(out/'factors.npz',**{k:final[k] for k in ('pop','svd32','svd64','bpr')});sparse.save_npz(out/'content.npz',content);sparse.save_npz(out/'knn.npz',final['knn']);sparse.save_npz(out/'ease.npz',final['ease'])
    loaded,_=load_artifacts(out);assert loaded['champion']==champion
    print('Training and artifact validation complete',flush=True)
if __name__=='__main__':main()
,t);y=int(year[1]) if year else 0;title=t[:year.start()].strip() if year else t
        title=re.sub(r'^(.*), (The|A|An)(\s*\(.*\))?positive,seen,nu,ni,epochs=18,dim=32):
    rng=np.random.default_rng(42);U=rng.normal(0,.08,(nu,dim)).astype(np.float32);V=rng.normal(0,.08,(ni,dim)).astype(np.float32)
    pool=np.flatnonzero(np.bincount(seen[:,1].astype(int),minlength=ni));known=[set() for _ in range(nu)]
    for u,i,_,_ in seen:known[int(u)].add(int(i))
    if not len(positive):return V
    pairs=positive[:,:2].astype(int)
    for _ in range(epochs):
        order=rng.permutation(len(pairs))
        for start in range(0,len(order),512):
            b=pairs[order[start:start+512]];u=b[:,0];i=b[:,1];j=rng.choice(pool,len(b))
            for k in range(len(j)):
                for _try in range(40):
                    if j[k] not in known[u[k]]:break
                    j[k]=rng.choice(pool)
            ok=np.array([v not in known[uu] for uu,v in zip(u,j)]);u,i,j=u[ok],i[ok],j[ok]
            if not len(u):continue
            a=U[u].copy();x=V[i].copy();y=V[j].copy();g=1/(1+np.exp(np.clip(np.sum(a*(x-y),axis=1),-30,30)))
            lr=.035;reg=.015
            np.add.at(U,u,lr*(g[:,None]*(x-y)-reg*a))
            np.add.at(V,i,lr*(g[:,None]*a-reg*x))
            np.add.at(V,j,lr*(-g[:,None]*a-reg*y))
    return normalize(V).astype(np.float32)
def fit(movies,a,nu,content):
    ni=len(movies);pos=a[a[:,2]>=4];u=pos[:,0].astype(int);i=pos[:,1].astype(int)
    R=sparse.csr_matrix((np.ones(len(pos),np.float32),(u,i)),shape=(nu,ni));R.sum_duplicates();R.data[:]=1
    # Co-liked films: cosine normalisation, support shrinkage and sparse top-80 neighbours.
    count=np.asarray(R.sum(axis=0)).ravel();co=(R.T@R).tocsr();co.setdiag(0);co.eliminate_zeros()
    coo=co.tocoo();v=coo.data/(np.sqrt(count[coo.row]*count[coo.col])+1e-8)*coo.data/(coo.data+10)
    co=sparse.csr_matrix((v,(coo.row,coo.col)),shape=(ni,ni));rr=[];cc=[];vv=[]
    for k in range(ni):
        lo,hi=co.indptr[k:k+2];take=np.argsort(-co.data[lo:hi])[:80]+lo;rr.extend([k]*len(take));cc.extend(co.indices[take]);vv.extend(co.data[take])
    knn=sparse.csr_matrix((np.array(vv,dtype=np.float32),(rr,cc)),shape=(ni,ni))
    # SVD learns item embeddings from users' positive interaction matrix.
    factors={}
    for d in (32,64):
        svd=TruncatedSVD(n_components=min(d,nu-1,ni-1),random_state=42,n_iter=7)
        svd.fit(normalize(R,norm='l2',axis=1));factors['svd'+str(d)]=normalize(svd.components_.T*svd.singular_values_**.5).astype(np.float32)
    factors['bpr']=fit_bpr(pos,a,nu,ni)
    # Unrated movies have no collaborative representation.
    for v in factors.values():v[count==0]=0
    return {'movies':movies,'content':content,'knn':knn,'pop':np.log1p(count).astype(np.float32),**factors}
def evaluate(model,history,future,nu,k=10):
    warm=np.bincount(history[:,1].astype(int),minlength=len(model['movies']))>0
    accum={name:[] for name in CANDIDATES};hits={name:set() for name in CANDIDATES};skipped=0;cold=0;total_positive=0
    for u in range(nu):
        h=history[history[:,0]==u];f=future[(future[:,0]==u)&(future[:,2]>=4)]
        if not len(f):continue
        seen=set(h[:,1].astype(int));likes=h[h[:,2]>=4][-30:,1].astype(int).tolist();dis=h[h[:,2]<=2][-15:,1].astype(int).tolist()
        relevant=set(f[:,1].astype(int))-seen;total_positive+=len(relevant);cold+=sum(not warm[i] for i in relevant);relevant={i for i in relevant if warm[i]}
        if len(likes)<3 or not relevant:skipped+=1;continue
        parts=scores(model,likes,dis);mask=~warm.copy();mask[list(seen)]=True
        for name in CANDIDATES:
            s=combine(parts,name).copy();s[mask]=-np.inf;order=np.argsort(-s,kind='stable')[:k];rel=np.array([int(i in relevant) for i in order]);n=rel.sum()
            dcg=(rel/np.log2(np.arange(2,len(rel)+2))).sum();ideal=(1/np.log2(np.arange(2,min(k,len(relevant))+2))).sum()
            accum[name].append([n/k,n/len(relevant),dcg/ideal,float(n>0)]);hits[name].update(map(int,order))
    if not accum['popularity']:raise RuntimeError('No eligible users in temporal split')
    report={}
    for name,values in accum.items():
        v=np.mean(values,axis=0);report[name]={'precision_at_10':float(v[0]),'recall_at_10':float(v[1]),'ndcg_at_10':float(v[2]),'hit_rate_at_10':float(v[3]),'coverage':len(hits[name])/int(warm.sum()),'users':len(values)}
    return report,{'skipped_users':skipped,'cold_positive_items':cold,'total_positive_items':total_positive,'warm_candidates':int(warm.sum())},accum
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default=str(ROOT/'data'/'ml-latest-small.zip'));p.add_argument('--out',default=str(ROOT/'model'));args=p.parse_args()
    movies,a,nu,sha,license=acquire(args.data);content=make_content(movies)
    t1,t2=np.quantile(a[:,3],[.8,.9],method='nearest');train=a[a[:,3]<t1];val=a[(a[:,3]>=t1)&(a[:,3]<t2)];test=a[a[:,3]>=t2]
    assert train[:,3].max()<val[:,3].min()<=val[:,3].max()<test[:,3].min()
    model=fit(movies,train,nu,content);v,vc,_=evaluate(model,train,val,nu)
    champion=max(v,key=lambda k:(v[k]['ndcg_at_10'],v[k]['recall_at_10']));print('Validation champion:',champion,flush=True)
    # Refit through validation, freeze selection, evaluate test only once.
    hist=np.concatenate([train,val]);m=fit(movies,hist,nu,content);t,tc,rows=evaluate(m,hist,test,nu)
    rng=np.random.default_rng(42);diff=np.array(rows[champion])[:,2]-np.array(rows['popularity'])[:,2]
    boot=[rng.choice(diff,len(diff),replace=True).mean() for _ in range(2000)]
    report={'dataset':'MovieLens latest-small','source':URL,'sha256':sha,'ratings':len(a),'users':nu,'movies':len(movies),'split':{'train':len(train),'validation':len(val),'test':len(test),'validation_start_timestamp':int(t1),'test_start_timestamp':int(t2)},'champion':champion,'validation':v,'test':t,'validation_population':vc,'test_population':tc,'ndcg_gain_ci_vs_popularity':list(map(float,np.quantile(boot,[.025,.975]))),'note':'Global chronological evaluation; warm items and users with >=3 positive history ratings, full warm-catalogue ranking, all previously rated items excluded. Ratings >=4 relevant. Diversity not applied in evaluation. Historical benchmark, not live user satisfaction. No test-set model selection.'}
    print(json.dumps(report,indent=2),flush=True)
    # Full-data fit for deployment only after untouched test evaluation.
    final=fit(movies,a,nu,content);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'catalog.json').write_text(json.dumps(movies,ensure_ascii=False));(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False));(out/'MOVIELENS_LICENSE.txt').write_text(license)
    np.savez_compressed(out/'factors.npz',**{k:final[k] for k in ('pop','svd32','svd64','bpr')});sparse.save_npz(out/'content.npz',content);sparse.save_npz(out/'knn.npz',final['knn'])
    loaded,_=load_artifacts(out);assert loaded['champion']==champion
    print('Training and artifact validation complete',flush=True)
if __name__=='__main__':main()
,lambda m:m[2]+' '+m[1]+(m[3] or ''),title)
        link=links.get(mid,{})
        movies.append({'id':mid+1000000,'movielens_id':mid,'title':title,'year':y,'genres':[g for g in x['genres'].split('|') if g!='(no genres listed)'],'directors':'','actors':'','tmdb_id':int(link['tmdbId']) if link.get('tmdbId') else None,'imdb_id':'tt'+link['imdbId'].zfill(7) if link.get('imdbId') else None})
    midx={m['movielens_id']:i for i,m in enumerate(movies)};users=sorted(set(raw[:,0].astype(int)));uix={u:i for i,u in enumerate(users)}
    included=np.array([int(i) in midx for i in raw[:,1]],dtype=bool)
    excluded={'sampled_ratings':len(raw),'excluded_catalogue_ratings':int((~included).sum()),'excluded_test_positive_items':int(((~included)&(raw[:,3]>=t2)&(raw[:,2]>=4)).sum()),'source_catalogue_movies':len(films),'sampling':'userId modulo 32 = 0; up to 12,000 items selected by pre-validation interaction counts'}
    a=np.array([(uix[int(u)],midx[int(i)],r,t) for u,i,r,t in raw[included]],dtype=np.float64)
    assert len(movies)>9000 and len(a)>500000
    return movies,a,len(users),sha,license,(t1,t2),excluded
def fit_bpr(positive,seen,nu,ni,epochs=18,dim=32):
    rng=np.random.default_rng(42);U=rng.normal(0,.08,(nu,dim)).astype(np.float32);V=rng.normal(0,.08,(ni,dim)).astype(np.float32)
    pool=np.flatnonzero(np.bincount(seen[:,1].astype(int),minlength=ni));known=[set() for _ in range(nu)]
    for u,i,_,_ in seen:known[int(u)].add(int(i))
    if not len(positive):return V
    pairs=positive[:,:2].astype(int)
    for _ in range(epochs):
        order=rng.permutation(len(pairs))
        for start in range(0,len(order),512):
            b=pairs[order[start:start+512]];u=b[:,0];i=b[:,1];j=rng.choice(pool,len(b))
            for k in range(len(j)):
                for _try in range(40):
                    if j[k] not in known[u[k]]:break
                    j[k]=rng.choice(pool)
            ok=np.array([v not in known[uu] for uu,v in zip(u,j)]);u,i,j=u[ok],i[ok],j[ok]
            if not len(u):continue
            a=U[u].copy();x=V[i].copy();y=V[j].copy();g=1/(1+np.exp(np.clip(np.sum(a*(x-y),axis=1),-30,30)))
            lr=.035;reg=.015
            np.add.at(U,u,lr*(g[:,None]*(x-y)-reg*a))
            np.add.at(V,i,lr*(g[:,None]*a-reg*x))
            np.add.at(V,j,lr*(-g[:,None]*a-reg*y))
    return normalize(V).astype(np.float32)
def fit(movies,a,nu,content):
    ni=len(movies);pos=a[a[:,2]>=4];u=pos[:,0].astype(int);i=pos[:,1].astype(int)
    R=sparse.csr_matrix((np.ones(len(pos),np.float32),(u,i)),shape=(nu,ni));R.sum_duplicates();R.data[:]=1
    # Co-liked films: cosine normalisation, support shrinkage and sparse top-80 neighbours.
    count=np.asarray(R.sum(axis=0)).ravel();co=(R.T@R).tocsr();co.setdiag(0);co.eliminate_zeros()
    coo=co.tocoo();v=coo.data/(np.sqrt(count[coo.row]*count[coo.col])+1e-8)*coo.data/(coo.data+10)
    co=sparse.csr_matrix((v,(coo.row,coo.col)),shape=(ni,ni));rr=[];cc=[];vv=[]
    for k in range(ni):
        lo,hi=co.indptr[k:k+2];take=np.argsort(-co.data[lo:hi])[:80]+lo;rr.extend([k]*len(take));cc.extend(co.indices[take]);vv.extend(co.data[take])
    knn=sparse.csr_matrix((np.array(vv,dtype=np.float32),(rr,cc)),shape=(ni,ni))
    # SVD learns item embeddings from users' positive interaction matrix.
    factors={}
    for d in (32,64):
        svd=TruncatedSVD(n_components=min(d,nu-1,ni-1),random_state=42,n_iter=7)
        svd.fit(normalize(R,norm='l2',axis=1));factors['svd'+str(d)]=normalize(svd.components_.T*svd.singular_values_**.5).astype(np.float32)
    factors['bpr']=fit_bpr(pos,a,nu,ni)
    # Unrated movies have no collaborative representation.
    for v in factors.values():v[count==0]=0
    return {'movies':movies,'content':content,'knn':knn,'pop':np.log1p(count).astype(np.float32),**factors}
def evaluate(model,history,future,nu,k=10):
    warm=np.bincount(history[:,1].astype(int),minlength=len(model['movies']))>0
    accum={name:[] for name in CANDIDATES};hits={name:set() for name in CANDIDATES};skipped=0;cold=0;total_positive=0
    for u in range(nu):
        h=history[history[:,0]==u];f=future[(future[:,0]==u)&(future[:,2]>=4)]
        if not len(f):continue
        seen=set(h[:,1].astype(int));likes=h[h[:,2]>=4][-30:,1].astype(int).tolist();dis=h[h[:,2]<=2][-15:,1].astype(int).tolist()
        relevant=set(f[:,1].astype(int))-seen;total_positive+=len(relevant);cold+=sum(not warm[i] for i in relevant);relevant={i for i in relevant if warm[i]}
        if len(likes)<3 or not relevant:skipped+=1;continue
        parts=scores(model,likes,dis);mask=~warm.copy();mask[list(seen)]=True
        for name in CANDIDATES:
            s=combine(parts,name).copy();s[mask]=-np.inf;order=np.argsort(-s,kind='stable')[:k];rel=np.array([int(i in relevant) for i in order]);n=rel.sum()
            dcg=(rel/np.log2(np.arange(2,len(rel)+2))).sum();ideal=(1/np.log2(np.arange(2,min(k,len(relevant))+2))).sum()
            accum[name].append([n/k,n/len(relevant),dcg/ideal,float(n>0)]);hits[name].update(map(int,order))
    if not accum['popularity']:raise RuntimeError('No eligible users in temporal split')
    report={}
    for name,values in accum.items():
        v=np.mean(values,axis=0);report[name]={'precision_at_10':float(v[0]),'recall_at_10':float(v[1]),'ndcg_at_10':float(v[2]),'hit_rate_at_10':float(v[3]),'coverage':len(hits[name])/int(warm.sum()),'users':len(values)}
    return report,{'skipped_users':skipped,'cold_positive_items':cold,'total_positive_items':total_positive,'warm_candidates':int(warm.sum())},accum
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default=str(ROOT/'data'/'ml-latest-small.zip'));p.add_argument('--out',default=str(ROOT/'model'));args=p.parse_args()
    movies,a,nu,sha,license=acquire(args.data);content=make_content(movies)
    t1,t2=np.quantile(a[:,3],[.8,.9],method='nearest');train=a[a[:,3]<t1];val=a[(a[:,3]>=t1)&(a[:,3]<t2)];test=a[a[:,3]>=t2]
    assert train[:,3].max()<val[:,3].min()<=val[:,3].max()<test[:,3].min()
    model=fit(movies,train,nu,content);v,vc,_=evaluate(model,train,val,nu)
    champion=max(v,key=lambda k:(v[k]['ndcg_at_10'],v[k]['recall_at_10']));print('Validation champion:',champion,flush=True)
    # Refit through validation, freeze selection, evaluate test only once.
    hist=np.concatenate([train,val]);m=fit(movies,hist,nu,content);t,tc,rows=evaluate(m,hist,test,nu)
    rng=np.random.default_rng(42);diff=np.array(rows[champion])[:,2]-np.array(rows['popularity'])[:,2]
    boot=[rng.choice(diff,len(diff),replace=True).mean() for _ in range(2000)]
    report={'dataset':'MovieLens latest-small','source':URL,'sha256':sha,'ratings':len(a),'users':nu,'movies':len(movies),'split':{'train':len(train),'validation':len(val),'test':len(test),'validation_start_timestamp':int(t1),'test_start_timestamp':int(t2)},'champion':champion,'validation':v,'test':t,'validation_population':vc,'test_population':tc,'ndcg_gain_ci_vs_popularity':list(map(float,np.quantile(boot,[.025,.975]))),'note':'Global chronological evaluation; warm items and users with >=3 positive history ratings, full warm-catalogue ranking, all previously rated items excluded. Ratings >=4 relevant. Diversity not applied in evaluation. Historical benchmark, not live user satisfaction. No test-set model selection.'}
    print(json.dumps(report,indent=2),flush=True)
    # Full-data fit for deployment only after untouched test evaluation.
    final=fit(movies,a,nu,content);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    (out/'catalog.json').write_text(json.dumps(movies,ensure_ascii=False));(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False));(out/'MOVIELENS_LICENSE.txt').write_text(license)
    np.savez_compressed(out/'factors.npz',**{k:final[k] for k in ('pop','svd32','svd64','bpr')});sparse.save_npz(out/'content.npz',content);sparse.save_npz(out/'knn.npz',final['knn'])
    loaded,_=load_artifacts(out);assert loaded['champion']==champion
    print('Training and artifact validation complete',flush=True)
if __name__=='__main__':main()
