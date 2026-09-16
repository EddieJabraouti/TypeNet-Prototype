"""Matched, participant-disjoint 5-versus-10 accumulated-window experiment.

Run --phase develop, then --phase seal. --smoke uses training identities only.
No previously trained model or TypeNet embedding is used.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
import time

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from paper_typenet.nn import events_to_features
from prototype_net.exp3.features import enrollment, session_vector, bundle_vector
from prototype_net.exp3.run import ROOT, DATA, MANIFEST, digest, write_json, calibrated, metrics
from prototype_net.perturbation_generator.perturb import PerturbationConfig, StructuredTimingPerturber

HERE=Path(__file__).resolve().parent
SEED=9172026
BUDGETS=(5,10)
CANDIDATES=[dict(max_leaf_nodes=n,l2_regularization=l2,max_iter=300,
                 learning_rate=.06,min_samples_leaf=40)
            for n,l2 in ((15,5.),(31,10.),(63,20.))]


def ordered_blocks(path):
    """Distinct events; chronological sentence halves; never split a sentence across sides."""
    sessions={}
    with path.open(newline='',encoding='utf8',errors='replace') as f:
        for row in csv.DictReader(f,delimiter='\t'):
            try:
                event=tuple(int(row[k]) for k in ('PRESS_TIME','RELEASE_TIME','KEYCODE','KEYSTROKE_ID'))
                sid=row['TEST_SECTION_ID']
            except (ValueError,KeyError,TypeError): continue
            sessions.setdefault(sid,set()).add(event)
    sessions=[sorted(events,key=lambda e:(e[0],e[3])) for events in sessions.values() if len(events)>=6]
    sessions.sort(key=lambda events:(events[0][0],events[0][3]))
    midpoint=len(sessions)//2
    if midpoint==0: raise ValueError('Insufficient distinct sentences')
    left,right=sessions[:midpoint],sessions[midpoint:]
    if max(e[1] for s in left for e in s)>=min(e[0] for s in right for e in s):
        raise ValueError('Baseline and query periods overlap')
    def windows(group):
        result=[]
        for sentence in group:
            for start in range(0,len(sentence),50):
                events=sentence[start:start+50]
                if len(events)<6: continue
                x,n=events_to_features(events,50)
                if not np.isfinite(x).all(): raise ValueError('Nonfinite source timings')
                # Hash observed timings/content without identity-specific event IDs or timestamps.
                key=hashlib.sha256(x[:n].tobytes()).hexdigest()
                result.append((x,n,key))
        return result
    return windows(left),windows(right)


def rng_for(role,uid,cls):
    value=hashlib.sha256(f'accumulated:{SEED}:{role}:{uid}:{cls}'.encode()).digest()
    return np.random.default_rng(int.from_bytes(value[:8],'little'))


def choose(out,smoke):
    """Select membership using source availability only, before any synthetic outcomes."""
    files=[Path(__file__),HERE/'audit.py',HERE/'predict.py',ROOT/'prototype_net/exp3/features.py',
           ROOT/'prototype_net/exp3/run.py',ROOT/'paper_typenet/nn.py',
           ROOT/'prototype_net/perturbation_generator/perturb.py']
    if (out/'protocol.json').exists():
        p=json.loads((out/'protocol.json').read_text())
        for path,sha in p['source_hashes'].items():
            if digest(ROOT/path)!=sha: raise RuntimeError(f'Changed registered source: {path}')
        return p
    roles=json.loads(MANIFEST.read_text())['roles']
    old=json.loads((ROOT/'prototype_net/exp3/runs/timing_v1/protocol.json').read_text())['splits']
    previously_opened=set().union(*(set(old[r]) for r in ('selection','calibration','validation','test')))
    def order(ids):
        return sorted((str(u) for u in ids),key=lambda u:hashlib.sha256(f'{SEED}:{u}'.encode()).hexdigest())
    train=order(roles['train']['participant_ids'])
    selection=order(set(roles['selection']['participant_ids'])-previously_opened)
    held=order(set(roles['final_test']['participant_ids'])-previously_opened)
    pools={'train':train,'selection':selection,'calibration':held[:2500],
           'validation':held[2500:5500],'test':held[5500:]}
    counts=dict(train=12000,selection=800,calibration=600,validation=800,test=2000)
    if smoke:
        pools={r:train[i*60:(i+1)*60] for i,r in enumerate(counts)}
        counts={r:8 for r in counts}
    splits={}; audit={}; used_hashes={}
    for role,pool in pools.items():
        ids=[]; skipped=[]; hashes={}; lengths=[]
        for uid in pool:
            try:
                g,q=ordered_blocks(DATA/f'{uid}_keystrokes.txt')
                if min(len(g),len(q))<max(BUDGETS): raise ValueError('Fewer than ten windows on either side of sentence boundary')
                # Same matched participants at both budgets; no repeated window on either side.
                rows=g[-10:]+q[:10]
                keys=[v[2] for v in rows]
                if len(set(keys))!=len(keys): raise ValueError('Repeated source window within participant')
                if any(k in used_hashes for k in keys): raise ValueError('Repeated source window across participants')
            except (OSError,ValueError) as exc:
                skipped.append(dict(id=uid,reason=str(exc))); continue
            for k in keys: used_hashes[k]=uid
            ids.append(uid); hashes[uid]=digest(DATA/f'{uid}_keystrokes.txt')
            lengths.extend(int(v[1]) for v in rows)
            if len(ids)==counts[role]: break
        if len(ids)<counts[role]: raise RuntimeError(f'Only {len(ids)} eligible {role} participants; requested {counts[role]}')
        splits[role]=ids
        audit[role]=dict(retained=len(ids),skipped=skipped,source_sha256=hashes,
                         keys_per_window_quantiles=np.quantile(lengths,[0,.25,.5,.75,1]).tolist())
        print('REGISTER',role,len(ids),'skipped',len(skipped),flush=True)
    assert sum(map(len,splits.values()))==len(set().union(*map(set,splits.values())))
    if not smoke:
        assert not (set().union(*(set(splits[r]) for r in ('selection','calibration','validation','test')))&previously_opened)
    p=dict(seed=SEED,smoke=smoke,splits=splits,budgets=list(BUDGETS),candidates=CANDIDATES,
           generator=asdict(PerturbationConfig()),manifest_sha256=digest(MANIFEST),
           source_hashes={str(f.relative_to(ROOT)):digest(f) for f in files},
           primary_comparison='Paired 10-minus-5 window accuracy on the identical participant cohort.',
           baseline='Last N windows of first chronological half of sentences.',
           query='First N windows of second chronological half; identical nested realizations and profile across budgets.',
           selection='4-way accuracy then lower log loss; three tree candidates per budget; no row-random early stopping.',
           calibration='Separate scalar temperature for each budget; independent calibration participants.',
           uncertainty='1000 participant-cluster bootstrap samples; all four classes of an identity stay together.',
           limitations=['Only 5 and 10 windows per side are supported at cohort scale.',
             'Windows contain 6–50 keys, with no overlap; several windows may belong to one sentence.',
             'Chronological within one typing task, not longitudinal over days/months.',
             'Current exp3 evaluation IDs excluded. Historical v4 aggregate exposure remains.',
             'Stable synthetic severity/profile across query period; no clinical or changing-severity claim.'])
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'availability_selection.json',audit)
    write_json(out/'protocol.json',p)
    return p


def make_row(task):
    role,uid,sha=task
    path=DATA/f'{uid}_keystrokes.txt'
    if digest(path)!=sha: raise RuntimeError(f'Source changed: {uid}')
    g,q=ordered_blocks(path); g=g[-10:]; q=q[:10]
    refs={n:enrollment(np.stack([x for x,_,_ in g[-n:]]),[k for _,k,_ in g[-n:]]) for n in BUDGETS}
    perturber=StructuredTimingPerturber(PerturbationConfig())
    result={str(n):[] for n in BUDGETS}
    for cls in range(4):
        rng=rng_for(role,uid,cls)
        profile=perturber.sample_profile(cls-1,rng) if cls else None
        queries=[(perturber.perturb(x,k,cls-1,rng,profile=profile) if cls else x,k) for x,k,_ in q]
        for n in BUDGETS:
            result[str(n)].append(bundle_vector(np.stack([session_vector(x,k,refs[n]) for x,k in queries[:n]])))
    return {k:np.stack(v) for k,v in result.items()}


def dataset(out,p,role):
    path=out/f'{role}.npz'
    if path.exists():
        metadata=json.loads((out/f'{role}_cache.json').read_text())
        if digest(path)!=metadata['sha256']: raise RuntimeError('Modified feature cache')
        return dict(np.load(path,allow_pickle=False))
    audit=json.loads((out/'availability_selection.json').read_text())[role]
    tasks=[(role,u,audit['source_sha256'][u]) for u in p['splits'][role]]
    rows=[]; start=time.monotonic()
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,row in enumerate(pool.map(make_row,tasks,chunksize=8)):
            rows.append(row)
            if (i+1)%500==0: print('FEATURES',role,i+1,len(tasks),round(time.monotonic()-start),flush=True)
    data={str(n):np.concatenate([r[str(n)] for r in rows]) for n in BUDGETS}
    data.update(y=np.tile(np.arange(4),len(tasks)),users=np.repeat(p['splits'][role],4))
    np.savez_compressed(path,**data)
    write_json(out/f'{role}_cache.json',dict(sha256=digest(path),seconds=time.monotonic()-start))
    return data


def develop(out,p):
    if (out/'freeze.json').exists(): raise RuntimeError('Already frozen; refusing refit')
    train,sel=dataset(out,p,'train'),dataset(out,p,'selection')
    models={}; search=[]
    for n in BUDGETS:
        key=str(n); best=(-1.,-float('inf'))
        for original in p['candidates']:
            config={**original,'max_iter':5,'min_samples_leaf':4} if p['smoke'] else original
            model=HistGradientBoostingClassifier(**config,early_stopping=False,random_state=SEED)
            model.fit(train[key],train['y']); prob=model.predict_proba(sel[key])
            acc=float(np.mean(prob.argmax(1)==sel['y'])); loss=float(log_loss(sel['y'],prob))
            search.append(dict(windows_per_side=n,config=config,selection_accuracy=acc,selection_log_loss=loss))
            write_json(out/'search.json',search); print('CANDIDATE',json.dumps(search[-1]),flush=True)
            if (acc,-loss)>best: best=(acc,-loss); models[key]=dict(model=model,config=config)
    # Model selection finished before calibration is generated.
    cal=dataset(out,p,'calibration'); report={}
    for key,item in models.items():
        prob=item['model'].predict_proba(cal[key])
        fit=minimize_scalar(lambda t:log_loss(cal['y'],calibrated(prob,np.exp(t))),bounds=(-2.3,2.3),method='bounded')
        item['temperature']=float(np.exp(fit.x))
        report[key]={role:metrics(d['y'],calibrated(item['model'].predict_proba(d[key]),item['temperature']),d['users'])
                     for role,d in [('train',train),('selection',sel),('calibration',cal)]}
    joblib.dump(models,out/'models.joblib')
    write_json(out/'development.json',report)
    write_json(out/'freeze.json',dict(model_sha256=digest(out/'models.joblib'),protocol_sha256=digest(out/'protocol.json'),time=time.time()))
    print('FROZEN',flush=True)


def paired_difference(y,p5,p10,users):
    delta=(p10.argmax(1)==y).astype(float)-(p5.argmax(1)==y)
    unique,idx=np.unique(users,return_inverse=True)
    per_user=np.bincount(idx,weights=delta)/np.bincount(idx)
    rng=np.random.default_rng(SEED)
    bootstrap=[rng.choice(per_user,len(per_user),replace=True).mean() for _ in range(1000)]
    return dict(accuracy_difference=float(delta.mean()),participant_bootstrap_95ci=np.quantile(bootstrap,[.025,.975]).tolist())


def seal(out,p):
    freeze=json.loads((out/'freeze.json').read_text())
    assert digest(out/'models.joblib')==freeze['model_sha256']
    assert digest(out/'protocol.json')==freeze['protocol_sha256']
    if (out/'seal_started.json').exists(): raise RuntimeError('Evaluation already started; never silently repeat')
    write_json(out/'seal_started.json',freeze)
    models=joblib.load(out/'models.joblib'); results={}
    for role in ('validation','test'):
        data=dataset(out,p,role); results[role]={}; predictions={}
        for key,item in models.items():
            raw=item['model'].predict_proba(data[key]); prob=calibrated(raw,item['temperature'])
            predictions[key]=prob
            results[role][key]=dict(raw=metrics(data['y'],raw,data['users']),calibrated=metrics(data['y'],prob,data['users']))
            np.savez_compressed(out/f'{role}_{key}_predictions.npz',y=data['y'],p=prob,users=data['users'])
            m=results[role][key]['calibrated']
            print('SEALED',role,key,'accuracy',m['accuracy'],'ECE',m['ece_15_equal_width'],'EER',m['pooled_detection_eer'],flush=True)
        results[role]['paired_10_minus_5']=paired_difference(data['y'],predictions['5'],predictions['10'],data['users'])
        write_json(out/f'{role}_results.json',results[role])
    write_json(out/'results.json',results)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--phase',choices=['register','develop','seal'],default='develop')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(); out=HERE/'runs'/('smoke' if args.smoke else 'v1')
    p=choose(out,args.smoke)
    if args.phase!='register':
        with threadpool_limits(limits=4): (develop if args.phase=='develop' else seal)(out,p)

if __name__=='__main__': main()
