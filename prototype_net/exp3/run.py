"""CPU-efficient, preregistered raw-timing severity benchmark.

python -m prototype_net.exp3.run --phase develop
python -m prototype_net.exp3.run --phase seal

The seal phase is single-use. It evaluates two already-frozen observation
budgets, never chooses between them using test results. No TypeNet is trained
or used; identity information is limited to the person's clean enrollment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, roc_curve
from threadpoolctl import threadpool_limits

from paper_typenet import nn as typenet
from prototype_net.perturbation_generator.perturb import (
    CLASS_NAMES, PerturbationConfig, StructuredTimingPerturber,
)
from .features import bundle_vector, enrollment, session_vector

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MANIFEST = ROOT / 'prototype_net/weights/protocol/synthetic_impairment_protocol_v4_64x512_unseen_triplet_manifest.json'
DATA = ROOT / 'data/Keystrokes/files'
SEED = 9162026
CANDIDATES = [dict(max_leaf_nodes=n, l2_regularization=l2, max_iter=250,
                   learning_rate=.06, min_samples_leaf=40)
              for n, l2 in ((15, 5.), (31, 10.), (63, 20.))]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def prepare(out, smoke=False):
    if (out/'protocol.json').exists():
        protocol = json.loads((out/'protocol.json').read_text())
        for file, sha in protocol['source_hashes'].items():
            if digest(ROOT/file) != sha:
                raise RuntimeError(f'Source changed since registration: {file}; use a new output directory')
        return protocol
    source = json.loads(MANIFEST.read_text())
    roles = source['roles']
    def shuffled(role):
        return sorted(roles[role]['participant_ids'],
                      key=lambda u: hashlib.sha256(f'{SEED}:{role}:{u}'.encode()).hexdigest())
    train, val, test = shuffled('train'), shuffled('selection'), shuffled('final_test')
    counts = [6000, 1200, 1000, 1200, 4000] if not smoke else [16, 8, 8, 8, 8]
    a,b,c,d,e = counts
    splits = dict(train=train[:a], selection=val[:b], calibration=val[b:b+c],
                  validation=val[b+c:b+c+d], test=test[:e])
    if smoke:
        # Smoke runs never consume held-out participants, even when sealing.
        offset=0
        for role,count in zip(splits,counts):
            splits[role]=train[offset:offset+count]
            offset+=count
    sets = [set(v) for v in splits.values()]
    assert sum(map(len, sets)) == len(set.union(*sets)), 'Participant overlap'
    files = [Path(__file__), HERE/'features.py', ROOT/'prototype_net/perturbation_generator/perturb.py',
             ROOT/'paper_typenet/nn.py']
    protocol = dict(seed=SEED, splits=splits, smoke=smoke,
        candidates=CANDIDATES, generator=asdict(PerturbationConfig()),
        source_hashes={str(f.relative_to(ROOT)):digest(f) for f in files},
        legacy_manifest_sha256=digest(MANIFEST),
        selection_metric='4-way accuracy; tie break lower log loss',
        calibration='one scalar temperature fit on separate participants by log loss',
        observation_budgets=['one query session (<=50 keys)', 'five query sessions, shared profile'],
        controls=['round ALL classes to milliseconds', 'discard final transition in ALL classes',
                  'no clean query twin, trace, ID, class-dependent seed reuse or metadata as features',
                  'clean enrollment sessions 0:10; query sessions 10:15',
                  'participant-disjoint roles; test opened only after model/calibration frozen'],
        caveats=['Historical test cohort has published aggregate results from earlier experiments.',
                 'Exp3 uses fresh synthetic draws and a preselected 4000-person test subset.',
                 'Five-session results are a different observation budget, not single-session accuracy.',
                 'Synthetic severity is not clinical cognitive-health validation.'])
    out.mkdir(parents=True, exist_ok=True)
    write_json(out/'protocol.json', protocol)
    return protocol


def seed_for(role, user, cls):
    raw = hashlib.sha256(f'{SEED}:{role}:{user}:{cls}'.encode()).digest()
    return int.from_bytes(raw[:8], 'little')


def dataset(out, protocol, role):
    dest = out/f'{role}.npz'
    if dest.exists():
        return dict(np.load(dest, allow_pickle=False))
    start = time.monotonic()
    perturber = StructuredTimingPerturber(PerturbationConfig())
    rows, bundles, labels, users, skipped, sources = [], [], [], [], [], {}
    for i, uid in enumerate(protocol['splits'][role]):
        path = DATA/f'{uid}_keystrokes.txt'
        try:
            user = typenet.load_user_sequences(path, 50)
            if len(user.features)<15 or np.any(user.lengths[:15]<6):
                skipped.append(dict(id=uid, reason='fewer than 15 usable sessions (>=6 keys)'))
                continue
        except (OSError, ValueError) as exc:
            skipped.append(dict(id=uid, reason=str(exc)))
            continue
        sources[uid]=hashlib.sha256(user.features[:15].tobytes()+user.lengths[:15].tobytes()).hexdigest()
        ref = enrollment(user.features[:10], user.lengths[:10])
        for cls in range(4):
            rng = np.random.default_rng(seed_for(role, uid, cls))
            profile = perturber.sample_profile(cls-1, rng) if cls else None
            queries = []
            for x, n in zip(user.features[10:15], user.lengths[10:15]):
                q = perturber.perturb(x, int(n), cls-1, rng, profile=profile) if cls else x
                queries.append(session_vector(q, n, ref))
            queries = np.stack(queries)
            rows.append(queries)
            bundles.append(bundle_vector(queries))
            labels.append(cls)
            users.append(uid)
        if (i+1)%200==0:
            print(f'features {role}: {i+1}/{len(protocol["splits"][role])}, {time.monotonic()-start:.0f}s', flush=True)
    result = dict(single=np.asarray(rows), bundle=np.asarray(bundles),
                  y=np.asarray(labels), users=np.asarray(users))
    if not len(labels):
        raise RuntimeError(f'No usable participants in {role}')
    np.savez_compressed(dest, **result)
    write_json(out/f'{role}_data_audit.json', dict(requested=len(protocol['splits'][role]),
        retained=len(set(users)), skipped=skipped, seconds=time.monotonic()-start,
        cache_sha256=digest(dest), source_sequence_hashes=sources))
    seen={}
    for audit in out.glob('*_data_audit.json'):
        for uid,sha in json.loads(audit.read_text())['source_sequence_hashes'].items():
            if sha in seen:
                raise RuntimeError(f'Duplicate source sequences: {uid} and {seen[sha]}')
            seen[sha]=uid
    print(f'{role}: {len(set(users))} participants, {result["single"].shape} single, {result["bundle"].shape} bundle', flush=True)
    return result


def xy(data, budget):
    if budget=='single':
        return data['single'].reshape(-1, data['single'].shape[-1]), np.repeat(data['y'],5), np.repeat(data['users'],5)
    return data['bundle'], data['y'], data['users']


def calibrated(p, temperature):
    return softmax(np.log(np.clip(p, 1e-12, 1))/temperature, axis=1)


def metrics(y, p, users):
    pred, conf = p.argmax(1), p.max(1)
    correct = pred==y
    bins, ece = [], 0.
    for lo, hi in zip(np.linspace(0,1,16)[:-1], np.linspace(0,1,16)[1:]):
        mask = (conf>=lo)&((conf<hi) if hi<1 else (conf<=hi))
        if mask.any():
            acc, confidence = float(correct[mask].mean()), float(conf[mask].mean())
            ece += mask.mean()*abs(acc-confidence)
            bins.append(dict(lower=float(lo), upper=float(hi), count=int(mask.sum()),
                             accuracy=acc, confidence=confidence))
    unique, inv = np.unique(users, return_inverse=True)
    by_user = np.bincount(inv, weights=correct)/np.bincount(inv)
    rng = np.random.default_rng(SEED)
    boots = [np.mean(rng.choice(by_user,len(by_user),replace=True)) for _ in range(1000)]
    fpr, tpr, _ = roc_curve(y!=0, 1-p[:,0])
    ix = np.argmin(np.abs(fpr-(1-tpr)))
    matrix = confusion_matrix(y,pred,labels=np.arange(4))
    return dict(accuracy=float(correct.mean()), accuracy_cluster_bootstrap_95ci=np.quantile(boots,[.025,.975]).tolist(),
        log_loss=float(log_loss(y,p,labels=np.arange(4))),
        multiclass_brier=float(np.mean(np.sum((p-np.eye(4)[y])**2,axis=1))),
        ece_15_equal_width=float(ece), mean_confidence=float(conf.mean()),
        pooled_detection_eer=float((fpr[ix]+1-tpr[ix])/2),
        per_class_recall=dict(zip(CLASS_NAMES, (matrix.diagonal()/matrix.sum(1)).tolist())),
        confusion_counts=matrix.tolist(), reliability=bins, samples=len(y), participants=len(unique))


def develop(out, protocol):
    if (out/'frozen.joblib').exists():
        print('Frozen models already exist; no refitting.'); return
    train, selection = dataset(out,protocol,'train'), dataset(out,protocol,'selection')
    winners, search = {}, []
    for budget in ('single','bundle'):
        x,y,_ = xy(train,budget)
        vx,vy,_ = xy(selection,budget)
        best = (-1., -float('inf'))
        for config in protocol['candidates']:
            if protocol['smoke']:
                config = {**config, 'max_iter':10, 'min_samples_leaf':5}
            model = HistGradientBoostingClassifier(**config, early_stopping=False, random_state=SEED)
            t = time.monotonic()
            model.fit(x,y)
            p = model.predict_proba(vx)
            acc, loss = accuracy_score(vy,p.argmax(1)), log_loss(vy,p)
            record = dict(budget=budget,config=config,selection_accuracy=float(acc),
                          selection_log_loss=float(loss),seconds=time.monotonic()-t)
            search.append(record)
            write_json(out/'search.json',search)
            print('CANDIDATE '+json.dumps(record),flush=True)
            if (acc,-loss)>best:
                best = (acc,-loss)
                winners[budget] = dict(model=model,config=config)
    # All model choices finished before looking at calibration participants.
    calibration = dataset(out,protocol,'calibration')
    report = {}
    for budget, item in winners.items():
        x,y,users = xy(calibration,budget)
        p = item['model'].predict_proba(x)
        result = minimize_scalar(lambda logt: log_loss(y, calibrated(p,np.exp(logt))),
                                 bounds=(-2.3,2.3),method='bounded')
        item['temperature'] = float(np.exp(result.x))
        report[budget] = dict(temperature=item['temperature'],
                              calibration_raw=metrics(y,p,users),
                              calibration_fitted=metrics(y,calibrated(p,item['temperature']),users))
        for role,data in (('train',train),('selection',selection)):
            x,y,users = xy(data,budget)
            report[budget][role] = metrics(y,calibrated(item['model'].predict_proba(x),item['temperature']),users)
        print(f'FROZEN {budget}: train={report[budget]["train"]["accuracy"]:.4f} selection={report[budget]["selection"]["accuracy"]:.4f} T={item["temperature"]:.3f}',flush=True)
    joblib.dump(winners,out/'frozen.joblib')
    write_json(out/'development.json',report)
    write_json(out/'freeze.json',dict(model_sha256=digest(out/'frozen.joblib'),
                                    protocol_sha256=digest(out/'protocol.json'),time=time.time()))


def seal(out, protocol):
    if (out/'sealed_results.json').exists():
        raise RuntimeError('Already sealed: do not re-evaluate or tune against test.')
    freeze = json.loads((out/'freeze.json').read_text())
    assert freeze['model_sha256']==digest(out/'frozen.joblib')
    assert freeze['protocol_sha256']==digest(out/'protocol.json')
    if (out/'seal_started.json').exists():
        raise RuntimeError('Seal was already started. Inspect interruption before any explicit recovery.')
    write_json(out/'seal_started.json',{**freeze,'seal_time':time.time()})
    winners = joblib.load(out/'frozen.joblib')
    results = {}
    for role in ('validation','test'):
        data = dataset(out,protocol,role)
        results[role] = {}
        for budget,item in winners.items():
            x,y,users = xy(data,budget)
            raw = item['model'].predict_proba(x)
            p = calibrated(raw,item['temperature'])
            results[role][budget] = dict(raw=metrics(y,raw,users),calibrated=metrics(y,p,users))
            np.savez_compressed(out/f'{role}_{budget}_predictions.npz',y=y,p=p,users=users)
            m=results[role][budget]['calibrated']
            print(f'SEALED {role} {budget}: acc={m["accuracy"]:.4f} ECE={m["ece_15_equal_width"]:.4f} NLL={m["log_loss"]:.4f} EER={m["pooled_detection_eer"]:.4f}',flush=True)
        write_json(out/f'{role}_results.json',results[role])
    write_json(out/'sealed_results.json',results)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--phase',choices=['develop','seal'],default='develop')
    parser.add_argument('--output',type=Path,default=HERE/'runs/timing_v1')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.smoke and args.output==HERE/'runs/timing_v1':
        args.output=HERE/'runs/smoke'
    protocol=prepare(args.output,args.smoke)
    with threadpool_limits(limits=4):
        (develop if args.phase=='develop' else seal)(args.output,protocol)


if __name__=='__main__':
    main()
