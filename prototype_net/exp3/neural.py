"""Selection-only second model family. Frozen trees remain a candidate.

The experiment addendum is written BEFORE training. Final validation and test
are never loaded here. The tree run is preserved, and combined winners are
written in their own directory for one subsequent sealed evaluation.
"""
import copy
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from .run import HERE, SEED, xy, write_json, digest, calibrated, metrics
from .net import Net, TorchClassifier


def main():
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    device=torch.device('mps')
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; do not silently change compute target')
    source=HERE/'runs/timing_v1'
    out=HERE/'runs/combined_v1'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'addendum.json').exists() and '--resume' not in sys.argv:
        raise RuntimeError('This comparison has already started; do not silently rerun')
    configs=[dict(width=128,depth=2),dict(width=256,depth=3)]
    registration=dict(time=time.time(),candidates=configs,epochs=40,
        selection_every=5,batch_size=1024,learning_rate=.001,weight_decay=.01,
        trigger='Initial tree single-session selection accuracy around 54%, below target.',
        rule='Best selection accuracy, tie break log loss, including frozen tree candidate.',
        source_hashes={str(p):digest(p) for p in [Path(__file__),HERE/'net.py']})
    if '--resume' in sys.argv:
        if (out/'frozen.joblib').exists():
            raise RuntimeError('Already frozen; cannot resume fitting')
        write_json(out/'recovery.json',dict(reason='Tree run had not yet written its final model; deterministic replay of unchanged candidates.',**registration))
        if (out/'search.json').exists():
            (out/'search.json').rename(out/'first_attempt_search.json')
    else:
        write_json(out/'addendum.json',registration)
    train=dict(np.load(source/'train.npz'))
    val=dict(np.load(source/'selection.npz'))
    candidates={}; history=[]
    for budget in ('single','bundle'):
        x,y,_=xy(train,budget); vx,vy,_=xy(val,budget)
        mean=x.mean(0); scale=np.maximum(x.std(0),.01)
        tx=torch.from_numpy(np.clip((x-mean)/scale,-10,10).astype('float32')).to(device)
        ty=torch.from_numpy(y).long().to(device)
        tv=torch.from_numpy(np.clip((vx-mean)/scale,-10,10).astype('float32')).to(device)
        best=(-1.,-float('inf'))
        for cfg in configs:
            torch.manual_seed(SEED)
            rng=np.random.default_rng(SEED)
            net=Net(x.shape[1],**cfg).to(device)
            optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
            started=time.monotonic()
            for epoch in range(1,41):
                net.train()
                order=rng.permutation(len(y))
                for start in range(0,len(y),1024):
                    ix=torch.from_numpy(order[start:start+1024].copy()).to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss=torch.nn.functional.cross_entropy(net(tx[ix]),ty[ix])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(),5)
                    optimizer.step()
                if epoch%5==0:
                    net.eval()
                    with torch.inference_mode():
                        p=torch.cat([net(tv[j:j+2048]).softmax(1) for j in range(0,len(vy),2048)]).cpu().numpy()
                    acc=float(np.mean(p.argmax(1)==vy)); nll=float(log_loss(vy,p))
                    record=dict(budget=budget,**cfg,epoch=epoch,selection_accuracy=acc,
                                selection_log_loss=nll,seconds=time.monotonic()-started)
                    history.append(record); write_json(out/'search.json',history)
                    print('NEURAL '+json.dumps(record),flush=True)
                    if (acc,-nll)>best:
                        best=(acc,-nll)
                        candidates[budget]=dict(model=TorchClassifier(copy.deepcopy(net),mean,scale),
                                                config=record)
            del net,optimizer
        del tx,ty,tv
        torch.mps.empty_cache()
    # Wait only at the shell/orchestrator level if trees have not finished.
    joblib.dump(candidates,out/'neural_candidates.joblib')
    trees=joblib.load(source/'frozen.joblib')
    selection={}
    winners={}
    for budget in ('single','bundle'):
        x,y,_=xy(val,budget)
        compared=[]
        for kind,item in (('tree',trees[budget]),('neural',candidates[budget])):
            p=item['model'].predict_proba(x)
            score=(float(np.mean(p.argmax(1)==y)),-float(log_loss(y,p)))
            compared.append((score,kind,item))
        score,kind,item=max(compared,key=lambda v:v[0])
        winners[budget]=dict(model=item['model'],config=item['config'],family=kind)
        selection[budget]=dict(family=kind,accuracy=score[0],log_loss=-score[1])
    write_json(out/'selected.json',selection)
    # Copy or hard-link fixed data caches and protocol; never revise the old run.
    protocol=json.loads((source/'protocol.json').read_text())
    protocol['model_search_addendum_sha256']=digest(out/'addendum.json')
    for path in (Path(__file__),HERE/'net.py'):
        protocol['source_hashes'][str(path.relative_to(HERE.parents[1]))]=digest(path)
    write_json(out/'protocol.json',protocol)
    for role in ('train','selection','calibration'):
        for suffix in ('.npz','_data_audit.json'):
            target=out/f'{role}{suffix}'
            if not target.exists():
                os.link(source/f'{role}{suffix}',target)
    cal=dict(np.load(out/'calibration.npz'))
    reports={}
    for budget,item in winners.items():
        x,y,users=xy(cal,budget)
        p=item['model'].predict_proba(x)
        fit=minimize_scalar(lambda t:log_loss(y,calibrated(p,np.exp(t))),bounds=(-2.3,2.3),method='bounded')
        item['temperature']=float(np.exp(fit.x))
        reports[budget]=dict(family=item['family'],temperature=item['temperature'],
                            calibration_raw=metrics(y,p,users),
                            calibration_fitted=metrics(y,calibrated(p,item['temperature']),users))
        for role,data in (('train',train),('selection',val)):
            x,y,users=xy(data,budget)
            reports[budget][role]=metrics(y,calibrated(item['model'].predict_proba(x),item['temperature']),users)
        print('WINNER '+json.dumps(dict(budget=budget,**selection[budget],temperature=item['temperature'])),flush=True)
    joblib.dump(winners,out/'frozen.joblib')
    write_json(out/'development.json',reports)
    write_json(out/'freeze.json',dict(model_sha256=digest(out/'frozen.joblib'),
        protocol_sha256=digest(out/'protocol.json'),time=time.time()))


if __name__=='__main__':
    with threadpool_limits(limits=4):
        main()
