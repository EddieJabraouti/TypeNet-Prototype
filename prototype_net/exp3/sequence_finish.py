"""Select a tree/sequence combination, calibrate, then evaluate once.

Register the five fixed mixture weights before looking at sequence results:
  python -m prototype_net.exp3.sequence_finish --phase register
After training:
  python -m prototype_net.exp3.sequence_finish --phase select --device mps
Only after all modeling decisions are finished:
  python -m prototype_net.exp3.sequence_finish --phase seal --device mps

Mixtures interpolate probabilities, not embeddings. A weight of zero retains
the summary-feature tree; a weight of one uses only the temporal model.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from .run import HERE, ROOT, digest, write_json, dataset, calibrated, metrics
from .sequence_train import raw_cohort, fixed_queries, register as verify_training
from .sequence_model import TimingSequenceNet, probabilities

OUT=HERE/'runs/sequence_v1'
TREES=HERE/'runs/combined_v1'


def register():
    dest=OUT/'finish_plan.json'
    if dest.exists(): return json.loads(dest.read_text())
    plan=dict(weights=[0.,.25,.5,.75,1.],created=time.time(),
        selection='4-way selection accuracy; ties use lower log loss; separate observation budgets',
        calibration='Scalar temperature, independent calibration participants, preserves argmax',
        evaluation='One final validation and test pass after selected models and calibration are frozen',
        test_scope='4000 participants chosen before modeling; retained count logged; old v4 test cohort',
        finish_source_sha256=digest(Path(__file__)))
    write_json(dest,plan)
    return plan


def components(role,protocol,device):
    raw=raw_cohort(OUT,protocol,role)
    observed=fixed_queries(OUT,raw,role)
    # Build timing summaries from precisely the same participants and draws.
    summaries=dataset(OUT,protocol,role)
    np.testing.assert_array_equal(observed['users'],summaries['users'])
    np.testing.assert_array_equal(observed['y'],summaries['y'])
    trees=joblib.load(TREES/'frozen.joblib')
    result={}
    for budget in ('single','bundle'):
        ck=torch.load(OUT/f'best_{budget}.pt',map_location='cpu',weights_only=False)
        assert ck['protocol_sha256']==digest(OUT/'protocol.json')
        model=TimingSequenceNet(ck['width']).to(device); model.load_state_dict(ck['model'])
        seq=probabilities(model,observed,device)[budget].reshape(-1,4)
        x=summaries[budget]
        if budget=='single': x=x.reshape(-1,x.shape[-1])
        tree=trees[budget]['model'].predict_proba(x)
        y=np.repeat(observed['y'],5) if budget=='single' else observed['y']
        users=np.repeat(observed['users'],5) if budget=='single' else observed['users']
        result[budget]=dict(sequence=seq,tree=tree,y=y,users=users,epoch=ck['epoch'])
        del model
    return result


def select(protocol,plan,device):
    if not (OUT/'training_complete.json').exists(): raise RuntimeError('Training must finish first')
    if (OUT/'final_freeze.json').exists(): raise RuntimeError('Final model already frozen')
    choices={}; search=[]
    selection=components('selection',protocol,device)
    for budget,data in selection.items():
        best=(-1.,-float('inf'))
        for weight in plan['weights']:
            p=weight*data['sequence']+(1-weight)*data['tree']
            acc=float(np.mean(p.argmax(1)==data['y'])); nll=float(log_loss(data['y'],p))
            search.append(dict(budget=budget,sequence_weight=weight,accuracy=acc,log_loss=nll))
            if (acc,-nll)>best:
                best=(acc,-nll)
                choices[budget]=dict(sequence_weight=weight,epoch=data['epoch'],selection_accuracy=acc,selection_log_loss=nll)
    # Choices are committed before calibration data are opened.
    write_json(OUT/'mixture_search.json',search); write_json(OUT/'choices_before_calibration.json',choices)
    calibration=components('calibration',protocol,device)
    for budget,choice in choices.items():
        data=calibration[budget]; w=choice['sequence_weight']
        p=w*data['sequence']+(1-w)*data['tree']
        fit=minimize_scalar(lambda t:log_loss(data['y'],calibrated(p,np.exp(t))),bounds=(-2.3,2.3),method='bounded')
        choice['temperature']=float(np.exp(fit.x))
        choice['calibration_raw']=metrics(data['y'],p,data['users'])
        choice['calibration_fitted']=metrics(data['y'],calibrated(p,choice['temperature']),data['users'])
        print(f'FINAL CHOICE {budget}: sequence_weight={w}, selection_acc={choice["selection_accuracy"]:.4f}, T={choice["temperature"]:.3f}',flush=True)
    paths=[OUT/'best_single.pt',OUT/'best_bundle.pt',TREES/'frozen.joblib',OUT/'protocol.json',OUT/'finish_plan.json',Path(__file__)]
    write_json(OUT/'final_freeze.json',dict(choices=choices,time=time.time(),
                hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}))


def seal(protocol,device):
    if (OUT/'final_evaluation_started.json').exists(): raise RuntimeError('Final evaluation already started; no automatic rerun')
    frozen=json.loads((OUT/'final_freeze.json').read_text())
    for file,sha in frozen['hashes'].items():
        if digest(ROOT/file)!=sha: raise RuntimeError(f'Frozen component changed: {file}')
    write_json(OUT/'final_evaluation_started.json',dict(time=time.time(),freeze_sha256=digest(OUT/'final_freeze.json')))
    result={}
    for role in ('validation','test'):
        predictions=components(role,protocol,device)
        result[role]={}
        for budget,data in predictions.items():
            choice=frozen['choices'][budget]; w=choice['sequence_weight']
            p=w*data['sequence']+(1-w)*data['tree']
            cp=calibrated(p,choice['temperature'])
            result[role][budget]=dict(raw=metrics(data['y'],p,data['users']),calibrated=metrics(data['y'],cp,data['users']))
            np.savez_compressed(OUT/f'final_{role}_{budget}.npz',p=cp,y=data['y'],users=data['users'])
            m=result[role][budget]['calibrated']
            print(f'FINAL {role} {budget}: acc={m["accuracy"]:.4f}, ECE={m["ece_15_equal_width"]:.4f}, EER={m["pooled_detection_eer"]:.4f}',flush=True)
        write_json(OUT/f'final_{role}_results.json',result[role])
    write_json(OUT/'final_results.json',result)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--phase',choices=['register','select','seal'],required=True)
    parser.add_argument('--device',choices=['cpu','mps'],default='cpu')
    args=parser.parse_args()
    plan=register()
    if args.phase=='register': print('Finish plan registered; no evaluation data opened.'); return
    if digest(Path(__file__))!=plan['finish_source_sha256']: raise RuntimeError('Finish code differs from plan')
    protocol=verify_training(OUT,False)
    torch.set_num_threads(4); device=torch.device(args.device)
    with threadpool_limits(limits=4):
        if args.phase=='select': select(protocol,plan,device)
        else: seal(protocol,device)


if __name__=='__main__': main()
