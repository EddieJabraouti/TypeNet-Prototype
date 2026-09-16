"""Final selection/calibration/sealing for the three registered model families.

Weights are ordered [summary tree, statistical-baseline sequence, full-gallery
sequence]. The 15 quarter-step mixtures are fixed before relational training.
No test data are loaded until the explicit single-use seal phase.
"""
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

from .run import HERE,ROOT,digest,write_json,dataset,calibrated,metrics
from .sequence_train import raw_cohort,register as verify_sequence
from .relational_train import fixed_queries,register as verify_relational
from .sequence_model import TimingSequenceNet,probabilities
from .relational import RelationalTimingNet

OUT=HERE/'runs/final_v1'
BASE=HERE/'runs/sequence_v1'
REL=HERE/'runs/relational_v1'
TREE=HERE/'runs/combined_v1/frozen.joblib'


def register():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'plan.json'
    if path.exists():
        plan=json.loads(path.read_text())
        if plan['source_sha256']!=digest(Path(__file__)): raise RuntimeError('Registered finalization code changed')
        return plan
    plan=dict(time=time.time(),source_sha256=digest(Path(__file__)),
        weights=[[a/4,b/4,(4-a-b)/4] for a in range(5) for b in range(5-a)],
        model_order=['summary_tree','sequence_v1','relational_v1'],
        rule='Select accuracy, tie-break log loss, on selection participants only; independently per observation budget',
        calibration='One temperature fit on separate calibration participants',
        final_evaluation='One pass on final validation and test after all choices are frozen',
        reason='Longer training authorized; compare whole gallery sequences as well as baseline statistics; test remains unopened.')
    write_json(path,plan)
    protocol=json.loads((BASE/'protocol.json').read_text())
    write_json(OUT/'protocol.json',protocol)
    return plan


def components(role,protocol,device):
    observed=fixed_queries(OUT,raw_cohort(OUT,protocol,role),role)
    summaries=dataset(OUT,protocol,role)
    np.testing.assert_array_equal(observed['users'],summaries['users'])
    np.testing.assert_array_equal(observed['y'],summaries['y'])
    query_only={**observed,**{key:observed[key][:,10:] for key in ('values','keys','mask')}}
    trees=joblib.load(TREE); result={}
    for budget in ('single','bundle'):
        x=summaries[budget]
        if budget=='single': x=x.reshape(-1,x.shape[-1])
        probs=[trees[budget]['model'].predict_proba(x)]
        epochs=[]
        for directory,cls,data in ((BASE,TimingSequenceNet,query_only),(REL,RelationalTimingNet,observed)):
            ck=torch.load(directory/f'best_{budget}.pt',map_location='cpu',weights_only=False)
            assert ck['protocol_sha256']==digest(directory/'protocol.json')
            model=cls(ck['width']).to(device); model.load_state_dict(ck['model'])
            probs.append(probabilities(model,data,device)[budget].reshape(-1,4)); epochs.append(ck['epoch'])
            del model
        y=np.repeat(observed['y'],5) if budget=='single' else observed['y']
        users=np.repeat(observed['users'],5) if budget=='single' else observed['users']
        result[budget]=dict(p=np.stack(probs),y=y,users=users,epochs=epochs)
    return result


def blend(p,weights):
    return np.tensordot(np.asarray(weights),p,axes=(0,0))


def select(protocol,plan,device):
    if (OUT/'freeze.json').exists(): raise RuntimeError('Already frozen')
    for directory,verify in ((BASE,verify_sequence),(REL,verify_relational)):
        if not (directory/'training_complete.json').exists(): raise RuntimeError(f'Training unfinished: {directory}')
        verify(directory,False)
    predictions=components('selection',protocol,device)
    choices={}; search=[]
    for budget,data in predictions.items():
        best=(-1.,-float('inf'))
        for weights in plan['weights']:
            p=blend(data['p'],weights)
            acc=float(np.mean(p.argmax(1)==data['y'])); nll=float(log_loss(data['y'],p))
            search.append(dict(budget=budget,weights=weights,accuracy=acc,log_loss=nll))
            if (acc,-nll)>best:
                best=(acc,-nll)
                choices[budget]=dict(weights=weights,selection_accuracy=acc,selection_log_loss=nll,sequence_epochs=data['epochs'])
    write_json(OUT/'search.json',search); write_json(OUT/'choices_before_calibration.json',choices)
    calibration=components('calibration',protocol,device)
    for budget,choice in choices.items():
        data=calibration[budget]; p=blend(data['p'],choice['weights'])
        fit=minimize_scalar(lambda t:log_loss(data['y'],calibrated(p,np.exp(t))),bounds=(-2.3,2.3),method='bounded')
        choice['temperature']=float(np.exp(fit.x))
        choice['calibration_raw']=metrics(data['y'],p,data['users'])
        choice['calibration_fitted']=metrics(data['y'],calibrated(p,choice['temperature']),data['users'])
        print('FROZEN '+json.dumps({k:v for k,v in choice.items() if not k.startswith('calibration_')}|{'budget':budget}),flush=True)
    paths=[TREE,OUT/'protocol.json',OUT/'plan.json',Path(__file__)]
    for directory in (BASE,REL): paths.extend([directory/'best_single.pt',directory/'best_bundle.pt',directory/'protocol.json'])
    write_json(OUT/'freeze.json',dict(choices=choices,time=time.time(),hashes={str(p.relative_to(ROOT)):digest(p) for p in paths}))


def seal(protocol,device):
    if (OUT/'seal_started.json').exists(): raise RuntimeError('Sealed evaluation already started; no automatic reruns')
    freeze=json.loads((OUT/'freeze.json').read_text())
    for file,sha in freeze['hashes'].items():
        if digest(ROOT/file)!=sha: raise RuntimeError(f'Frozen component changed: {file}')
    write_json(OUT/'seal_started.json',dict(time=time.time(),freeze_sha256=digest(OUT/'freeze.json')))
    results={}
    for role in ('validation','test'):
        predictions=components(role,protocol,device); results[role]={}
        for budget,data in predictions.items():
            choice=freeze['choices'][budget]; p=blend(data['p'],choice['weights'])
            cp=calibrated(p,choice['temperature'])
            results[role][budget]=dict(raw=metrics(data['y'],p,data['users']),calibrated=metrics(data['y'],cp,data['users']))
            np.savez_compressed(OUT/f'{role}_{budget}_predictions.npz',p=cp,y=data['y'],users=data['users'])
            m=results[role][budget]['calibrated']
            print(f'SEALED {role} {budget}: acc={m["accuracy"]:.4f} ECE={m["ece_15_equal_width"]:.4f} EER={m["pooled_detection_eer"]:.4f}',flush=True)
        write_json(OUT/f'{role}_results.json',results[role])
    write_json(OUT/'results.json',results)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--phase',choices=['register','select','seal'],required=True)
    parser.add_argument('--device',choices=['cpu','mps'],default='cpu')
    args=parser.parse_args(); plan=register()
    if args.phase=='register': print('Final search registered; no calibration or evaluation data opened.'); return
    protocol=json.loads((OUT/'protocol.json').read_text())
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        if args.phase=='select': select(protocol,plan,torch.device(args.device))
        else: seal(protocol,torch.device(args.device))


if __name__=='__main__': main()
