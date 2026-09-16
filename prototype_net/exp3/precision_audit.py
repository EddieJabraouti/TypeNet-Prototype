"""Selection-only stress audit: coarsen ALL classes and enrollment to 5ms.

This is a distribution-shift diagnostic, not another training configuration or
a replacement for the primary accuracy. No test or calibration data are read.
"""
import json
import time
import numpy as np
import torch
from .run import HERE,seed_for,write_json
from .sequence_model import baseline_reference,observed_sequence,TimingSequenceNet,probabilities
from .sequence_train import score
from prototype_net.perturbation_generator.perturb import StructuredTimingPerturber,PerturbationConfig


def round_grid(x):
    x=x.copy()
    x[...,0]=np.rint(x[...,0]*200)/200
    x[...,2]=np.rint(x[...,2]*200)/200
    return x


def main():
    out=HERE/'runs/sequence_v1'
    if not (out/'training_complete.json').exists(): raise RuntimeError('Wait for fixed sequence checkpoints')
    dest=out/'precision_audit.json'
    if dest.exists(): raise RuntimeError('Audit already recorded')
    torch.set_num_threads(4)
    xs=np.load(out/'raw_selection_features.npy',mmap_mode='r')
    ns=np.load(out/'raw_selection_lengths.npy',mmap_mode='r')
    ids=json.loads((out/'raw_selection.json').read_text())['ids']
    perturber=StructuredTimingPerturber(PerturbationConfig())
    rows=[]; ys=[]; started=time.monotonic()
    for i,uid in enumerate(ids):
        reference,ctx=baseline_reference(round_grid(xs[i,:10]),ns[i,:10])
        for cls in range(4):
            rng=np.random.default_rng(seed_for('selection',uid,cls))
            profile=perturber.sample_profile(cls-1,rng) if cls else None
            query=[]
            for x,n in zip(xs[i,10:15],ns[i,10:15]):
                q=perturber.perturb(x,int(n),cls-1,rng,profile=profile) if cls else x
                query.append(observed_sequence(round_grid(q),int(n),reference,ctx))
            rows.append((*[np.stack([q[j] for q in query]) for j in range(3)],ctx)); ys.append(cls)
    data={name:np.stack([row[i] for row in rows]) for i,name in enumerate(('values','keys','mask','context'))}
    data['y']=np.asarray(ys)
    result={}
    for budget in ('single','bundle'):
        ck=torch.load(out/f'best_{budget}.pt',map_location='cpu',weights_only=False)
        model=TimingSequenceNet(ck['width']); model.load_state_dict(ck['model'])
        p=probabilities(model,data,torch.device('cpu'))
        result[budget]=dict(epoch=ck['epoch'],primary_selection=ck['selection'][budget],
                            coarsened_selection=score(p,data['y'])[budget])
    write_json(dest,dict(results=result,seconds=time.monotonic()-started,
        note='Selection-only stress test at 5ms versus 1ms. Not a new headline task or an assurance of biological realism.'))
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
