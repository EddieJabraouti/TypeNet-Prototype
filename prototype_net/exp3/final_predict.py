"""Calibrated inference using the final frozen exp3 model combination.

python -m prototype_net.exp3.final_predict observed_sessions.npz --budget single

The NPZ contains gallery [10,50,5], gallery_lengths [10], query [Q,50,5]
and query_lengths [Q]. Single mode accepts 1..5 query sessions and outputs one
classification each. Bundle mode needs exactly five sessions under one condition.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .run import ROOT,HERE,CLASS_NAMES,digest,calibrated
from .features import enrollment,session_vector,bundle_vector
from .sequence_model import baseline_reference,observed_sequence,TimingSequenceNet
from .relational import RelationalTimingNet


def validate(gallery,lengths,query,query_lengths,budget):
    if gallery.shape!=(10,50,5) or lengths.shape!=(10,): raise ValueError('Expected ten enrollment sessions')
    if query.ndim!=3 or query.shape[1:]!=(50,5) or query_lengths.shape!=(len(query),): raise ValueError('Invalid query shape')
    if budget not in ('single','bundle'): raise ValueError('Unknown observation budget')
    if not 1<=len(query)<=5 or (budget=='bundle' and len(query)!=5): raise ValueError('Bundle mode requires five queries')
    if not np.isfinite(gallery).all() or not np.isfinite(query).all(): raise ValueError('Nonfinite timings')
    for n in (lengths,query_lengths):
        if not np.issubdtype(n.dtype,np.integer) or np.any(n<6) or np.any(n>50):
            raise ValueError('Lengths must be integers between 6 and 50')


@torch.inference_mode()
def component_probabilities(gallery,lengths,query,query_lengths,budget,device='cpu'):
    """Same preprocessing and probability order as finalize.components."""
    validate(gallery,lengths,query,query_lengths,budget)
    ref=enrollment(gallery,lengths)
    x=np.stack([session_vector(q,int(n),ref) for q,n in zip(query,query_lengths)])
    if budget=='bundle': x=bundle_vector(x)[None]
    tree=joblib.load(HERE/'runs/combined_v1/frozen.joblib')[budget]['model']
    outputs=[tree.predict_proba(x)]
    reference,context=baseline_reference(gallery,lengths)
    qr=[observed_sequence(q,int(n),reference,context) for q,n in zip(query,query_lengths)]
    gr=[observed_sequence(g,int(n),reference,context) for g,n in zip(gallery,lengths)]
    for directory,cls,rows in ((HERE/'runs/sequence_v1',TimingSequenceNet,qr),
                               (HERE/'runs/relational_v1',RelationalTimingNet,gr+qr)):
        ck=torch.load(directory/f'best_{budget}.pt',map_location='cpu',weights_only=False)
        model=cls(ck['width']).to(device).eval(); model.load_state_dict(ck['model'])
        inputs=[torch.from_numpy(np.stack([r[i] for r in rows])[None]).to(device) for i in range(3)]
        single,bundle=model(*inputs,torch.from_numpy(context[None]).to(device))
        outputs.append((single if budget=='single' else bundle).softmax(-1).cpu().numpy().reshape(-1,4))
    return np.stack(outputs)


def predict(gallery,lengths,query,query_lengths,budget='single',device='cpu'):
    frozen=json.loads((HERE/'runs/final_v1/freeze.json').read_text())
    for file,sha in frozen['hashes'].items():
        if digest(ROOT/file)!=sha: raise RuntimeError(f'Frozen artifact changed: {file}')
    # A checkpoint hash alone does not protect its preprocessing implementation.
    for name in ('sequence_v1','relational_v1'):
        protocol=json.loads((HERE/f'runs/{name}/protocol.json').read_text())
        for file,sha in protocol['source_hashes'].items():
            if digest(ROOT/file)!=sha: raise RuntimeError(f'Frozen source changed: {file}')
    choice=frozen['choices'][budget]
    with threadpool_limits(limits=4):
        p=component_probabilities(gallery,lengths,query,query_lengths,budget,device)
        mixed=np.tensordot(np.asarray(choice['weights']),p,axes=(0,0))
        probs=calibrated(mixed,choice['temperature'])
    return [dict(predicted_class=CLASS_NAMES[row.argmax()],probabilities=dict(zip(CLASS_NAMES,map(float,row)))) for row in probs]


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--budget',choices=['single','bundle'],default='single')
    parser.add_argument('--device',choices=['cpu','mps'],default='cpu')
    args=parser.parse_args(); torch.set_num_threads(4)
    with np.load(args.input,allow_pickle=False) as data:
        result=predict(data['gallery'],data['gallery_lengths'],data['query'],data['query_lengths'],args.budget,args.device)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
