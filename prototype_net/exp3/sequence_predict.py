"""Inference for the frozen calibrated tree/sequence combination.

Requires sequence_finish --phase select to have finished. NPZ inputs use
gallery [10,50,5], gallery_lengths [10], query [Q,50,5], query_lengths [Q].
Single mode gives one prediction per query; bundle mode requires Q=5.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .features import enrollment,session_vector,bundle_vector
from .sequence_model import baseline_reference,observed_sequence,TimingSequenceNet
from .run import HERE,ROOT,CLASS_NAMES,calibrated,digest


@torch.inference_mode()
def predict(gallery,gallery_lengths,query,query_lengths,budget='single',device='cpu'):
    out=HERE/'runs/sequence_v1'
    frozen=json.loads((out/'final_freeze.json').read_text())
    for file,sha in frozen['hashes'].items():
        if digest(ROOT/file)!=sha: raise RuntimeError(f'Frozen component changed: {file}')
    if gallery.shape!=(10,50,5) or gallery_lengths.shape!=(10,): raise ValueError('Expected ten enrollment sessions')
    if query.ndim!=3 or query.shape[1:]!=(50,5) or len(query)!=len(query_lengths): raise ValueError('Invalid query shape')
    if not 1<=len(query)<=5 or (budget=='bundle' and len(query)!=5): raise ValueError('Bundle mode needs five query sessions')
    if not np.isfinite(gallery).all() or not np.isfinite(query).all(): raise ValueError('Nonfinite input')
    for lengths in (gallery_lengths,query_lengths):
        if np.any(lengths<6) or np.any(lengths>50): raise ValueError('Valid lengths must be 6..50')
    choice=frozen['choices'][budget]; weight=choice['sequence_weight']
    ref,ctx=baseline_reference(gallery,gallery_lengths)
    rows=[observed_sequence(q,int(n),ref,ctx) for q,n in zip(query,query_lengths)]
    tensors=[torch.from_numpy(np.stack([r[i] for r in rows])[None]).to(device) for i in range(3)]
    ck=torch.load(out/f'best_{budget}.pt',map_location='cpu',weights_only=False)
    model=TimingSequenceNet(ck['width']).to(device).eval(); model.load_state_dict(ck['model'])
    s,b=model(*tensors,torch.from_numpy(ctx[None]).to(device))
    seq=(s if budget=='single' else b).softmax(-1).cpu().numpy().reshape(-1,4)
    summary_ref=enrollment(gallery,gallery_lengths)
    values=np.stack([session_vector(q,n,summary_ref) for q,n in zip(query,query_lengths)])
    if budget=='bundle': values=bundle_vector(values)[None]
    tree=joblib.load(HERE/'runs/combined_v1/frozen.joblib')[budget]['model'].predict_proba(values)
    probs=calibrated(weight*seq+(1-weight)*tree,choice['temperature'])
    return [dict(predicted_class=CLASS_NAMES[p.argmax()],probabilities=dict(zip(CLASS_NAMES,map(float,p)))) for p in probs]


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--budget',choices=['single','bundle'],default='single')
    parser.add_argument('--device',choices=['cpu','mps'],default='cpu')
    args=parser.parse_args(); torch.set_num_threads(4)
    with np.load(args.input,allow_pickle=False) as data,threadpool_limits(limits=4):
        result=predict(data['gallery'],data['gallery_lengths'],data['query'],data['query_lengths'],args.budget,args.device)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
