"""Predict one accumulated-period class from equal-size clean/query windows.

python -m prototype_net.exp3.accumulated.predict observed.npz
Arrays: gallery/query (N,50,5), gallery_lengths/query_lengths (N,), N=5 or 10.
Provide distinct chronological periods and non-overlapping source windows.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from prototype_net.exp3.features import enrollment, session_vector, bundle_vector
from prototype_net.exp3.run import ROOT, digest, calibrated
from prototype_net.perturbation_generator.perturb import CLASS_NAMES

HERE=Path(__file__).resolve().parent

def vector(gallery,gallery_lengths,query,query_lengths):
    n=len(gallery)
    if n not in (5,10) or gallery.shape!=(n,50,5) or query.shape!=(n,50,5):
        raise ValueError('Exactly 5 or 10 distinct windows on each side are required')
    for x,lengths in ((gallery,gallery_lengths),(query,query_lengths)):
        if lengths.shape!=(n,) or not np.issubdtype(lengths.dtype,np.integer) or np.any((lengths<6)|(lengths>50)):
            raise ValueError('Integer lengths between 6 and 50 required')
        if not np.isfinite(x).all(): raise ValueError('Finite features required')
    ref=enrollment(gallery,gallery_lengths)
    return bundle_vector(np.stack([session_vector(q,k,ref) for q,k in zip(query,query_lengths)]))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--model-dir',type=Path,default=HERE/'runs/v1')
    args=parser.parse_args(); out=args.model_dir
    freeze=json.loads((out/'freeze.json').read_text()); protocol=json.loads((out/'protocol.json').read_text())
    if digest(out/'models.joblib')!=freeze['model_sha256'] or digest(out/'protocol.json')!=freeze['protocol_sha256']:
        raise RuntimeError('Frozen artifacts changed')
    for file,sha in protocol['source_hashes'].items():
        if digest(ROOT/file)!=sha: raise RuntimeError(f'Frozen source changed: {file}')
    with np.load(args.input,allow_pickle=False) as data:
        n=len(data['gallery'])
        x=vector(*(data[k] for k in ('gallery','gallery_lengths','query','query_lengths')))
    item=joblib.load(out/'models.joblib')[str(n)]
    with threadpool_limits(limits=4):
        prob=calibrated(item['model'].predict_proba(x[None,:]),item['temperature'])[0]
    print(json.dumps(dict(windows_per_side=n,predicted_class=CLASS_NAMES[prob.argmax()],
                         probabilities=dict(zip(CLASS_NAMES,prob.tolist()))),indent=2))

if __name__=='__main__': main()
