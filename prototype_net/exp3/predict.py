"""Inference: calibrated synthetic-severity probabilities from observed timings.

NPZ input: gallery [10,50,5], gallery_lengths [10], query [Q,50,5],
query_lengths [Q]. Q=1..5 for single-session mode; exactly 5 for bundle mode.
No labels, original query, generator profile or participant ID are inputs.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .features import enrollment, session_vector, bundle_vector
from .run import calibrated, CLASS_NAMES, HERE


def predict(model_path, gallery, gallery_lengths, query, query_lengths, budget='single'):
    if gallery.shape != (10,50,5) or gallery_lengths.shape != (10,):
        raise ValueError('Expected ten clean enrollment sessions with shape [10,50,5]')
    if query.ndim != 3 or query.shape[1:] != (50,5) or len(query_lengths)!=len(query):
        raise ValueError('Expected query [Q,50,5] and Q lengths')
    if not np.isfinite(gallery).all() or not np.isfinite(query).all():
        raise ValueError('Nonfinite timing values')
    if np.any(gallery_lengths<6) or np.any(query_lengths<6) or np.any(gallery_lengths>50) or np.any(query_lengths>50):
        raise ValueError('Sessions must contain 6..50 valid keys')
    if budget=='bundle' and len(query)!=5:
        raise ValueError('Bundle model requires exactly five sessions under one condition')
    if budget=='single' and not 1<=len(query)<=5:
        raise ValueError('Expected one through five independent query sessions')
    ref=enrollment(gallery,gallery_lengths)
    rows=np.stack([session_vector(q,n,ref) for q,n in zip(query,query_lengths)])
    x=bundle_vector(rows)[None,:] if budget=='bundle' else rows
    item=joblib.load(model_path)[budget]
    with threadpool_limits(limits=4):
        p=calibrated(item['model'].predict_proba(x),item['temperature'])
    return [dict(predicted_class=CLASS_NAMES[row.argmax()],
                 probabilities=dict(zip(CLASS_NAMES,map(float,row)))) for row in p]


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--model',type=Path,default=HERE/'runs/timing_v1/frozen.joblib')
    parser.add_argument('--budget',choices=['single','bundle'],default='single')
    args=parser.parse_args()
    with np.load(args.input,allow_pickle=False) as data:
        result=predict(args.model, data['gallery'],data['gallery_lengths'],
                       data['query'],data['query_lengths'],args.budget)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
