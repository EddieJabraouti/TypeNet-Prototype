"""Matched-data regularization study. Preserves v1; never regenerates samples.

--phase develop: selection-only search then independent temperature calibration.
--phase compare: one comparison on the already-inspected v1 validation/test sets.
"""
import argparse
import copy
import json
from pathlib import Path
import time
import joblib
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits
from prototype_net.exp3.run import ROOT,digest,write_json,metrics,calibrated
from .experiment import HERE,SEED,BUDGETS,paired_difference

CONFIGS=[
 dict(max_leaf_nodes=7,max_depth=3,min_samples_leaf=200,l2_regularization=20.,max_features=1.),
 dict(max_leaf_nodes=15,max_depth=4,min_samples_leaf=200,l2_regularization=50.,max_features=1.),
 dict(max_leaf_nodes=15,max_depth=4,min_samples_leaf=500,l2_regularization=100.,max_features=1.),
 dict(max_leaf_nodes=31,max_depth=5,min_samples_leaf=500,l2_regularization=100.,max_features=.7),
]
STAGES=[75,150,300,600]

def qualifies(accuracy,baseline_accuracy):
    return accuracy>=baseline_accuracy-.005-1e-12


def register(out,parent):
    if (out/'protocol.json').exists():
        p=json.loads((out/'protocol.json').read_text())
    else:
        p=json.loads((parent/'protocol.json').read_text())
        frozen=json.loads((parent/'freeze.json').read_text())
        assert digest(parent/'models.joblib')==frozen['model_sha256']
        assert digest(parent/'protocol.json')==frozen['protocol_sha256']
        p.update(parent=str(parent.relative_to(ROOT)),study='regularization_v1',configs=CONFIGS,stages=STAGES,
            learning_rate=.04,
            selection_rule='Lowest raw selection log loss among candidates within 0.5 percentage points of the original model selection accuracy; baseline is eligible. Ties: higher selection accuracy. No validation/test selection.',
            fixed_data='Exactly the existing v1 NPZ arrays: same people, labels, features, synthetic realizations, budgets and all five roles. No augmentation or data regeneration.',
            test_status='User explicitly requested matched evaluation on the already-inspected v1 validation/test cohorts. Comparative benchmark, not a new untouched test.',
            parent_artifacts={str(f.relative_to(ROOT)):digest(f) for f in [parent/'models.joblib',parent/'protocol.json',parent/'freeze.json']},
            data_hashes={r:json.loads((parent/f'{r}_cache.json').read_text())['sha256'] for r in p['splits']})
        p['source_hashes'][str(Path(__file__).resolve().relative_to(ROOT))]=digest(__file__)
        out.mkdir(parents=True,exist_ok=True);write_json(out/'protocol.json',p)
    for name,sha in {**p['source_hashes'],**p['parent_artifacts']}.items():
        if digest(ROOT/name)!=sha:raise RuntimeError(f'Frozen source/artifact changed: {name}')
    for role,sha in p['data_hashes'].items():
        if digest(parent/f'{role}.npz')!=sha:raise RuntimeError(f'Modified data cache: {role}')
    return p


def data(parent,p,role):
    d=dict(np.load(parent/f'{role}.npz',allow_pickle=False))
    assert np.array_equal(d['users'],np.repeat(p['splits'][role],4))
    assert np.array_equal(d['y'],np.tile(np.arange(4),len(p['splits'][role])))
    return d


def score(model,d,key):
    prob=model.predict_proba(d[key]);return dict(accuracy=float(np.mean(prob.argmax(1)==d['y'])),log_loss=float(log_loss(d['y'],prob)))


def develop(out,parent,p):
    if (out/'freeze.json').exists():raise RuntimeError('Already frozen; refusing to refit')
    if (out/'search.json').exists():raise RuntimeError('Search already started; inspect interruption before restarting')
    train,selection=data(parent,p,'train'),data(parent,p,'selection')
    original=joblib.load(parent/'models.joblib');winners={};history=[]
    for key in map(str,BUDGETS):
        baseline=original[key];bs=score(baseline['model'],selection,key)
        best=(bs['log_loss'],-bs['accuracy']);winners[key]=copy.deepcopy(baseline)
        winners[key]['regularization_choice']='original baseline'
        history.append(dict(budget=key,kind='baseline',selection=bs,train=score(baseline['model'],train,key)))
        write_json(out/'search.json',history)
        for ci,config in enumerate(p['configs']):
            model=HistGradientBoostingClassifier(**config,learning_rate=p['learning_rate'],
                early_stopping=False,warm_start=True,random_state=SEED)
            for stage in p['stages']:
                start=time.monotonic();model.set_params(max_iter=stage);model.fit(train[key],train['y'])
                ss=score(model,selection,key);ts=score(model,train,key)
                record=dict(budget=key,kind='regularized',config_index=ci,config=config,iterations=stage,
                            selection=ss,train=ts,gap=ts['accuracy']-ss['accuracy'],
                            eligible=qualifies(ss['accuracy'],bs['accuracy']),seconds=time.monotonic()-start)
                history.append(record);write_json(out/'search.json',history);print(json.dumps(record),flush=True)
                rank=(ss['log_loss'],-ss['accuracy'])
                if record['eligible'] and rank<best:
                    best=rank;winners[key]=dict(model=copy.deepcopy(model),config={**config,'max_iter':stage,'learning_rate':p['learning_rate']},regularization_choice=f'config {ci}, {stage} iterations')
                    joblib.dump(winners[key],out/f'best_{key}.joblib')
    cal=data(parent,p,'calibration');report={}
    for key,item in winners.items():
        raw=item['model'].predict_proba(cal[key])
        fit=minimize_scalar(lambda t:log_loss(cal['y'],calibrated(raw,np.exp(t))),bounds=(-2.3,2.3),method='bounded')
        item['temperature']=float(np.exp(fit.x));report[key]={}
        for role,d in [('train',train),('selection',selection),('calibration',cal)]:
            prob=item['model'].predict_proba(d[key])
            report[key][role]=dict(raw=metrics(d['y'],prob,d['users']),calibrated=metrics(d['y'],calibrated(prob,item['temperature']),d['users']))
        print('CHOSEN',key,item['regularization_choice'],'T',item['temperature'],flush=True)
    joblib.dump(winners,out/'models.joblib');write_json(out/'development.json',report)
    write_json(out/'freeze.json',dict(model_sha256=digest(out/'models.joblib'),protocol_sha256=digest(out/'protocol.json'),time=time.time()))


def compare(out,parent,p):
    if (out/'comparison_started.json').exists():raise RuntimeError('Comparison already opened; no repeat tuning')
    f=json.loads((out/'freeze.json').read_text())
    assert digest(out/'models.joblib')==f['model_sha256'] and digest(out/'protocol.json')==f['protocol_sha256']
    write_json(out/'comparison_started.json',f)
    original=joblib.load(parent/'models.joblib');new=joblib.load(out/'models.joblib');results={}
    for role in ('validation','test'):
        d=data(parent,p,role);results[role]={}
        for key in map(str,BUDGETS):
            op=calibrated(original[key]['model'].predict_proba(d[key]),original[key]['temperature'])
            # Reproduce previously saved predictions exactly before claiming matched comparison.
            with np.load(parent/f'{role}_{key}_predictions.npz') as saved:
                np.testing.assert_array_equal(saved['y'],d['y']);np.testing.assert_array_equal(saved['users'],d['users'])
                np.testing.assert_allclose(saved['p'],op,rtol=0,atol=1e-12)
            raw=new[key]['model'].predict_proba(d[key]);prob=calibrated(raw,new[key]['temperature'])
            results[role][key]=dict(baseline=metrics(d['y'],op,d['users']),regularized=metrics(d['y'],prob,d['users']),
                                   regularized_raw=metrics(d['y'],raw,d['users']),paired_change=paired_difference(d['y'],op,prob,d['users']))
            np.savez_compressed(out/f'{role}_{key}_predictions.npz',y=d['y'],p=prob,users=d['users'])
            print('COMPARISON',role,key,json.dumps(results[role][key]),flush=True)
        write_json(out/f'{role}_results.json',results[role])
    write_json(out/'results.json',results)
    write_report(out,parent,new,results)


def write_report(out,parent,models,results):
    dev=json.loads((out/'development.json').read_text());old=json.loads((parent/'development.json').read_text())
    lines=['# Matched-data regularization results','',
           'Every candidate used the exact cached v1 training/selection/calibration/validation/test arrays. Features, participant membership, labels, synthetic draws, and observation budgets are unchanged. Existing baseline artifacts are preserved.','',
           'Validation/test were previously inspected and reused at the user’s explicit request. This is a paired comparative benchmark, not new untouched external evidence. No candidate was selected using these outcomes.','',
           'Selection minimizes raw selection log loss while allowing no more than a 0.5-percentage-point selection-accuracy loss versus the original model. The original model remains eligible. Four regularization configurations and four iteration checkpoints were fixed before training; each budget has independent temperature calibration.','']
    for key,item in models.items():
        lines += [f'## {key} windows per side','',f'Selected: {item["regularization_choice"]}.','',
          f'Original training / selection accuracy: {old[key]["train"]["accuracy"]:.4%} / {old[key]["selection"]["accuracy"]:.4%}.',
          f'New training / selection accuracy: {dev[key]["train"]["raw"]["accuracy"]:.4%} / {dev[key]["selection"]["raw"]["accuracy"]:.4%}.','']
        for role in ('validation','test'):
            r=results[role][key];a,b=r['baseline'],r['regularized'];ci=r['paired_change']['participant_bootstrap_95ci']
            lines += [f'- {role.title()} four-way accuracy: {a["accuracy"]:.2%} → {b["accuracy"]:.2%}. Paired change {100*r["paired_change"]["accuracy_difference"]:+.2f} pp (95% participant-bootstrap CI {100*ci[0]:+.2f}, {100*ci[1]:+.2f}).',
                f'  EER: {a["pooled_detection_eer"]:.2%} → {b["pooled_detection_eer"]:.2%}; calibrated ECE: {100*a["ece_15_equal_width"]:.2f} → {100*b["ece_15_equal_width"]:.2f} pp; log loss: {a["log_loss"]:.4f} → {b["log_loss"]:.4f}.']
        lines += ['']
    lines+=['A smaller training–selection gap alone is not success: reduced capacity can also underfit. Generalization metrics and paired uncertainty determine whether the change helped. This experiment makes no claim about clinical cognitive severity.','',
            'Predict with the existing accumulated.predict CLI and --model-dir pointing to this run. No other experiment family, extra observations, new generator, or neuroQWERTY evaluation was introduced.']
    (out.parent.parent/'REGULARIZATION_RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--phase',choices=['develop','compare'],default='develop')
    args=parser.parse_args();parent=HERE/'runs/v1';out=HERE/'runs/regularization_v1';p=register(out,parent)
    with threadpool_limits(limits=4):(develop if args.phase=='develop' else compare)(out,parent,p)

if __name__=='__main__':main()
