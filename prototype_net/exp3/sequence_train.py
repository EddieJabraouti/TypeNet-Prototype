"""Longer MPS experiment with fresh augmentation and all training identities.

python -m prototype_net.exp3.sequence_train --smoke --device cpu
python -m prototype_net.exp3.sequence_train --device mps

Only the train and selection roles are read. Calibration and final evaluation
are deliberately separate operations, after all model choices are complete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import log_loss

from paper_typenet import nn as typenet
from prototype_net.perturbation_generator.perturb import PerturbationConfig, StructuredTimingPerturber
from .run import HERE, ROOT, MANIFEST, DATA, SEED, digest, write_json, seed_for
from .sequence_model import baseline_reference, observed_sequence, TimingSequenceNet, probabilities


def register(out,smoke):
    files=[Path(__file__),HERE/'sequence_model.py',HERE/'features.py',HERE/'run.py',
           ROOT/'paper_typenet/nn.py',ROOT/'prototype_net/perturbation_generator/perturb.py']
    if (out/'protocol.json').exists():
        protocol=json.loads((out/'protocol.json').read_text())
        for file,sha in protocol['source_hashes'].items():
            if digest(ROOT/file)!=sha:
                raise RuntimeError(f'Source differs from registered training: {file}')
        return protocol
    parent=json.loads((HERE/'runs/timing_v1/protocol.json').read_text())
    roles=json.loads(MANIFEST.read_text())['roles']
    train=sorted(roles['train']['participant_ids'],key=lambda u:hashlib.sha256(f'{SEED}:train:{u}'.encode()).hexdigest())
    parent['splits']['train']=train
    if smoke:
        parent['splits']={'train':train[:32],'selection':train[32:40]}
    parent['source_hashes']={str(p.relative_to(ROOT)):digest(p) for p in files}
    parent['sequence_experiment']=dict(epochs=100 if not smoke else 2,steps_per_epoch=300 if not smoke else 2,
        batch_size=64 if not smoke else 4,validate_every=2,width=64,
        optimizer='AdamW',learning_rate=.0003,weight_decay=.01,warmup_epochs=5,
        schedule='cosine decay to 10% of peak learning rate',
        patience_epochs=20,seed=SEED,
        training='Fresh class-balanced perturbations on sampled training participants each step; shared profile over five query sessions.',
        target='Four-way synthetic severity; one session and five-session outputs are separate.',
        preprocessing='1ms rounding, discard terminal transition; healthy enrollment key-conditional mean/variance; no clean query twin.',
        model_selection='Best selection accuracy per observation budget; ties use lower log loss.',
        extension_reason='User authorized longer computation; simple trees/MLPs fell below target on selection only.')
    parent['smoke']=smoke
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'protocol.json',parent)
    return parent


def raw_cohort(out,protocol,role):
    meta=out/f'raw_{role}.json'
    names=('features','lengths','reference','context')
    if meta.exists():
        result={name:np.load(out/f'raw_{role}_{name}.npy',mmap_mode='r') for name in names}
        result['ids']=json.loads(meta.read_text())['ids']
        return result
    requested=protocol['splits'][role]
    count=len(requested)
    shapes=((count,15,50,5),(count,15),(count,256,4),(count,4))
    dtypes=('float32','int64','float32','float32')
    arrays={name:np.lib.format.open_memmap(out/f'raw_{role}_{name}.npy',mode='w+',dtype=dtype,shape=shape)
            for name,shape,dtype in zip(names,shapes,dtypes)}
    ids=[]; skipped=[]; hashes={}; started=time.monotonic()
    for i,uid in enumerate(requested):
        try:
            user=typenet.load_user_sequences(DATA/f'{uid}_keystrokes.txt',50)
            if len(user.features)<15 or np.any(user.lengths[:15]<6):
                skipped.append(dict(id=uid,reason='fewer than 15 usable sessions (>=6 keys)')); continue
        except (OSError,ValueError) as exc:
            skipped.append(dict(id=uid,reason=str(exc))); continue
        j=len(ids)
        arrays['features'][j]=user.features[:15]; arrays['lengths'][j]=user.lengths[:15]
        ref,ctx=baseline_reference(user.features[:10],user.lengths[:10])
        arrays['reference'][j]=ref; arrays['context'][j]=ctx
        sha=hashlib.sha256(user.features[:15].tobytes()+user.lengths[:15].tobytes()).hexdigest()
        if sha in hashes:
            raise RuntimeError(f'Duplicate source sequence: {uid} and {hashes[sha]}')
        hashes[sha]=uid; ids.append(uid)
        if (i+1)%2000==0:
            print(f'RAW {role}: {i+1}/{count}, {time.monotonic()-started:.0f}s',flush=True)
    for arr in arrays.values(): arr.flush()
    for other in out.glob('raw_*.json'):
        overlap=set(hashes)&set(json.loads(other.read_text())['source_hashes'])
        if overlap: raise RuntimeError('Duplicate source sequences across participant roles')
    write_json(meta,dict(ids=ids,skipped=skipped,source_hashes=hashes,seconds=time.monotonic()-started))
    arrays['ids']=ids
    print(f'RAW {role}: retained {len(ids)}/{count}',flush=True)
    return arrays


def group(cohort,index,cls,rng,perturber):
    xs=cohort['features'][index]; ns=cohort['lengths'][index]
    ref=cohort['reference'][index]; ctx=cohort['context'][index]
    profile=perturber.sample_profile(cls-1,rng) if cls else None
    values=[]; keys=[]; masks=[]
    for x,n in zip(xs[10:15],ns[10:15]):
        q=perturber.perturb(x,int(n),cls-1,rng,profile=profile) if cls else x
        v,k,m=observed_sequence(q,int(n),ref,ctx)
        values.append(v); keys.append(k); masks.append(m)
    return np.asarray(values),np.asarray(keys),np.asarray(masks),np.asarray(ctx)


def fixed_queries(out,cohort,role):
    dest=out/f'{role}_queries.npz'
    if dest.exists(): return dict(np.load(dest,allow_pickle=False))
    perturber=StructuredTimingPerturber(PerturbationConfig())
    rows=[]; labels=[]; users=[]
    for i,uid in enumerate(cohort['ids']):
        for cls in range(4):
            rows.append(group(cohort,i,cls,np.random.default_rng(seed_for(role,uid,cls)),perturber))
            labels.append(cls); users.append(uid)
    data={name:np.stack([row[i] for row in rows]) for i,name in enumerate(('values','keys','mask','context'))}
    data.update(y=np.asarray(labels,dtype='int64'),users=np.asarray(users))
    np.savez_compressed(dest,**data)
    return data


def score(predictions,y):
    report={}
    for budget,p in predictions.items():
        targets=np.repeat(y,5) if budget=='single' else y
        p=p.reshape(-1,4)
        report[budget]=dict(accuracy=float(np.mean(p.argmax(1)==targets)),log_loss=float(log_loss(targets,p)))
    return report


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--device',choices=['mps','cpu'],default='mps')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    out=HERE/'runs'/('sequence_smoke' if args.smoke else 'sequence_v1')
    if (out/'training_complete.json').exists():
        print('Training already completed; preserving recorded results.'); return
    protocol=register(out,args.smoke); config=protocol['sequence_experiment']
    torch.set_num_threads(4); torch.manual_seed(SEED)
    device=typenet.resolve_device(args.device)
    if args.device=='mps' and device.type!='mps': raise RuntimeError('MPS was requested but unavailable')
    train=raw_cohort(out,protocol,'train')
    selection=fixed_queries(out,raw_cohort(out,protocol,'selection'),'selection')
    model=TimingSequenceNet(config['width']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    rng=np.random.default_rng(SEED)
    perturber=StructuredTimingPerturber(PerturbationConfig())
    first=1; best={'single':(-1.,-float('inf')),'bundle':(-1.,-float('inf'))}
    history=[]; last_improved=0
    if (out/'latest.pt').exists() and not args.resume:
        raise RuntimeError('Training already exists; use --resume to continue the same run')
    if args.resume:
        ck=torch.load(out/'latest.pt',map_location=device,weights_only=False)
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        rng.bit_generator.state=ck['numpy_rng']; torch.set_rng_state(ck['torch_rng'].cpu())
        if device.type=='mps': torch.mps.set_rng_state(ck['mps_rng'].cpu())
        first=ck['epoch']+1; best=ck['best']; last_improved=ck['last_improved']; history=ck['history']
    print(f'SEQUENCE device={device} params={sum(p.numel() for p in model.parameters()):,} train_users={len(train["ids"]):,} epochs={config["epochs"]} steps={config["steps_per_epoch"]}',flush=True)
    started=time.monotonic()
    for epoch in range(first,config['epochs']+1):
        model.train(); total_loss=0.; single_correct=0; bundle_correct=0
        progress=max(0.,(epoch-config['warmup_epochs'])/max(1,config['epochs']-config['warmup_epochs']))
        factor=min(1.,epoch/config['warmup_epochs'])*(.1+.9*.5*(1+math.cos(math.pi*progress)))
        for pg in optimizer.param_groups: pg['lr']=config['learning_rate']*factor
        epoch_start=time.monotonic()
        for step in range(config['steps_per_epoch']):
            batch=config['batch_size']
            labels=np.arange(batch)%4; rng.shuffle(labels)
            indices=rng.integers(0,len(train['ids']),size=batch)
            rows=[group(train,int(i),int(cls),rng,perturber) for i,cls in zip(indices,labels)]
            tensors=[torch.from_numpy(np.stack([row[i] for row in rows])).to(device) for i in range(4)]
            targets=torch.from_numpy(labels).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            s,b=model(*tensors)
            loss=(torch.nn.functional.cross_entropy(s.reshape(-1,4),targets.repeat_interleave(5))+
                  torch.nn.functional.cross_entropy(b,targets))/2
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step()
            total_loss+=float(loss.detach().cpu())
            single_correct+=int((s.argmax(-1)==targets[:,None]).sum().detach().cpu())
            bundle_correct+=int((b.argmax(-1)==targets).sum().detach().cpu())
        denom=config['steps_per_epoch']*config['batch_size']
        record=dict(epoch=epoch,loss=total_loss/config['steps_per_epoch'],
                    train_single_accuracy=single_correct/(denom*5),train_bundle_accuracy=bundle_correct/denom,
                    learning_rate=optimizer.param_groups[0]['lr'],epoch_seconds=time.monotonic()-epoch_start,
                    elapsed_seconds=time.monotonic()-started)
        if epoch==1 or epoch%config['validate_every']==0 or epoch==config['epochs']:
            report=score(probabilities(model,selection,device),selection['y'])
            record['selection']=report
            for budget,m in report.items():
                value=(m['accuracy'],-m['log_loss'])
                if value>best[budget]:
                    best[budget]=value; last_improved=epoch
                    torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                                    epoch=epoch,width=config['width'],selection=report,
                                    train_single_accuracy=record['train_single_accuracy'],
                                    train_bundle_accuracy=record['train_bundle_accuracy'],
                                    protocol_sha256=digest(out/'protocol.json')),out/f'best_{budget}.pt')
        history.append(record); write_json(out/'history.json',history)
        print('EPOCH '+json.dumps(record),flush=True)
        checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
            numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
            mps_rng=torch.mps.get_rng_state() if device.type=='mps' else None,
            best=best,last_improved=last_improved,history=history)
        torch.save(checkpoint,out/'latest.tmp.pt'); (out/'latest.tmp.pt').replace(out/'latest.pt')
        if epoch-last_improved>=config['patience_epochs']:
            print(f'EARLY STOP: no selection improvement for {config["patience_epochs"]} epochs',flush=True); break
    write_json(out/'training_complete.json',dict(best=best,last_epoch=epoch,seconds=time.monotonic()-started,
        note='No calibration, final validation or test data were opened by this trainer.'))


if __name__=='__main__': main()
