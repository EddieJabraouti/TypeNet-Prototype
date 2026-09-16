"""A timing sequence model conditioned on clean enrollment statistics.

The model sees each key's observed timing relative to that person's clean
distribution. Dilated convolutions detect local bursts and longer correlated
changes. Separate outputs use one session or five sessions. Neither output
receives an unperturbed copy of the query or any generator state.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from .features import channels


def baseline_reference(gallery, lengths):
    observed=[channels(x,n) for x,n in zip(gallery,lengths)]
    holds=np.concatenate([h[:-1] for h,p,k in observed])
    presses=np.concatenate([p for h,p,k in observed])
    keys=np.clip(np.concatenate([k[:-1] for h,p,k in observed]),0,255)
    # Eight pseudo-observations prevent unreliable estimates for rare keys.
    count=np.bincount(keys,minlength=256).astype(float)
    ref=np.zeros((256,4),dtype='float32')
    global_stats=[]
    for column,v in enumerate((holds,presses)):
        mean=float(np.mean(v)); var=float(np.var(v))
        sums=np.bincount(keys,weights=v,minlength=256)
        squares=np.bincount(keys,weights=v*v,minlength=256)
        means=(sums+8*mean)/(count+8)
        variances=np.maximum((squares+8*(var+mean*mean))/(count+8)-means*means,.01)
        ref[:,column]=means
        ref[:,column+2]=np.sqrt(variances)
        global_stats.extend((mean,max(np.sqrt(var),.1)))
    return ref,np.asarray(global_stats,dtype='float32')


def observed_sequence(query,length,reference,global_stats):
    h,p,k=channels(query,length)
    # Both timing channels have a real following key; never expose the terminal
    # transition convention that differs between clean and synthetic data.
    h=h[:-1]; k=np.clip(k[:-1],0,255)
    n=len(p)
    hm,hs,pm,ps=global_stats
    kr=reference[k]
    features=np.stack((h+2.,p+1.5,h-hm,p-pm,h-kr[:,0],p-kr[:,1],
                       (h-hm)/hs,(p-pm)/ps,(h-kr[:,0])/kr[:,2],
                       (p-kr[:,1])/kr[:,3]),axis=-1)
    out=np.zeros((49,10),dtype='float32')
    keys=np.zeros(49,dtype='int64'); mask=np.zeros(49,dtype='float32')
    out[:n]=np.clip(features,-10,10)
    keys[:n]=k; mask[:n]=1
    return out,keys,mask


class TemporalBlock(nn.Module):
    def __init__(self,width,dilation):
        super().__init__()
        self.norm=nn.LayerNorm(width)
        self.conv=nn.Conv1d(width,2*width,5,padding=2*dilation,dilation=dilation)
        self.project=nn.Conv1d(width,width,1)
        self.dropout=nn.Dropout(.1)

    def forward(self,x,mask):
        z=self.norm(x.transpose(1,2)).transpose(1,2)*mask
        a,b=self.conv(z).chunk(2,dim=1)
        z=self.project(torch.nn.functional.silu(a)*torch.sigmoid(b))
        return (x+self.dropout(z))*mask


class TimingSequenceNet(nn.Module):
    def __init__(self,width=64):
        super().__init__()
        self.keys=nn.Embedding(256,8)
        self.input=nn.Conv1d(18,width,1)
        self.blocks=nn.ModuleList([TemporalBlock(width,d) for d in (1,2,4,8)])
        self.encode=nn.Sequential(nn.Linear(width*3+4,128),nn.SiLU(),nn.Dropout(.1))
        self.single=nn.Linear(128,4)
        self.bundle=nn.Sequential(nn.Linear(128*3,128),nn.SiLU(),nn.Dropout(.1),nn.Linear(128,4))

    def forward(self,values,keys,mask,context):
        batch,queries,time,_=values.shape
        v=values.reshape(batch*queries,time,10)
        k=keys.reshape(batch*queries,time)
        m=mask.reshape(batch*queries,1,time)
        x=torch.cat((v,self.keys(k)),dim=-1).transpose(1,2)*m
        x=self.input(x)*m
        for block in self.blocks:
            x=block(x,m)
        count=m.sum(-1).clamp_min(1)
        mean=x.sum(-1)/count
        std=(((x-mean.unsqueeze(-1)).square()*m).sum(-1)/count+1e-5).sqrt()
        peak=x.masked_fill(m==0,-1e4).amax(-1)
        ctx=context[:,None,:].expand(batch,queries,4).reshape(-1,4)
        encoded=self.encode(torch.cat((mean,std,peak,ctx),dim=-1)).reshape(batch,queries,128)
        single=self.single(encoded)
        bundle=self.bundle(torch.cat((encoded.mean(1),encoded.std(1,unbiased=False),encoded.amax(1)),dim=1))
        return single,bundle


@torch.inference_mode()
def probabilities(model,data,device,batch_size=128):
    model.eval()
    singles=[]; bundles=[]
    for start in range(0,len(data['y']),batch_size):
        sl=slice(start,start+batch_size)
        args=[torch.from_numpy(np.asarray(data[key][sl])).to(device) for key in ('values','keys','mask','context')]
        s,b=model(*args)
        singles.append(s.softmax(-1).cpu().numpy()); bundles.append(b.softmax(-1).cpu().numpy())
    return dict(single=np.concatenate(singles),bundle=np.concatenate(bundles))
