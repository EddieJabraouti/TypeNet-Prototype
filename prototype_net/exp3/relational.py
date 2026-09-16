"""Compare each query to each individual clean enrollment sequence.

Uses the same temporal encoder and observed-timing normalization as sequence_v1.
The ten gallery representations are compared individually to each query, and
the learned comparison responses are pooled. No gallery embedding centroid is
subtracted from a query.
"""
import numpy as np
import torch
from torch import nn
from .sequence_model import TimingSequenceNet,observed_sequence
from .sequence_train import group as query_group


class RelationalTimingNet(TimingSequenceNet):
    def __init__(self,width=64):
        super().__init__(width)
        self.pair=nn.Sequential(nn.Linear(512,128),nn.SiLU(),nn.Dropout(.1),nn.Linear(128,128),nn.SiLU())
        self.reduce=nn.Sequential(nn.Linear(384,128),nn.SiLU(),nn.Dropout(.1))

    def forward(self,values,keys,mask,context):
        batch,sessions,time,_=values.shape
        if not 11<=sessions<=15: raise ValueError('Expected ten clean gallery and one through five query sessions')
        queries=sessions-10
        v=values.reshape(batch*sessions,time,10)
        k=keys.reshape(batch*sessions,time)
        m=mask.reshape(batch*sessions,1,time)
        x=self.input(torch.cat((v,self.keys(k)),dim=-1).transpose(1,2)*m)*m
        for block in self.blocks: x=block(x,m)
        count=m.sum(-1).clamp_min(1)
        mean=x.sum(-1)/count
        std=(((x-mean.unsqueeze(-1)).square()*m).sum(-1)/count+1e-5).sqrt()
        peak=x.masked_fill(m==0,-1e4).amax(-1)
        ctx=context[:,None,:].expand(batch,sessions,4).reshape(-1,4)
        z=self.encode(torch.cat((mean,std,peak,ctx),dim=-1)).reshape(batch,sessions,128)
        g=z[:,:10,None,:].transpose(1,2).expand(-1,queries,-1,-1)
        q=z[:,10:,:,None].transpose(2,3).expand(-1,-1,10,-1)
        pair=self.pair(torch.cat((q,g,q-g,(q-g).abs()),dim=-1))
        # This is the architecture actually loaded by the queued training run.
        # See runs/relational_v1/architecture_recovery.json for the replay audit.
        compared=self.reduce(torch.cat((pair.mean(2),pair.std(2,unbiased=False),pair.amax(2)),dim=-1))
        s=self.single(compared)
        b=self.bundle(torch.cat((compared.mean(1),compared.std(1,unbiased=False),compared.amax(1)),dim=-1))
        return s,b


def group(cohort,index,cls,rng,perturber):
    values,keys,mask,ctx=query_group(cohort,index,cls,rng,perturber)
    gallery=[observed_sequence(x,int(n),cohort['reference'][index],ctx)
             for x,n in zip(cohort['features'][index,:10],cohort['lengths'][index,:10])]
    return (np.concatenate((np.stack([g[0] for g in gallery]),values)),
            np.concatenate((np.stack([g[1] for g in gallery]),keys)),
            np.concatenate((np.stack([g[2] for g in gallery]),mask)),ctx)
