"""Small MPS-compatible nonlinear classifier for observed timing summaries."""
import numpy as np
import torch
from torch import nn


class Block(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.layers=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),nn.SiLU(),nn.Dropout(.1))

    def forward(self,x):
        return x+self.layers(x)


class Net(nn.Module):
    def __init__(self,dim,width,depth):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(dim,width),nn.SiLU(),
                                 *[Block(width) for _ in range(depth)],nn.Linear(width,4))

    def forward(self,x):
        return self.layers(x)


class TorchClassifier:
    def __init__(self,net,mean,scale):
        self.net=net.cpu().eval()
        self.mean,self.scale=mean,scale

    @torch.inference_mode()
    def predict_proba(self,x):
        self.net.eval()
        result=[]
        for start in range(0,len(x),2048):
            z=np.clip((x[start:start+2048]-self.mean)/self.scale,-10,10).astype('float32')
            result.append(self.net(torch.from_numpy(z)).softmax(1).numpy())
        return np.concatenate(result)
