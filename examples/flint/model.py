import torch
from torch import nn
import torch.nn.functional as F
from recurrence import recurrence

class Model(nn.Module):
    def __init__(self):
        super().__init__();self.emb=nn.Embedding(256,64);self.proj=nn.Linear(64,4*64+8)
        self.out=nn.Linear(64,64);self.norm=nn.LayerNorm(64);self.mlp=nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Linear(128,64))
        self.fast=nn.Parameter(torch.randn(1,4,16,16)*.01);self.slow=nn.Parameter(torch.randn(1,4,16,16)*.01)
    def forward(self,ids):
        x=self.emb(ids);batch,length,_=x.shape
        q,k,v,decay,gates=self.proj(self.norm(x)).split([64]*4+[8],dim=-1)
        q,k,v,decay=(z.reshape(batch,length,4,16).transpose(1,2) for z in (q,k,v,decay))
        beta,transfer=gates.reshape(batch,length,2,4).permute(2,0,3,1)
        state={'fast':self.fast.expand(batch,-1,-1,-1).contiguous(),'slow':self.slow.expand(batch,-1,-1,-1).contiguous()}
        outputs=[]
        for start in range(0,length,64):
            args=[z[:,:,start:start+64].contiguous() for z in (q,k,v,decay,beta,transfer)]
            y,state=recurrence(*args,state,.25);outputs.append(y)
        y=torch.cat(outputs,dim=2).transpose(1,2).reshape(batch,length,64)
        x=x+self.out(y.to(x.dtype));x=x+self.mlp(self.norm(x))
        return F.linear(self.norm(x),self.emb.weight)
