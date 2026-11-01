"""Small recurrent regression experiment on a Mac; ordinary autograd and AdamW."""
import argparse
import json
import os
from pathlib import Path
import time
import torch
from recurrence import recurrence

parser=argparse.ArgumentParser()
parser.add_argument('--steps',type=int,default=30)
parser.add_argument('--length',type=int,default=8)
parser.add_argument('--eager',action='store_true')
a=parser.parse_args()
if not torch.backends.mps.is_available():raise SystemExit('This example requires an Apple Silicon Mac with MPS.')
torch.manual_seed(7)
x=torch.randn(4,a.length,32).to('mps')
target=torch.sin(x.cumsum(dim=1)*0.1)
weight=torch.nn.Parameter(torch.ones(32,device='mps')*0.5)
h0=torch.nn.Parameter(torch.zeros(4,32,device='mps'))
c0=torch.nn.Parameter(torch.zeros(4,32,device='mps'))
optimizer=torch.optim.AdamW([weight,h0,c0],lr=.01)
fn=recurrence if a.eager else torch.compile(recurrence,fullgraph=True,dynamic=False)
losses=[];times=[]
for step in range(a.steps):
    torch.mps.synchronize();start=time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    # Float32 recurrence inputs; AMP context is exercised without changing their contract.
    with torch.autocast('mps',dtype=torch.float16):
        y,h,c=fn(x*weight,h0,c0)
        loss=(y-target).square().mean()+.01*(h.square().mean()+c.square().mean())
    loss.backward();optimizer.step();torch.mps.synchronize()
    losses.append(float(loss.detach()));times.append((time.perf_counter()-start)*1000)
print(f'loss {losses[0]:.5f} → {losses[-1]:.5f}')
folder=Path('.flint');folder.mkdir(exist_ok=True)
(folder/('training-'+os.environ.get('FLINT_RUN_ID','normal')+'.json')).write_text(json.dumps(
    {'losses':losses,'step_ms':times,'dispatch':recurrence.status()},indent=2))
