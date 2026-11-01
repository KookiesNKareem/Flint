"""Small byte-language-model experiment with connected recurrent states."""
import argparse,json,os,time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from recurrence import recurrence

ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=20);ap.add_argument('--length',type=int,default=128)
ap.add_argument('--eager',action='store_true');ap.add_argument('--capture',action='store_true');ap.add_argument('--data',default='dataset.txt');a=ap.parse_args()
torch.manual_seed(1337)
from model import Model
net=Model().cuda();optimizer=torch.optim.AdamW(net.parameters(),lr=.001)
data=torch.tensor(list(Path(a.data).read_bytes()),dtype=torch.long,device='cuda')
cut=int(len(data)*.9);training=data[:cut];validation=data[cut:]
rng=torch.Generator(device='cuda').manual_seed(123)
def batch(source,rng):
    starts=torch.randint(len(source)-a.length-1,(2,),device='cuda',generator=rng)
    positions=starts[:,None]+torch.arange(a.length+1,device='cuda')[None,:]
    tokens=source[positions];return tokens[:,:-1],tokens[:,1:]
if a.capture:
    x,y=batch(training,rng)
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=F.cross_entropy(net(x).flatten(0,1),y.flatten())
    loss.backward();print(recurrence.job);print(recurrence.status());raise SystemExit(0)
if not a.eager:net=torch.compile(net,fullgraph=True,dynamic=False)
losses=[];times=[]
for step in range(a.steps):
    torch.cuda.synchronize();start=time.perf_counter();optimizer.zero_grad(set_to_none=True)
    values=[]
    for _ in range(2):
        x,y=batch(training,rng)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=F.cross_entropy(net(x).flatten(0,1),y.flatten())
        (loss/2).backward();values.append(float(loss.detach()))
    torch.nn.utils.clip_grad_norm_(net.parameters(),1.0);optimizer.step();torch.cuda.synchronize()
    times.append((time.perf_counter()-start)*1000);losses.append(sum(values)/2)
    print(json.dumps({'step':step,'loss':losses[-1],'ms':times[-1]}),flush=True)
with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
    x,y=batch(validation,torch.Generator(device='cuda').manual_seed(42));val=float(F.cross_entropy(net(x).flatten(0,1),y.flatten()))
out=Path('.flint')/('result-'+os.environ.get('FLINT_RUN_ID','normal')+'.json');out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps({'losses':losses,'times_ms':times,'validation_loss':val,'peak_bytes':torch.cuda.max_memory_allocated(),'dispatch':recurrence.status()},indent=2))
print('validation',val,flush=True)

torch.save(net.state_dict(),out.with_suffix(".pt"))
