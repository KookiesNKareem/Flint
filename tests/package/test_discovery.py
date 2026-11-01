import json
import sys
from collections import Counter
from pathlib import Path
import pytest
from flint_gpu.suggestions import functions,propose,apply_proposal,eligibility
from flint_gpu.discovery import Window,rank


def test_proposal_preserves_reference_and_refuses_stale_or_modified_patch(tmp_path):
    source='"""Example."""\nfrom __future__ import annotations\n\ndef recurrence(x):\n    return x.sin().cos()\n'
    p=tmp_path/'model.py';p.write_text(source);item=next(iter(functions(tmp_path).values()))
    proposal=propose(tmp_path,item,{'share':.6})
    assert p.read_text()==source
    assert '@flint.fast\ndef recurrence' in proposal['after']
    assert proposal['after'].index('from __future__')<proposal['after'].index('import flint_gpu')
    altered={**proposal,'after':proposal['after'].replace('sin()','exp()')}
    with pytest.raises(ValueError,match='more than'):apply_proposal(altered)
    apply_proposal(proposal);assert p.read_text()==proposal['after']
    assert json.loads((tmp_path/'.flint/qualification.json').read_text())['accepted'] is False
    with pytest.raises(ValueError,match='Source changed'):apply_proposal(proposal)


def test_ranking_avoids_library_ops_and_cpu_guesses(tmp_path):
    (tmp_path/'model.py').write_text('def scan(x):\n    return x.cumsum(1)\n\ndef norm(x):\n    return x\n')
    inventory=functions(tmp_path);keys=list(inventory)
    records={k:{'device_us':60 if i==0 else 40,'cpu_us':5,'forward_us':20,'backward_us':40,
                'ops':Counter({'aten::cumsum':60} if i==0 else {'aten::native_layer_norm':40})}
             for i,k in enumerate(keys)}
    ranked=rank(records,inventory,100,10)
    assert ranked[0]['function']=='scan' and ranked[0]['suggest']
    assert not ranked[1]['suggest'] and 'library' in ranked[1]['reason']
    assert all(not r['suggest'] for r in rank(records,inventory,0,10))


def test_training_controls_are_not_kernel_proposals(tmp_path):
    (tmp_path/'train.py').write_text('def train(loss):\n    loss.backward()\n')
    assert 'training control' in eligibility(next(iter(functions(tmp_path).values())))


def test_discovery_is_bounded_and_restores_training_and_profiler(tmp_path,monkeypatch):
    monkeypatch.setenv('FLINT_CONFIG',str(tmp_path/'no-personal-config.toml'))
    monkeypatch.chdir(tmp_path)
    (tmp_path/'train.py').write_text('''import torch
from pathlib import Path
def recurrence(x):
    return (x.sin().cos()).sum()
x=torch.nn.Parameter(torch.ones(8));opt=torch.optim.SGD([x],lr=.1)
for i in range(9):
    opt.zero_grad();recurrence(x).backward();opt.step()
Path('finished.txt').write_text(str(i))
''')
    before=(tmp_path/'train.py').read_bytes();previous=sys.getprofile()
    window=Window(tmp_path,tmp_path/'.flint/discovery',warmup=2,steps=3)
    result=window.run(['train.py'])
    assert window.profile is None and not window.inventory and not window.code
    assert result['training_success'] and result['profiled_steps']==3
    assert result['optimizer_steps_seen']==5 and result['instrumentation_removed']
    assert sys.getprofile() is previous and (tmp_path/'finished.txt').read_text()=='8'
    assert before==(tmp_path/'train.py').read_bytes() and result['model_calls']==0
    assert any(r['function']=='recurrence' for r in result['regions'])


def test_cuda_annotation_ranges_are_not_counted_as_kernel_work(tmp_path):
    from types import SimpleNamespace as E
    from flint_gpu.discovery import analyze,PREFIX
    (tmp_path/'model.py').write_text('def scan(x):\n    return x.cumsum(1)\n')
    inventory=functions(tmp_path);key=next(iter(inventory))
    base=dict(device_type='CPU',kernels=[],self_cpu_time_total=0,scope=0,sequence_nr=-1,thread=1,
              fwd_thread=1,is_user_annotation=False,cpu_parent=None,self_device_time_total=0)
    scope=E(**{**base,'id':1,'name':PREFIX+key,'is_user_annotation':True})
    op=E(**{**base,'id':2,'name':'aten::cumsum','kernels':[E(duration=20)],'cpu_parent':scope})
    kernel=E(**{**base,'id':3,'name':'scan_kernel','device_type':'CUDA','self_device_time_total':20})
    annotation=E(**{**base,'id':4,'name':PREFIX+key,'device_type':'CUDA','self_device_time_total':1000,'is_user_annotation':True})
    result=analyze([scope,op,kernel,annotation],inventory,.1)
    assert result['device_work_us']==20 and result['regions'][0]['share']==1
    assert result['unattributed_device_us']==0


def test_profiler_failure_is_fail_open(tmp_path,monkeypatch):
    import torch
    monkeypatch.chdir(tmp_path)
    (tmp_path/'train.py').write_text('import torch\nx=torch.nn.Parameter(torch.ones(1))\no=torch.optim.SGD([x],lr=.1)\nfor _ in range(3):\n    x.sum().backward();o.step()\nopen("finished", "w").write("yes")\n')
    def fail(*args,**kwargs):raise RuntimeError('profiler unavailable')
    monkeypatch.setattr(torch.profiler,'profile',fail)
    result=Window(tmp_path,tmp_path/'.flint/discovery',1,1).run(['train.py'])
    assert result['training_success'] and result['instrumentation_removed']
    assert 'profiler unavailable' in result['profiling_error']
    assert (tmp_path/'finished').read_text()=='yes'
