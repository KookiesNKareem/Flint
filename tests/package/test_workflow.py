import json
import sys
from pathlib import Path
import pytest
import torch
from flint_gpu.rehearsal import Rehearsal,RehearsalComplete
from flint_gpu.suggestions import functions,propose_region,apply_proposal


def test_bounded_rehearsal_stops_after_updates_and_removes_hook():
    x=torch.nn.Parameter(torch.ones(1));opt=torch.optim.SGD([x],lr=.1)
    rehearsal=Rehearsal(2).start()
    try:
        with pytest.raises(RehearsalComplete):
            for _ in range(10):opt.zero_grad();x.sum().backward();opt.step()
    finally:rehearsal.stop()
    assert rehearsal.report()['complete'] and rehearsal.completed==2
    opt.step()  # no leftover stop hook


def test_region_extraction_preserves_outputs_and_gradients(tmp_path):
    source='''import torch
def scan(a,b):
    return (a.cumsum(1)+b).tanh()
class Model:
    def forward(self,x,g):
        if x.shape[1]==1:
            out=x+g
        else:
            a=-torch.nn.functional.softplus(g)
            b=x.sigmoid()
            out=scan(a,b)
        return out*2
'''
    path=tmp_path/'model.py';path.write_text(source)
    item=next(v for v in functions(tmp_path).values() if v['function']=='scan')
    proposal=propose_region(tmp_path,item,{'share':.5})
    assert proposal['kind']=='extract_region' and 'scan_region' in proposal['diff']
    # Test extraction semantics independently of Flint dispatch/decorator behavior.
    original={};changed={};exec(source,original)
    plain=proposal['after'].replace('import flint_gpu as flint\n','').replace('@flint.fast\n','')
    exec(plain,changed)
    for length in (1,7):
        x=torch.randn(2,length,3,requires_grad=True);g=torch.randn_like(x,requires_grad=True)
        a=original['Model']().forward(x,g);b=changed['Model']().forward(x,g)
        torch.testing.assert_close(a,b)
        for u,v in zip(torch.autograd.grad(a.sum(),(x,g)),torch.autograd.grad(b.sum(),(x,g))):torch.testing.assert_close(u,v)
    apply_proposal(proposal);assert path.read_text()==proposal['after']


def test_prepared_failure_never_launches_full_training(monkeypatch,tmp_path):
    import flint_gpu.workflow as w
    actions=[]
    monkeypatch.setattr(w,'rpc',lambda remote,request,log:actions.append(request['action']) or {'ready':False})
    monkeypatch.setattr(w,'optimize_request',lambda request:{'accepted':False,'reason':'memory_regression'})
    result=w.run(['train.py'],policy='prepared',out=tmp_path/'task')
    assert result['state']=='not_ready' and not result['launched'] and actions==['ready']


def test_background_starts_only_after_successful_foreground(monkeypatch,tmp_path):
    import flint_gpu.workflow as w
    actions=[]
    def rpc(remote,request,log):
        actions.append(request['action'])
        return {'ready':False} if request['action']=='ready' else {'launched':True,'result':{'success':True,'dispatch':[{'name':'scan'}]}}
    class Child:
        pid=123
        def __init__(self,*args,**kwargs):actions.append('background')
    monkeypatch.setattr(w,'rpc',rpc);monkeypatch.setattr(w.subprocess,'Popen',Child)
    result=w.run(['train.py'],policy='background',out=tmp_path/'task')
    assert actions==['ready','run','background'] and result['state']=='optimization_started'


def test_worker_lease_prevents_cross_project_contention(monkeypatch,tmp_path):
    from flint_gpu.leases import acquire,release
    monkeypatch.setenv('FLINT_WORKER_DIR',str(tmp_path/'worker'))
    acquire('one','profile')
    try:
        with pytest.raises(RuntimeError,match='busy'):acquire('two','optimization')
        release('two')
        with pytest.raises(RuntimeError,match='busy'):acquire('two','training')
    finally:release('one')
    acquire('two','training');release('two')


def test_unchanged_rejection_prevents_repeat_benchmarks_and_model_calls(monkeypatch,tmp_path):
    import flint_gpu.workflow as w
    actions=[]
    def rpc(remote,request,log):
        actions.append(request['action'])
        if request['action']=='ready':return {'ready':False,'retry_allowed':False,'reason':'speed inconclusive'}
        return {'launched':True,'result':{'success':True,'dispatch':[{}]}}
    monkeypatch.setattr(w,'rpc',rpc)
    def forbidden(*args,**kwargs):raise AssertionError('unchanged rejection must not retry')
    monkeypatch.setattr(w,'optimize_request',forbidden)
    result=w.run(['train.py'],policy='prepared',out=tmp_path/'prepared')
    assert result['state']=='not_ready' and actions==['ready']
    result=w.run(['train.py'],policy='background',out=tmp_path/'background')
    assert result['state']=='reference_retained' and actions==['ready','ready','run']


def test_rejection_cache_binds_source_environment_and_candidate(monkeypatch,tmp_path):
    from flint_gpu.workflow import rejected_unchanged
    from flint_gpu.qualification import snapshot,sha,SCHEMA
    monkeypatch.chdir(tmp_path);(tmp_path/'train.py').write_text('pass\n')
    cache=tmp_path/'.flint/cache';job=cache/'job';job.mkdir(parents=True)
    kernel=job/'kernel.py';kernel.write_text('pass\n');digest=sha(kernel)
    (job/'status.json').write_text(json.dumps({'kernel':'kernel.py','kernel_sha256':digest}))
    receipt={'schema':SCHEMA,'accepted':False,'environment':{},'workload':snapshot(tmp_path,['train.py']),
             'pairs':[{'reference':{'success':True},'benchmark':{'success':True,'artifacts':{'job':digest}}} for _ in range(3)]}
    assert rejected_unchanged(receipt,['train.py'],{'cache_dir':str(cache)},{})
    kernel.write_text('changed=True\n')
    assert not rejected_unchanged(receipt,['train.py'],{'cache_dir':str(cache)},{})
    kernel.write_text('pass\n');(tmp_path/'train.py').write_text('changed=True\n')
    assert not rejected_unchanged(receipt,['train.py'],{'cache_dir':str(cache)},{})


@pytest.mark.parametrize('choice',[True,False,None])
def test_pending_tradeoff_is_reviewed_on_next_foreground_run(monkeypatch,tmp_path,choice):
    import flint_gpu.workflow as w
    actions=[];ready_calls=[]
    def rpc(remote,request,log):
        actions.append(request['action'])
        if request['action']=='ready':
            ready_calls.append(1)
            return {'ready':True} if len(ready_calls)>1 else {'ready':False,'approval_available':True,'retry_allowed':False}
        return {'launched':True,'result':{'success':True}}
    monkeypatch.setattr(w,'rpc',rpc)
    monkeypatch.setattr('flint_gpu.approval.review_on_terminal',lambda *a,**kw: {'accepted':True} if choice is True else choice)
    monkeypatch.setattr(w,'optimize_request',lambda r:pytest.fail('Pending review must not regenerate'))
    result=w.run(['train.py'],policy='prepared',out=tmp_path/'task')
    assert result['state']==('completed' if choice is True else 'review_required')
    assert ('run' in actions)==(choice is True)


@pytest.mark.parametrize('choice',[False,None])
def test_adaptive_pending_review_keeps_reference_without_reoptimization(monkeypatch,tmp_path,choice):
    import flint_gpu.workflow as w
    def rpc(remote,request,log):
        if request['action']=='ready':return {'ready':False,'approval_available':True,'retry_allowed':False}
        assert request['action']=='run' and request['retain_reference'] is True
        return {'launched':True,'result':{'success':True}}
    monkeypatch.setattr(w,'rpc',rpc)
    monkeypatch.setattr('flint_gpu.approval.review_on_terminal',lambda *a,**kw:choice)
    monkeypatch.setattr(w,'optimize_request',lambda r:pytest.fail('Review must not restart optimization'))
    result=w.run(['train.py'],policy='adaptive',out=tmp_path/'task')
    assert result['state']=='completed'
