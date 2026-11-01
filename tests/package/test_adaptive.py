import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from flint_gpu.observation import Observation
from flint_gpu.units import workload, subscribe
from flint_gpu.config import DEFAULTS


@pytest.fixture(autouse=True)
def isolated_compiler_cache():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def test_observation_stability_is_scoped_and_budgeted():
    o=Observation(max_units=16)
    for _ in range(7):assert o.add(.01,{'a'})=='observing'
    assert o.add(.01,{'a'})=='sufficient'
    assert o.report()['signature_count']==1
    o=Observation(max_units=12)
    for i in range(12):state=o.add(.01,{str(i)})
    assert state=='inconclusive'
    assert len(o.history)<=16


def test_new_signature_and_startup_reset_stability():
    o=Observation(max_units=40)
    for i in range(6):o.add(2 if i==0 else .01,{'a'})
    assert o.add(.01,{'a','b'})=='observing'
    for _ in range(6):assert o.add(.01,{'a','b'})=='observing'
    assert o.add(.01,{'a','b'})=='sufficient'


def test_batch_boundary_nested_and_failed_calls():
    events=[];remove=subscribe(lambda *a:events.append(a),'batch')
    @workload
    def inner(x):return x+1
    @workload
    def outer(x):return inner(x)
    @workload
    def fails():raise ValueError('failed work')
    try:
        assert outer(1)==2
        with pytest.raises(ValueError):fails()
        assert len(events)==1 and events[0][0]=='batch'
    finally:remove()


def test_batch_rehearsal_uses_completed_work_units(monkeypatch):
    from flint_gpu.rehearsal import Rehearsal,RehearsalComplete
    monkeypatch.setattr('flint_gpu.config.read_config',lambda:{'workload_kind':'batch'})
    @workload
    def predict(x):return x+1
    rehearsal=Rehearsal(3).start()
    try:
        assert predict(1)==2 and predict(2)==3
        with pytest.raises(RehearsalComplete):predict(3)
        assert rehearsal.report()['complete']
    finally:rehearsal.stop()


def test_live_requires_replay_permission_and_never_calls_provider(monkeypatch,tmp_path):
    from flint_gpu.live import Live
    from flint_gpu.compilation import REGISTRY
    monkeypatch.setattr('flint_gpu.live.Observation',lambda **kw:SimpleNamespace(units=8,add=lambda *a:'sufficient',report=lambda:{}))
    runtime=SimpleNamespace(hash='abc',seen=[{'signature':[]}],disabled=False)
    monkeypatch.setattr('flint_gpu.compilation.REGISTRY',{'r':runtime})
    live=Live(tmp_path,'owner',dict(DEFAULTS),exchange=lambda _:pytest.fail('unsafe replay'))
    live.boundary('training',.01)
    assert live.phase=='replay_contract_required'


def test_live_rejection_retains_reference(monkeypatch,tmp_path):
    from flint_gpu.live import Live
    monkeypatch.setattr('flint_gpu.live.Observation',lambda **kw:SimpleNamespace(units=8,add=lambda *a:'sufficient',report=lambda:{}))
    runtime=SimpleNamespace(registry_key='r',hash='abc',seen=[{'signature':[]}],disabled=False,pin=None,pin_missing=False,performance_qualified=False)
    monkeypatch.setattr('flint_gpu.compilation.REGISTRY',{'r':runtime})
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    live=Live(tmp_path,'owner',{**DEFAULTS,'replay_safe':True},exchange=lambda _: {'accepted':False,'reason':'wrong gradient'})
    live.boundary('training',.01)
    assert live.phase=='reference_retained' and not runtime.performance_qualified
    assert live.events[0]['reason']=='wrong gradient'


def test_compiled_runtime_retraces_after_boundary_and_old_backward_survives(monkeypatch,tmp_path):
    from flint_gpu import Fast
    from flint_gpu.live import activate
    from kernel_contract.cases import FUZZ_POLICY
    monkeypatch.setenv('FLINT_CONFIG',str(tmp_path/'missing.toml'))
    monkeypatch.setenv('FLINT_CACHE',str(tmp_path/'cache'))
    monkeypatch.setenv('FLINT_LIVE','1')
    path=tmp_path/'model.py';path.write_text('def reference(x):\n    return x * 2\n')
    spec=importlib.util.spec_from_file_location('adaptive_test_model',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    (tmp_path/'.flint').mkdir()
    (tmp_path/'.flint/qualification.json').write_text('{"require_unit_gain":true}')
    runtime=Fast(module.reference,2)
    graphs=[]
    def backend(gm,inputs):graphs.append(gm);return gm.forward
    compiled=torch.compile(runtime,backend=backend,fullgraph=True)
    x=torch.ones(4,requires_grad=True)
    old=compiled(x)
    assert torch.equal(old,x*2)
    runtime.job.mkdir(parents=True)
    (runtime.job/'status.json').write_text(json.dumps({'fuzz_policy':FUZZ_POLICY,'kernel_sha256':'new'}))
    monkeypatch.setattr('flint_gpu.qualification.activation',lambda *a:(True,'verified test fixture'))
    def load():
        torch.manual_seed(999)
        runtime.kernel=lambda x:x+x
        runtime.verified=[runtime._signature((x,))]
        runtime.native_loaded=True;runtime.performance_qualified=True;runtime.activation_reason=None
    monkeypatch.setattr(runtime,'_load_if_ready',load)
    monkeypatch.setattr(runtime,'status',lambda:{'kernel_sha256':'new'})
    rng=torch.get_rng_state().clone()
    assert activate([runtime])['activated']
    assert torch.equal(torch.get_rng_state(),rng)
    old.sum().backward()
    assert torch.equal(x.grad,torch.full_like(x,2))
    assert torch.equal(compiled(x),x*2)
    assert len(graphs)==2
    assert runtime.stats['compiled kernel graphs']==1
    y=torch.ones(7,requires_grad=True)
    assert torch.equal(compiled(y),y*2)
    assert runtime.stats['compiled reference graphs: unsupported signature']==1


def test_compound_fuzzer_catches_bug_independent_axes_miss(tmp_path):
    from kernel_contract.spec import capture
    from kernel_contract.cases import build_cases
    from kernel_contract.runner import run_contract
    spec=capture('torch.softmax',(torch.randn(3,7),1))
    path=tmp_path/'bad.py'
    path.write_text('import torch\ndef kernel_function(x,dim):\n    y=torch.softmax(x,dim)\n    return y+1 if not x.is_contiguous() and bool((x<=-1).all()) else y\n')
    cases=build_cases(spec)
    assert run_contract(spec,path,device='cpu',cases=[c for c in cases if c['axis']!='compound']).outcome=='accept'
    result=run_contract(spec,path,device='cpu',cases=[c for c in cases if c['axis']=='compound'])
    assert result.outcome=='wrong'
    assert any(r['reproducer'].get('transform')=='negative' for r in result.failures)


def test_holdout_is_reproducible_and_not_recurrence_specific():
    from kernel_contract.holdout import holdout_cases
    a=holdout_cases(None,seed=123)
    assert a==holdout_cases(None,seed=123) and a!=holdout_cases(None,seed=124)
    assert all('stress' not in c for c in a['cases'])
    assert any('transform' in c and 'layout' in c for c in a['cases'])


def test_noisy_window_does_not_count_toward_stable_evidence():
    o=Observation(max_units=24)
    for seconds in (10.,.01,.01,.01):o.add(seconds,{'a'})
    assert o.stable==0
    for _ in range(4):assert o.add(.01,{'a'})=='observing'
    for _ in range(3):assert o.add(.01,{'a'})=='observing'
    assert o.add(.01,{'a'})=='sufficient'


def test_lost_controller_never_resumes_on_busy_worker(monkeypatch,tmp_path):
    from flint_gpu.live import Live,ResumeBlocked
    from flint_gpu.optimize_worker import atomic
    live=Live(tmp_path,'owner',dict(DEFAULTS))
    monkeypatch.setattr('flint_gpu.leases.release',lambda owner:None)
    def busy(*args):raise RuntimeError('another session owns GPU')
    monkeypatch.setattr('flint_gpu.leases.acquire',busy)
    def respond(*args,**kwargs):atomic(tmp_path/'live-response.json',{'accepted':False})
    monkeypatch.setattr('builtins.print',respond)
    with pytest.raises(ResumeBlocked):live.request({})


def test_initial_compilation_does_not_exhaust_observation_budget():
    o=Observation(max_seconds=1)
    o.started=0
    assert o.add(10.,{'a'},now=100)=='observing'
    assert o.add(.01,{'a'},now=100.1)=='observing'
    assert o.add(.01,{'a'},now=102)=='inconclusive'


def test_adaptive_discovery_reports_actual_bounded_window(tmp_path,monkeypatch):
    from flint_gpu.discovery import Window
    monkeypatch.chdir(tmp_path)
    (tmp_path/'pyproject.toml').write_text('[tool.flint]\nlaunch_policy="adaptive"\nobservation_max_units=12\n')
    (tmp_path/'train.py').write_text('import torch,time\nx=torch.nn.Parameter(torch.ones(1))\no=torch.optim.SGD([x],lr=.1)\nfor i in range(20):\n    x.square().sum().backward();o.step();time.sleep(.003)\n')
    window=Window(tmp_path,tmp_path/'.flint/discovery',warmup=0,steps=2)
    r=window.run(['train.py'])
    assert r['training_success'] and r['profiled_steps']==12
    assert r['observation']['state']=='inconclusive'  # no attributable project tensor helper
    assert r['instrumentation_removed']


def test_startup_speedup_cannot_admit_slower_live_work():
    from flint_gpu.qualification import decision
    pairs=[]
    for _ in range(3):
        pairs.append({m:{'success':True,'seconds':sec,'peak_allocated_bytes':100,'artifacts':{'x':'hash'},
                         'work_units':{'kind':'batch','synchronized':True,'median_seconds':unit}}
                      for m,sec,unit in [('reference',10,.01),('benchmark',5,.02)]})
    assert decision(pairs)[0]
    assert not decision(pairs,require_unit_gain=True)[0]
    for p in pairs:p['benchmark']['work_units']['median_seconds']=.005
    assert decision(pairs,require_unit_gain=True)[0]
