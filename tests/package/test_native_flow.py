import json
from pathlib import Path
import pytest
from flint_gpu import workflow as w
from flint_gpu.config import read_config
from flint_gpu.setup import save_settings,setup,save_credentials,provider_environment


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch,tmp_path):
    monkeypatch.setenv('FLINT_CONFIG',str(tmp_path/'personal.toml'))
    monkeypatch.setenv('FLINT_CACHE',str(tmp_path/'cache'))
    monkeypatch.setenv('FLINT_WORKER_DIR',str(tmp_path/'worker'))


def test_setup_preserves_settings_budget_and_credentials(monkeypatch,tmp_path):
    path=tmp_path/'personal.toml';path.write_text('# mine\n[flint]\nremote="my-worker"\n[other]\nkeep=42\n')
    ledger=tmp_path/'budget.json';ledger.write_text('{"usd":3.29,"reservations":{}}')
    setup('prepared',6,str(ledger),interactive=False)
    config=read_config(tmp_path)
    assert config['launch_policy']=='prepared' and config['max_usd']==6
    assert config['budget_ledger']==str(ledger) and config['remote']=='my-worker'
    assert config['setup_complete'] and config['auto_build'] is False
    assert 'keep=42' in path.read_text() and json.loads(ledger.read_text())['usd']==3.29
    save_credentials({'MODEL_API_KEY':'test-private-value','ignored':'no'})
    secret=path.with_name('providers.json')
    assert secret.stat().st_mode & 0o777==0o600
    assert 'test-private-value' not in path.read_text()
    assert provider_environment({})['MODEL_API_KEY']=='test-private-value'
    assert provider_environment({'MODEL_API_KEY':'override'})['MODEL_API_KEY']=='override'


def test_project_cannot_raise_or_reset_saved_budget(tmp_path):
    save_settings({'max_usd':6,'budget_ledger':str(tmp_path/'shared.json'),'launch_policy':'background'})
    project=tmp_path/'project';project.mkdir()
    (project/'pyproject.toml').write_text('[tool.flint]\nmax_usd=99\nbudget_ledger="fresh.json"\nlaunch_policy="prepared"\n')
    c=read_config(project)
    assert c['max_usd']==6 and c['budget_ledger']==str(tmp_path/'shared.json') and c['launch_policy']=='prepared'


def fake_flow(monkeypatch,tmp_path,answers,policy='background'):
    save_settings({'launch_policy':policy,'max_usd':6,'budget_ledger':str(tmp_path/'shared.json'),'optimization_minutes':9})
    events=[];accepted=False;answers=iter(answers)
    proposal={'id':'proposal','diff':'--- a/model.py\n+++ b/model.py\n+@flint.fast\n','workload_sources':{'model.py':'hash'}}
    report={'training_success':True,'suggestions':[proposal],'source_hashes':{'model.py':'hash'}}
    def rpc(remote,request,log):
        nonlocal accepted
        events.append((request['action'],request))
        if request['action']=='ready':return {'ready':False,'has_targets':accepted,'cuda_available':True}
        if request['action']=='review_valid':return {'valid':True}
        if request['action']=='discover':return {'result':report}
        if request['action']=='apply':accepted=True;return {'applied':'proposal'}
        if request['action']=='run':return {'launched':True,'result':{'success':True,'dispatch':[{}]}}
        raise AssertionError(request)
    monkeypatch.setattr(w,'rpc',rpc)
    monkeypatch.setattr(w,'review_proposal',lambda proposal:next(answers))
    def background(request,out):events.append(('background',request));return {'state':'optimization_started'}
    monkeypatch.setattr(w,'start_background',background)
    def optimize(request):events.append(('optimize',request));return {'accepted':True}
    monkeypatch.setattr(w,'optimize_request',optimize)
    return events


def test_one_command_reviews_applies_and_starts_background(monkeypatch,tmp_path):
    events=fake_flow(monkeypatch,tmp_path,[True])
    assert w.run(['train.py'],out=tmp_path/'first')['state']=='optimization_started'
    assert [a for a,_ in events]==['ready','discover','apply','ready','background']
    options=events[-1][1]['optimization']
    assert options['ledger']==str(tmp_path/'shared.json') and options['max_usd']==6 and options['minutes']==9
    assert 'rehearsal_steps' not in events[1][1]['discovery']


def test_prepared_discovers_bounded_then_qualifies_before_full_launch(monkeypatch,tmp_path):
    events=fake_flow(monkeypatch,tmp_path,[True],policy='prepared')
    assert w.run(['train.py'],out=tmp_path/'first')['state']=='completed'
    assert [a for a,_ in events]==['ready','discover','apply','ready','optimize','run']
    assert events[1][1]['discovery']['rehearsal_steps']==20
    assert events[-1][1]['required'] is True


def test_noninteractive_review_resumes_without_replaying_for_the_diff(monkeypatch,tmp_path):
    events=fake_flow(monkeypatch,tmp_path,[None,True])
    assert w.run(['train.py'],out=tmp_path/'first')['state']=='review_required'
    assert w.run(['train.py'],out=tmp_path/'second')['state']=='optimization_started'
    assert [a for a,_ in events].count('discover')==1
    assert [a for a,_ in events].count('review_valid')==1


def test_decline_is_remembered_and_never_starts_models(monkeypatch,tmp_path):
    events=fake_flow(monkeypatch,tmp_path,[False])
    assert w.run(['train.py'],out=tmp_path/'first')['state']=='reference_retained'
    assert w.run(['train.py'],out=tmp_path/'second')['state']=='reference_retained'
    assert all(a not in ('apply','background','optimize') for a,_ in events)


def test_target_detection_uses_actual_flint_imports(tmp_path):
    path=tmp_path/'model.py';path.write_text('import other\n@other.fast\ndef x(y):return y\n')
    assert not w.has_targets(tmp_path)
    path.write_text('from flint_gpu import fast as accelerate\n@accelerate\ndef x(y):return y\n')
    assert w.has_targets(tmp_path)
    path.write_text('import flint_gpu as f\n@f.fast(observe_calls=2)\ndef x(y):return y\n')
    assert w.has_targets(tmp_path)


def test_discovery_rehearsal_stops_at_boundary_before_full_program(tmp_path,monkeypatch):
    from flint_gpu.discovery import Window
    monkeypatch.chdir(tmp_path)
    (tmp_path/'train.py').write_text('import torch\nfrom pathlib import Path\nx=torch.nn.Parameter(torch.ones(1))\no=torch.optim.SGD([x],lr=.1)\nfor i in range(20):\n    x.sum().backward();o.step()\nPath("full-completed").touch()\n')
    result=Window(tmp_path,tmp_path/'.flint/discovery',warmup=1,steps=2).run(['train.py'],rehearsal_steps=4)
    assert result['training_success'] and result['rehearsal']['completed_steps']==4
    assert result['profiled_steps']==2 and not (tmp_path/'full-completed').exists()


def test_repeated_run_waits_for_owned_background_job(monkeypatch,tmp_path):
    request={'remote':'bash -c','project':str(tmp_path),'command':['train.py']}
    task=tmp_path/'previous';task.mkdir();(task/'request.json').write_text('{}')
    (task/'state.json').write_text('{"state":"optimizing","pid":123}')
    (tmp_path/('.active-'+w.workload_key(request)+'.json')).write_text(json.dumps({'task':str(task)}))
    monkeypatch.setattr(w,'controller_alive',lambda task,pid:pid==123)
    waits=[]
    def finish(seconds):
        waits.append(seconds);(task/'state.json').write_text('{"state":"ready"}')
    monkeypatch.setattr(w.time,'sleep',finish)
    w.wait_for_previous(request,tmp_path/'next')
    assert waits==[.5]


def test_discovery_detaches_without_breaking_other_optimizer_hooks(tmp_path,monkeypatch):
    from flint_gpu.discovery import Window
    from torch.optim.optimizer import register_optimizer_step_post_hook,_global_optimizer_post_hooks
    import torch
    monkeypatch.chdir(tmp_path)
    (tmp_path/'train.py').write_text('import torch\nx=torch.nn.Parameter(torch.ones(1))\no=torch.optim.SGD([x],lr=.1)\nfor i in range(8):\n    x.sum().backward();o.step()\n')
    window=Window(tmp_path,tmp_path/'.flint/discovery',warmup=1,steps=2)
    observations=[];handles=[]
    def register(hook):
        handle=register_optimizer_step_post_hook(hook)
        handles.append(register_optimizer_step_post_hook(lambda *args:observations.append(1)))
        return handle
    import importlib
    monkeypatch.setattr(importlib.import_module('torch.optim.optimizer'),'register_optimizer_step_post_hook',register)
    before=len(_global_optimizer_post_hooks)
    try:
        result=window.run(['train.py'])
        assert result['training_success'] and len(observations)==8
        assert len(_global_optimizer_post_hooks)==before+1
    finally:
        for h in handles:h.remove()
