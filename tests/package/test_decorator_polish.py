import json
import pytest
import torch
import flint_gpu as flint
from flint_gpu.optimization import Session


def scaled(x, scale=2):
    return x.sin() * scale


@pytest.mark.parametrize('compiled', [False, True])
def test_keywords_defaults_use_same_accepted_path(monkeypatch, tmp_path, compiled):
    monkeypatch.setenv('FLINT_CACHE', str(tmp_path/'cache'))
    monkeypatch.setenv('FLINT_MODE', 'offline')
    wrapped=flint.fast(scaled)
    x=torch.ones(3,requires_grad=True)
    wrapped.kernel=lambda x,scale: x.sin()*scale
    wrapped.native_loaded=True
    wrapped.verified=[flint.signature((x,2))]
    def forbidden(*a,**kw):raise AssertionError('accepted call inspected source/files')
    original_hash=flint.source_hash
    monkeypatch.setattr(flint,'source_hash',lambda fn: forbidden() if fn is scaled else original_hash(fn))
    fn=torch.compile(wrapped,fullgraph=True,backend='aot_eager') if compiled else wrapped
    for call in (lambda:fn(x),lambda:fn(x=x),lambda:fn(x,scale=2)):
        result=call();torch.testing.assert_close(result,scaled(x))
    result.sum().backward();torch.testing.assert_close(x.grad,2*x.detach().cos())
    if not compiled:assert wrapped.stats['kernel']==3
    torch.testing.assert_close(fn(x,scale=3),scaled(x,3))
    with pytest.raises(TypeError):wrapped(x,x=x)
    with pytest.raises(TypeError):wrapped(unknown=x)


def test_explicit_empty_backward_disables_inherited_handoff(monkeypatch,tmp_path):
    monkeypatch.setenv('FLINT_CACHE',str(tmp_path))
    assert flint.fast(scaled,backward_model='').backward_model==''
    monkeypatch.setenv('FLINT_BACKWARD_MODEL','')
    assert flint.fast(scaled).backward_model==''


def test_controller_model_policy_overrides_worker_handoff(monkeypatch,tmp_path):
    job=tmp_path/'job';job.mkdir()
    (job/'job.json').write_text(json.dumps({'models':['old'],'backward_model':'wonder'}))
    (job/'status.json').write_text(json.dumps({'state':'ready','compile_passed':True,'holdout_passed':True}))
    s=Session.__new__(Session);s.only_model='spark';s.env={}
    s.remaining=lambda:1;s.left=lambda:10;s.event=lambda **kw:None;s.budget=lambda:{}
    class Done:
        def __init__(self,*a,**kw):pass
        def wait(self,**kw):return 0
    monkeypatch.setattr('flint_gpu.optimization.subprocess.Popen',Done)
    assert s.generate(job)
    policy=json.loads((job/'job.json').read_text())
    assert policy['models']==['spark'] and policy['backward_model']==''
    with pytest.raises(ValueError,match='single-model'):s.generate(job,{'kind':'performance'},'wonder')


def test_uninspectable_callable_keeps_reference(monkeypatch,tmp_path):
    monkeypatch.setenv('FLINT_CACHE',str(tmp_path))
    wrapped=flint.fast(torch.sin)
    x=torch.ones(3)
    torch.testing.assert_close(wrapped(x),x.sin())
