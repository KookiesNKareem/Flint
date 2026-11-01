import json
from pathlib import Path
import torch
import pytest
import muse_kernel
from muse_kernel.config import read_config


def plain(x):
    return x.sin() * 2


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4))
        self.bias = torch.nn.Parameter(torch.randn(4))

    @muse_kernel.fast
    def forward(self, x):
        return torch.nn.functional.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


def test_fullgraph_cold_capture_from_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv('MUSE_KERNEL_CACHE', str(tmp_path))
    monkeypatch.setenv('MUSE_MODE', 'capture')
    wrapped = muse_kernel.fast(plain, observe_calls=1)
    compiled = torch.compile(wrapped, backend='aot_eager', fullgraph=True)
    x = torch.randn(3, 4, requires_grad=True)
    y = compiled(x); y.sum().backward()
    torch.testing.assert_close(y, plain(x))
    torch.testing.assert_close(x.grad, 2*x.detach().cos())
    assert wrapped.status()['state'] == 'queued'
    job = json.loads((wrapped.job/'job.json').read_text())
    assert job['native'] and not job['domains']
    assert job['verified_signatures'][0][0]['device'] == 'cpu'
    assert job['verified_signatures'][0][0]['shape'] == [3, 4]
    # Captured inputs hold no training graph or GPU storage.
    assert wrapped.seen[0]['args'][0].device.type == 'meta'


def test_bound_method_keeps_live_parameters(monkeypatch, tmp_path):
    monkeypatch.setenv('MUSE_KERNEL_CACHE', str(tmp_path))
    monkeypatch.setenv('MUSE_MODE', 'capture')
    class Affine(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.randn(4))
        @muse_kernel.fast(observe_calls=1)
        def forward(self, x):
            return x * self.weight
    module = Affine()
    fn = torch.compile(module, fullgraph=True, backend='aot_eager')
    x = torch.randn(2, 4, requires_grad=True)
    torch.testing.assert_close(fn(x), x * module.weight)
    with torch.no_grad(): module.weight.add_(1)
    y = fn(x); y.sum().backward()
    torch.testing.assert_close(y, x*module.weight)
    torch.testing.assert_close(module.weight.grad, x.detach().sum(0))
    job = json.loads((Affine.forward.runtime.job/'job.json').read_text())
    assert '_muse_weight' in job['source']
    assert 'self.' not in job['source']


def test_accepted_dispatch_has_no_source_or_file_io(monkeypatch, tmp_path):
    monkeypatch.setenv('MUSE_KERNEL_CACHE', str(tmp_path));monkeypatch.setenv('MUSE_MODE','offline')
    wrapped = muse_kernel.fast(plain)
    x = torch.randn(3, 4, requires_grad=True)
    wrapped.kernel = plain; wrapped.native_loaded = True
    wrapped.verified = [muse_kernel.signature((x,))]
    def forbidden(*a, **kw): raise AssertionError('hot-path I/O')
    monkeypatch.setattr(muse_kernel, 'source_hash', forbidden)
    monkeypatch.setattr(wrapped, '_poll', forbidden)
    monkeypatch.setattr(wrapped, '_values_ok', forbidden)
    fn = torch.compile(wrapped, fullgraph=True, backend='aot_eager')
    for _ in range(3): torch.testing.assert_close(fn(x), plain(x))
    # Different shape takes the ordinary reference, with no build in offline mode.
    y = torch.randn(5, 4, requires_grad=True)
    torch.testing.assert_close(fn(y), plain(y))
    assert not wrapped.job.exists()


def test_config_precedence(monkeypatch, tmp_path):
    user = tmp_path/'user.toml'; user.write_text('[muse]\nmax_usd=3\nmodels=["first"]\n')
    project = tmp_path/'project'; project.mkdir()
    (project/'pyproject.toml').write_text('[tool.muse]\nmax_usd=2\n')
    monkeypatch.setenv('MUSE_CONFIG', str(user))
    assert read_config(project)['max_usd'] == 2
    assert read_config(project)['models'] == ['first']
    monkeypatch.setenv('MUSE_KERNEL_MAX_USD','1')
    assert read_config(project)['max_usd'] == 1


def test_cold_trace_only_needs_one_metadata_capture(monkeypatch, tmp_path):
    monkeypatch.setenv('MUSE_KERNEL_CACHE', str(tmp_path));monkeypatch.setenv('MUSE_MODE','capture')
    wrapped=muse_kernel.fast(plain)
    torch.compile(wrapped,fullgraph=True,backend='aot_eager')(torch.ones(3))
    assert wrapped.status()['state']=='queued'


def test_unsupported_outputs_and_method_forms_keep_reference(monkeypatch,tmp_path):
    monkeypatch.setenv('MUSE_KERNEL_CACHE',str(tmp_path));monkeypatch.setenv('MUSE_MODE','capture')
    def fn(x):return x.sin(), 17
    wrapped=muse_kernel.fast(fn)
    result=torch.compile(wrapped,fullgraph=True,backend='aot_eager')(torch.ones(3))
    assert result[1]==17 and not wrapped.job.exists()
    class Defaults(torch.nn.Module):
        @muse_kernel.fast
        def forward(self,x,scale=2):return x*scale
    layer=Defaults()
    torch.testing.assert_close(torch.compile(layer,fullgraph=True,backend='aot_eager')(torch.ones(2)),torch.full((2,),2.))


def test_project_lock_prevents_unpinned_builds(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path);monkeypatch.setenv('MUSE_KERNEL_CACHE',str(tmp_path/'cache'))
    (tmp_path/'muse.lock').write_text(json.dumps({'version':1,'artifacts':{}}))
    wrapped=muse_kernel.fast(plain,observe_calls=1)
    x=torch.ones(4)
    torch.testing.assert_close(torch.compile(wrapped,fullgraph=True,backend='aot_eager')(x),plain(x))
    assert not wrapped.job.exists()


def ready_artifact(wrapped):
    """Minimal synthetic accepted cache; activation tests do not need CUDA."""
    from kernel_contract.evidence import sha256
    wrapped.job.mkdir(parents=True)
    kernel=wrapped.job/'kernel.py'
    kernel.write_text('def kernel_function(x):\n    return x.sin() * 2\n')
    (wrapped.job/'domains.json').write_text('{}')
    (wrapped.job/'job.json').write_text(json.dumps({
        'native':True,'require_holdout':True,'memory_policy':None,
        'environment':wrapped.environment,'cache_version':muse_kernel.CACHE_VERSION,
        'verified_signatures':[muse_kernel.signature((torch.ones(3),))],
    }))
    (wrapped.job/'status.json').write_text(json.dumps({
        'state':'ready','kernel':'kernel.py','kernel_sha256':sha256(kernel),
        'holdout_passed':True,'compile_passed':True,
    }))


@pytest.mark.parametrize('mode',['auto','offline'])
def test_correctness_alone_never_activates_default(monkeypatch,tmp_path,mode):
    monkeypatch.setenv('MUSE_KERNEL_CACHE',str(tmp_path))
    monkeypatch.setenv('MUSE_MODE',mode)
    wrapped=muse_kernel.fast(plain)
    ready_artifact(wrapped)
    wrapped._load_if_ready()
    assert wrapped.kernel is None and not wrapped.native_loaded
    assert wrapped.status()['state']=='reference'
    assert wrapped.status()['artifact_state']=='ready'
    assert 'performance' in wrapped.status()['activation_reason']
    def forbidden(*a,**k):raise AssertionError('withheld candidate should just run reference')
    monkeypatch.setattr(wrapped,'_observe_metadata',forbidden)
    x=torch.ones(3)
    torch.testing.assert_close(wrapped(x),plain(x))
    torch.testing.assert_close(torch.compile(wrapped,fullgraph=True,backend='aot_eager')(x),plain(x))


def test_explicit_benchmark_can_measure_checked_candidate(monkeypatch,tmp_path):
    monkeypatch.setenv('MUSE_KERNEL_CACHE',str(tmp_path))
    monkeypatch.setenv('MUSE_MODE','benchmark')
    wrapped=muse_kernel.fast(plain)
    ready_artifact(wrapped)
    wrapped._load_if_ready()
    assert wrapped.native_loaded and wrapped.activation_reason is None
    x=torch.ones(3)
    torch.testing.assert_close(torch.compile(wrapped,fullgraph=True,backend='aot_eager')(x),plain(x))


def test_qualified_artifact_activates_only_on_matching_next_run(monkeypatch,tmp_path):
    torch._dynamo.reset()
    from flint_gpu.qualification import snapshot, SCHEMA
    monkeypatch.chdir(tmp_path);monkeypatch.setattr('sys.argv',['train.py','--steps','20'])
    (tmp_path/'train.py').write_text('pass\n')
    monkeypatch.setenv('MUSE_KERNEL_CACHE',str(tmp_path/'.flint/cache'));monkeypatch.setenv('MUSE_MODE','auto')
    first=muse_kernel.fast(plain);ready_artifact(first)
    state=json.loads((first.job/'status.json').read_text())
    samples=[{'reference':{'success':True,'seconds':10,'peak_allocated_bytes':100},
              'benchmark':{'success':True,'seconds':7,'peak_allocated_bytes':90,
                           'artifacts':{first.job.name:state['kernel_sha256']}}} for _ in range(3)]
    receipt={'schema':SCHEMA,'accepted':True,'minimum_gain':.05,'pairs':samples,
             'preparation':samples[0],'cache_policy':'one_complete_command_per_mode_before_fixed_pairs_v1',
             'environment':first.environment,'workload':snapshot(tmp_path,['train.py','--steps','20'])}
    (tmp_path/'.flint/qualification.json').write_text(json.dumps(receipt))
    first._load_if_ready();assert first.kernel is None
    next_run=muse_kernel.fast(plain)
    assert next_run.native_loaded and next_run.performance_qualified
    torch.testing.assert_close(torch.compile(next_run,fullgraph=True,backend='aot_eager')(torch.ones(3)),plain(torch.ones(3)))
    (tmp_path/'train.py').write_text('changed = True\n')
    changed=muse_kernel.fast(plain)
    assert changed.kernel is None and 'changed' in changed.activation_reason


def test_prepared_guard_refuses_missing_artifact_and_uncovered_inputs(monkeypatch,tmp_path):
    monkeypatch.setenv('MUSE_KERNEL_CACHE',str(tmp_path));monkeypatch.setenv('MUSE_MODE','offline')
    monkeypatch.setenv('FLINT_REQUIRE_QUALIFIED','1')
    with pytest.raises(RuntimeError,match='prepared launch'):muse_kernel.fast(plain)
    monkeypatch.delenv('FLINT_REQUIRE_QUALIFIED')
    wrapped=muse_kernel.fast(plain)
    wrapped.require_qualified=True;wrapped.native_loaded=True;wrapped.kernel=plain
    x=torch.ones(3);wrapped.verified=[muse_kernel.signature((x,))]
    compiled=torch.compile(wrapped,fullgraph=True,backend='aot_eager')
    torch.testing.assert_close(compiled(x),plain(x))
    with pytest.raises(RuntimeError,match='coverage'):wrapped(torch.ones(4))
    with pytest.raises(Exception):compiled(torch.ones(4))
    torch.testing.assert_close(wrapped(x=x),plain(x))
    with pytest.raises(RuntimeError,match='coverage'):wrapped(x=torch.ones(4))


def test_managed_capture_collects_whole_rehearsal_metadata(monkeypatch,tmp_path):
    monkeypatch.setenv('MUSE_KERNEL_CACHE',str(tmp_path));monkeypatch.setenv('MUSE_MODE','capture')
    monkeypatch.setenv('FLINT_EXTERNAL_BUILDER','1')
    wrapped=muse_kernel.fast(plain)
    for n in (3,3,5,7,3):wrapped(torch.ones(n))
    assert len(wrapped.seen)==3 and not wrapped.job.exists()
    wrapped.finish_capture()
    job=json.loads((wrapped.job/'job.json').read_text())
    assert {s[0]['shape'][0] for s in job['verified_signatures']}=={3,5,7}
    assert len(json.loads((wrapped.job/'extra_specs.json').read_text()))==2
    assert all(s['args'][0].device.type=='meta' for s in wrapped.seen)


def test_new_rehearsal_coverage_revokes_old_checks_before_reuse(monkeypatch,tmp_path):
    monkeypatch.setenv('MUSE_KERNEL_CACHE',str(tmp_path));monkeypatch.setenv('MUSE_MODE','capture')
    monkeypatch.setenv('FLINT_EXTERNAL_BUILDER','1')
    wrapped=muse_kernel.fast(plain);ready_artifact(wrapped)
    wrapped(torch.ones(5));wrapped.finish_capture()
    state=json.loads((wrapped.job/'status.json').read_text())
    assert state['state']=='needs_revalidation' and not state['holdout_passed'] and not state['compile_passed']
    assert (wrapped.job/'kernel.py').exists()
    assert list((wrapped.job/'coverage-history').glob('*/job.json'))


def test_native_partial_mask_gradient_rule_keeps_tolerance_and_oracle_bound():
    from flint_gpu.native_check import compare_gradients
    truth=torch.tensor([0.,1.],dtype=torch.float64)
    native=torch.tensor([1e-3,1.])
    better=torch.tensor([1e-5,1.])
    result=compare_gradients([better],[native],[truth])
    assert result[0]['used_reference_bound'] and result[0]['error']<result[0]['limit']
    with pytest.raises(AssertionError):compare_gradients([torch.tensor([.1,1.])],[native],[truth])
    with pytest.raises(AssertionError):compare_gradients([better],[torch.tensor([float('nan'),1.])],[truth])
    with pytest.raises(AssertionError):compare_gradients([better],[native])
