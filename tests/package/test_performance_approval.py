import copy
import json
from pathlib import Path
import pytest
from flint_gpu.approval import approve,inspect_receipt,evidence_digest
from flint_gpu.qualification import SCHEMA,activation,decision,sha,snapshot


@pytest.fixture
def measured(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path);monkeypatch.setattr('sys.argv',['train.py'])
    monkeypatch.setenv('FLINT_CONFIG',str(tmp_path.parent/(tmp_path.name+'-personal.toml')))
    cache=tmp_path/'.flint/cache';job=cache/'job';job.mkdir(parents=True)
    monkeypatch.setenv('FLINT_CACHE',str(cache))
    (tmp_path/'train.py').write_text('pass\n');(job/'kernel.py').write_text('pass\n')
    digest=sha(job/'kernel.py');env={'gpu':'test'}
    monkeypatch.setattr('flint_gpu.identity.environment',lambda:env)
    status={'state':'ready','kernel':'kernel.py','kernel_sha256':digest,'compile_passed':True,'holdout_passed':True}
    (job/'status.json').write_text(json.dumps(status))
    (job/'job.json').write_text(json.dumps({'environment':env,'native':True,'require_holdout':True}))
    pairs=[{'reference':{'success':True,'seconds':10+i*.1,'peak_allocated_bytes':1000000},
            'benchmark':{'success':True,'seconds':2.5+i*.01,'peak_allocated_bytes':1005000,'artifacts':{'job':digest}}} for i in range(3)]
    receipt={'schema':SCHEMA,'accepted':False,'reason':'candidate increases peak allocated CUDA memory',
             'minimum_gain':.05,'require_unit_gain':False,'performance_policy':{'max_memory_growth':0.0},
             'pairs':pairs,'environment':env,'preparation':copy.deepcopy(pairs[0]),
             'cache_policy':'one_complete_command_per_mode_before_fixed_pairs_v1',
             'workload':snapshot(tmp_path,['train.py'])}
    path=tmp_path/'.flint/qualification.json';path.write_text(json.dumps(receipt))
    return path,receipt,job,status,{'cache_dir':str(cache)},env


def test_explicit_approval_preserves_memory_tradeoff_and_guards(measured):
    path,r,job,status,config,env=measured
    view=inspect_receipt(r);assert view['approvable'] and view['extra_bytes']==5000
    assert not activation(job,status,config,env)[0]
    approved=approve(view['evidence_sha256']);assert approved['accepted']
    assert activation(job,status,config,env)[0]
    saved=json.loads(path.read_text());assert saved['pairs']==r['pairs']
    assert saved['approval']['max_peak_allocated_bytes']==1005000
    saved['pairs'][0]['benchmark']['seconds']=1;path.write_text(json.dumps(saved))
    assert not activation(job,status,config,env)[0]


@pytest.mark.parametrize('change',['source','kernel','holdout','compile','timing','digest'])
def test_approval_cannot_override_missing_or_stale_evidence(measured,change):
    path,r,job,status,config,env=measured
    digest=evidence_digest(r)
    if change=='source':Path('train.py').write_text('changed\n')
    elif change=='kernel':(job/'kernel.py').write_text('changed\n')
    elif change in ('holdout','compile'):
        status[change+'_passed']=False;(job/'status.json').write_text(json.dumps(status))
    elif change=='timing':
        r['pairs'][0]['benchmark']['seconds']=20;path.write_text(json.dumps(r));digest=evidence_digest(r)
    else:digest='stale'
    with pytest.raises(ValueError):approve(digest)
    assert not json.loads(path.read_text())['accepted']


def test_saved_memory_allowance_is_bounded_and_speed_is_still_mandatory(measured):
    _,r,*_=measured
    assert not decision(r['pairs'])[0]
    assert decision(r['pairs'],memory_growth=.01)[0]
    assert not decision(r['pairs'],memory_growth=.001)[0]
    r['pairs'][0]['benchmark']['seconds']=10
    assert not decision(r['pairs'],memory_growth=.01)[0]
    assert not decision(r['pairs'],memory_growth=float('inf'))[0]


def test_remember_only_saves_the_reviewed_allowance(measured):
    from flint_gpu.config import read_config
    _,r,*_=measured
    approve(evidence_digest(r),remember=True)
    assert read_config()['max_memory_growth']==pytest.approx(.005)
    assert read_config()['review_memory_tradeoffs'] is True


def test_owned_worker_approval_uses_real_evidence_checks(measured,monkeypatch,tmp_path):
    from flint_gpu.optimize_worker import handle
    path,r,job,status,config,env=measured
    monkeypatch.setenv('FLINT_WORKER_DIR',str(tmp_path.parent/(tmp_path.name+'-worker')))
    request={'project':str(tmp_path),'session':'terminaltest'}
    handle({**request,'action':'begin'})
    try:
        # Qualification writes the measured receipt after begin invalidates the old one.
        path.write_text(json.dumps(r))
        result=handle({**request,'action':'approve','evidence_sha256':evidence_digest(r)})
        assert result['accepted'] and activation(job,status,config,env)[0]
    finally:handle({**request,'action':'finish','summary':{}})


@pytest.mark.parametrize('answer,accepted',[('\n',True),('n\n',False),('',False)])
def test_terminal_review_writes_approval_only_after_yes(measured,monkeypatch,answer,accepted):
    import io,sys
    from flint_gpu.approval import review_on_terminal
    class TTY(io.StringIO):
        def isatty(self):return True
    path,r,*_=measured
    monkeypatch.setattr(sys,'stdin',TTY(answer));monkeypatch.setattr(sys,'stderr',TTY())
    monkeypatch.setenv('TERM','xterm');monkeypatch.setenv('FLINT_INTERACTIVE','1')
    review_on_terminal()
    assert json.loads(path.read_text())['accepted'] is accepted
