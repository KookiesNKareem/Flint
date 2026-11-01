import copy
import json
from pathlib import Path
import pytest
from flint_gpu.optimization import diagnose, run_loop
from flint_gpu.optimize_worker import handle


def receipt(kind='memory'):
    result={'accepted':kind=='accepted','minimum_gain':.05,'reason':'candidate increases peak allocated CUDA memory',
            'pairs':[{'reference':{'success':True,'seconds':10,'peak_allocated_bytes':100},
                      'benchmark':{'success':True,'seconds':7,'peak_allocated_bytes':110,'artifacts':{'job':'abc'}}} for _ in range(3)]}
    if kind=='accepted':
        for p in result['pairs']:p['benchmark']['peak_allocated_bytes']=90
    if kind=='speed':
        for p in result['pairs']:p['benchmark'].update(seconds=11,peak_allocated_bytes=90)
    return result


def test_diagnosis_separates_actionable_evidence_from_baseline_and_dispatch_failures():
    d=diagnose(receipt());assert d['repairable'] and d['failures']==['memory_regression']
    assert diagnose(receipt('speed'))['failures']==['latency_or_noise']
    r=receipt();r['pairs'][0]['reference']['success']=False
    assert not diagnose(r)['repairable'] and diagnose(r)['kind']=='baseline_failed'
    r=receipt();r['pairs'][0]['benchmark']['artifacts']={}
    assert not diagnose(r)['repairable']
    r=receipt();r['pairs'][0]['benchmark'].update(success=False,exit_code=-9,failure_tail='connection reset')
    assert not diagnose(r)['repairable']
    r['pairs'][0]['benchmark']['failure_tail']='CUDA out of memory'
    assert diagnose(r)['repairable']
    r=receipt();r['reason']='workload changed during qualification'
    assert not diagnose(r)['repairable']


class FakeSession:
    def __init__(self,tmp_path,receipts):
        self.out=tmp_path;self.id='session';self.receipts=iter(receipts);self.actions=[];self.models=[];self.dollars=1
        self.checked=True;self.activated=True
    def left(self):return 100
    def budget(self):return {'usd':1-self.dollars,'reserved_usd':0}
    def remaining(self):return self.dollars
    def event(self,**event):pass
    def worker(self,action,**payload):
        self.actions.append((action,payload))
        if action=='capture':return {'training':{'success':True},'jobs':{'/cache/job':{
            'job.json':'{}','kernel.py':'pass','status.json':json.dumps({'state':'ready','compile_passed':True,'holdout_passed':True})}}}
        if action=='qualify':return next(self.receipts)
        if action=='verify':return {'success':True,'artifacts':{'job':'abc'} if self.activated else {},
                                   'dispatch':[{'job':'/cache/job','performance_qualified':self.activated}]}
        return {}
    def generate(self,job,diagnosis=None,model=None):
        self.models.append(model);self.dollars-=.3
        assert diagnosis['repairable'] and 'seconds' in diagnosis
        return self.checked


def test_loop_repairs_rechecks_and_verifies_normal_activation(tmp_path):
    s=FakeSession(tmp_path,[receipt(),receipt('speed'),receipt('accepted')])
    result=run_loop(s,['train.py'])
    assert result['accepted'] and len(result['rounds'])==3
    assert s.models==['meta-models/muse-spark-open-1026','muse-wonder-1.0']
    assert [x for x,_ in s.actions].count('install')==3
    assert s.actions[-1][0]=='finish'
    assert (tmp_path/'summary.json').exists()


def test_failed_checks_do_not_install_or_qualify_new_candidate(tmp_path):
    s=FakeSession(tmp_path,[receipt()]);s.checked=False
    result=run_loop(s,['train.py'])
    assert not result['accepted'] and result['reason']=='repair_failed_checks_or_budget'
    assert [x for x,_ in s.actions].count('install')==1
    assert [x for x,_ in s.actions].count('qualify')==1


def test_budget_exhaustion_and_round_limit_keep_default(tmp_path):
    s=FakeSession(tmp_path,[receipt()]);s.dollars=0
    result=run_loop(s,['train.py'])
    assert result['reason']=='shared_budget_exhausted' and not s.models
    s=FakeSession(tmp_path/'other',[receipt()]);s.out.mkdir()
    result=run_loop(s,['train.py'],max_rounds=0)
    assert result['reason']=='repair_round_limit' and not s.models


def test_missing_normal_activation_revokes_approval(tmp_path):
    s=FakeSession(tmp_path,[receipt('accepted')]);s.activated=False
    result=run_loop(s,['train.py'])
    assert not result['accepted'] and any(a=='revoke' for a,_ in s.actions)


def test_interruption_after_qualification_revokes_approval(tmp_path):
    s=FakeSession(tmp_path,[receipt('accepted')]);worker=s.worker
    def interrupted(action,**payload):
        if action=='verify':raise KeyboardInterrupt()
        return worker(action,**payload)
    s.worker=interrupted
    result=run_loop(s,['train.py'])
    assert not result['accepted'] and any(a=='revoke' for a,_ in s.actions)
    assert s.actions[-1][0]=='finish'


def test_identical_repair_is_not_rebenchmarked_until_noise_passes(tmp_path):
    s=FakeSession(tmp_path,[receipt()]);worker=s.worker
    def with_digest(action,**payload):
        result=worker(action,**payload)
        if action=='capture':
            files=result['jobs']['/cache/job'];state=json.loads(files['status.json'])
            state['kernel_sha256']='same';files['status.json']=json.dumps(state)
        return result
    s.worker=with_digest
    result=run_loop(s,['train.py'])
    assert result['reason']=='no_implementation_change'
    assert [a for a,_ in s.actions].count('qualify')==1


def test_worker_lease_rejects_other_session_and_releases_owner(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('FLINT_WORKER_DIR',str(tmp_path/'worker'))
    base={'project':str(tmp_path),'session':'first'}
    handle({**base,'action':'begin'})
    with pytest.raises(FileExistsError):handle({**base,'session':'second','action':'begin'})
    with pytest.raises(RuntimeError):handle({**base,'session':'second','action':'revoke','reason':'test'})
    handle({**base,'action':'finish','summary':{}})
    assert not (tmp_path/'.flint/optimization-owner.json').exists()


def test_shared_provider_budget_reservations_survive_restarting_session(tmp_path):
    from utils.providers.budget import reserve,settle,BudgetExceeded,snapshot
    ledger=tmp_path/'budget.json'
    first=reserve(ledger,.7,1);settle(ledger,first,.6)
    second=reserve(ledger,.3,1)
    with pytest.raises(BudgetExceeded):reserve(ledger,.2,1)
    assert snapshot(ledger)=={'usd':.6,'reserved_usd':.3}
    settle(ledger,second,None)
    with pytest.raises(BudgetExceeded):reserve(ledger,.2,1)


def test_preparation_failure_diagnosis_does_not_spend_on_broken_reference():
    r=receipt();r['preparation']={'reference':{'success':False}}
    assert diagnose(r)['kind']=='baseline_failed' and not diagnose(r)['repairable']
    r['preparation']={'reference':{'success':True},'benchmark':{
        'success':False,'artifacts':{'job':'digest'},'failure_tail':'CUDA out of memory'}}
    assert diagnose(r)['kind']=='candidate_memory_failure' and diagnose(r)['repairable']


def test_failed_repair_process_cannot_reuse_old_ready_status(monkeypatch,tmp_path):
    import flint_gpu.optimization as module
    job=tmp_path/'job';job.mkdir()
    (job/'job.json').write_text('{}')
    (job/'status.json').write_text(json.dumps({'state':'ready','compile_passed':True,'holdout_passed':True}))
    session=module.Session.__new__(module.Session)
    session.env={};session.remaining=lambda:1;session.left=lambda:30
    session.event=lambda **event:None;session.budget=lambda:{'usd':0,'reserved_usd':0}
    class Failed:
        def __init__(self,*args,**kwargs):pass
        def wait(self,**kwargs):return 1
    monkeypatch.setattr(module.subprocess,'Popen',Failed)
    assert not session.generate(job,{'kind':'performance'},'model')
    assert json.loads((job/'status.json').read_text())['state']=='failed'


def test_memory_tradeoff_waits_for_review_without_spending_on_repairs(tmp_path):
    s=FakeSession(tmp_path,[receipt()]);s.review_memory_tradeoffs=True
    result=run_loop(s,['train.py'])
    assert result['reason']=='performance_tradeoff_needs_review'
    assert not s.models
    assert result['rounds'][0]['diagnosis']['approval_available']


@pytest.mark.parametrize('choice',[True,False,None])
def test_inline_tradeoff_review_controls_activation_without_regeneration(tmp_path,choice):
    s=FakeSession(tmp_path,[receipt()]);s.review_memory_tradeoffs=True
    s.review_tradeoff=lambda r:choice
    original=s.worker
    def worker(action,**kwargs):
        if action=='approve':
            s.actions.append((action,kwargs));return {'accepted':True,'approval':{'kind':'measured_memory_tradeoff'}}
        return original(action,**kwargs)
    s.worker=worker
    result=run_loop(s,['train.py'])
    assert not s.models
    assert result['accepted']==(choice is True)
    actions=[a for a,_ in s.actions]
    assert ('approve' in actions)==(choice is True)
    assert ('verify' in actions)==(choice is True)
    if choice is not True:assert 'performance_tradeoff_' in result['reason']


def test_inline_approval_is_revoked_if_normal_dispatch_fails(tmp_path):
    s=FakeSession(tmp_path,[receipt()]);s.review_memory_tradeoffs=True;s.activated=False
    s.review_tradeoff=lambda r:True;original=s.worker
    s.worker=lambda action,**kw: {'accepted':True} if action=='approve' else original(action,**kw)
    result=run_loop(s,['train.py'])
    assert not result['accepted'] and any(a=='revoke' for a,_ in s.actions)
