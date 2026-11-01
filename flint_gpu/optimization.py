"""Bounded capture → generate → qualify → diagnose → repair → requalify orchestration."""
from pathlib import Path
import contextlib
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
from .optimize_worker import ARTIFACT_FILES, atomic


def diagnose(receipt):
    """Classify observed failures; never send baseline/infrastructure faults to a kernel writer."""
    pairs=receipt.get('pairs',[])
    if receipt.get('accepted'):return {'kind':'accepted','repairable':False}
    if 'changed' in receipt.get('reason',''):
        return {'kind':'workload_or_artifact_changed','repairable':False}
    prep=receipt.get('preparation')
    if prep:
        if not prep.get('reference',{}).get('success'):
            return {'kind':'baseline_failed','repairable':False,'reason':'reference preparation failed'}
        candidate=prep.get('benchmark',{})
        if not candidate.get('success'):
            tail=candidate.get('failure_tail','') or ''
            oom='out of memory' in tail.lower()
            return {'kind':'candidate_memory_failure' if oom else 'candidate_execution_failed',
                    'repairable':bool(candidate.get('artifacts')) and oom,'failure_tails':[tail]}
        if not candidate.get('artifacts'):
            return {'kind':'candidate_not_dispatched','repairable':False}
    if not pairs or any(not p.get('reference',{}).get('success') for p in pairs):
        return {'kind':'baseline_failed','repairable':False,'reason':receipt.get('reason')}
    failures=[p.get('benchmark',{}) for p in pairs if not p.get('benchmark',{}).get('success')]
    if failures:
        tails=[r.get('failure_tail','') or '' for r in failures]
        # Missing reports/transport timeouts are not evidence of a kernel defect.
        observed=any(r.get('artifacts') for r in failures)
        oom=any('out of memory' in t.lower() for t in tails)
        return {'kind':'candidate_memory_failure' if oom else 'candidate_execution_failed',
                'repairable':observed and oom,'failure_tails':tails}
    groups=[p['benchmark'].get('artifacts',{}) for p in pairs]
    if not all(groups):return {'kind':'candidate_not_dispatched','repairable':False}
    if any(g!=groups[0] for g in groups):return {'kind':'artifact_set_changed','repairable':False}
    if len(pairs)<3:return {'kind':'insufficient_measurements','repairable':False}
    if any(not isinstance(p[m].get('peak_allocated_bytes'),int) or p[m]['peak_allocated_bytes']<=0
           for p in pairs for m in ('reference','benchmark')):
        return {'kind':'missing_memory_evidence','repairable':False}
    times={m:[p[m]['seconds'] for p in pairs] for m in ('reference','benchmark')}
    if any(not math.isfinite(t) or t<=0 for v in times.values() for t in v):
        return {'kind':'invalid_measurements','repairable':False}
    memory={m:[p[m]['peak_allocated_bytes'] for p in pairs] for m in times}
    speed=max(times['benchmark'])>=min(times['reference'])*(1-receipt['minimum_gain'])
    mem=max(memory['benchmark'])>min(memory['reference'])*(1+receipt.get('performance_policy',{}).get('max_memory_growth',0.0))
    failures=(['latency_or_noise'] if speed else [])+(['memory_regression'] if mem else [])
    units={}
    if receipt.get('require_unit_gain'):
        units={m:[(p[m].get('work_units') or {}).get('median_seconds') for p in pairs] for m in times}
        if any(not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for values in units.values() for x in values):
            return {'kind':'missing_work_unit_evidence','repairable':False}
        if max(units['benchmark'])>=min(units['reference'])*(1-receipt['minimum_gain']):failures.append('steady_state_latency_or_noise')
    if receipt.get('reason')=='MPS peak memory is sampled; review required':
        return {'kind':'performance','repairable':False,'approval_available':True,'failures':['memory_measurement_incomplete']}
    return {'kind':'performance','repairable':bool(failures),'approval_available':failures==['memory_regression'],'failures':failures,'seconds':times,
            'peak_allocated_bytes':memory,'work_unit_medians':units,'minimum_gain':receipt['minimum_gain'],
            'artifacts':groups[0],'scope':receipt.get('scope','full_command'),'rehearsal_steps':receipt.get('rehearsal_steps'),
            'note':'Complete subprocess measurements for the recorded scope; attribution to one operator is a hypothesis.'}


class Session:
    def __init__(self, project, remote, out, ledger, max_usd, minutes, worker_dir):
        self.project=project;self.remote=remote;self.out=Path(out).resolve();self.out.mkdir(parents=True,exist_ok=False)
        self.id=uuid.uuid4().hex;self.deadline=time.monotonic()+minutes*60
        self.ledger=Path(ledger).resolve();self.cap=max_usd;self.worker_dir=worker_dir
        from .config import read_config
        self.review_memory_tradeoffs=read_config()['review_memory_tradeoffs']
        self.performance_policy={'max_memory_growth':read_config()['max_memory_growth']}
        self.env={**os.environ,'MUSE_EXPERIMENT_LEDGER':str(self.ledger),'MUSE_EXPERIMENT_BUDGET_USD':str(max_usd),
                  'FLINT_REMOTE':remote,'MUSE_KERNEL_REMOTE':remote,'FLINT_WORKER_DIR':worker_dir,
                  'MUSE_KERNEL_REMOTE_DIR':worker_dir}

    def left(self):
        left=int(self.deadline-time.monotonic())
        if left<1:raise TimeoutError('Optimization wall-time limit reached')
        return left

    def budget(self):
        from utils.providers.budget import snapshot
        return snapshot(self.ledger)

    def remaining(self):
        b=self.budget();return max(0,self.cap-b['usd']-b['reserved_usd'])

    def event(self, **value):
        value['time']=time.time()
        with (self.out/'events.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
        event=value.get('event')
        if event=='capture_started': message='Observing your workload…'
        elif event=='generation_started': message='Generating forward and backward kernels…'
        elif event=='repair_started': message=f"Repairing with {value.get('model')}…"
        elif event=='candidate_checked':
            message='Checks passed.' if value.get('state')=='ready' else 'Checks rejected the candidate: '+str(value.get('reason'))
        elif event=='qualification_started': message='Comparing with your normal command…'
        elif event=='qualification':
            diagnosis=value.get('diagnosis',{})
            if diagnosis.get('approval_available') and self.review_memory_tradeoffs:
                from .terminal import interactive
                if not interactive():print('Faster, with a memory tradeoff. Review with flint approve; no further model spending.',flush=True)
                return
            message=('Qualified. Verifying normal dispatch…' if diagnosis.get('kind')=='accepted' else
                     'Kept PyTorch: '+', '.join(diagnosis.get('failures') or [diagnosis.get('kind','qualification failed')]).replace('_',' ')+'.')
        else: return
        from .terminal import terminal_output
        if not terminal_output() or (event=='candidate_checked' and value.get('state')!='ready'):
            print(message,flush=True)

    def review_tradeoff(self, receipt):
        from .approval import inspect_receipt
        from .terminal import confirm_tradeoff
        return confirm_tradeoff(inspect_receipt(receipt))

    def worker(self, action, **payload):
        request={'project':self.project,'session':self.id,'action':action,'performance_policy':self.performance_policy,**payload}
        # The outer deadline also protects a dropped controller connection. The worker catches
        # TERM and kills its separately grouped training subprocess before exiting.
        limit=self.left() if action not in ('finish','revoke') else 20
        python=shlex.quote(sys.executable) if self.remote=='bash -c' else 'python'
        command=f'{python} -m flint_gpu.deadline {limit} {python} -m flint_gpu.optimize_worker'
        from .terminal import Progress
        label={'capture':'Observing','qualify':'Comparing training runs','verify':'Verifying activation','approve':'Approving'}.get(action)
        with Progress(label) if label else contextlib.nullcontext():
            proc=subprocess.run(shlex.split(self.remote)+[command],input=json.dumps(request),text=True,
                                capture_output=True,timeout=limit+15)
        (self.out/(action+'-latest.log')).write_text(proc.stdout+proc.stderr)
        line=next((x for x in reversed(proc.stdout.splitlines()) if x.startswith('FLINT_RESULT ')),None)
        if not line:raise RuntimeError('GPU worker returned no receipt: '+(proc.stderr or proc.stdout)[-1000:])
        result=json.loads(line[13:])
        if result.get('worker_error'):raise RuntimeError(result['worker_error'])
        return result

    def generate(self, job, diagnosis=None, model=None):
        remaining=self.remaining()
        if remaining<=0:raise RuntimeError('Shared model budget exhausted')
        self.event(event='repair_started' if diagnosis else 'generation_started',artifact=job.name,
                   model=model,remaining_model_usd=remaining)
        meta=json.loads((job/'job.json').read_text());meta.update(max_usd=remaining,total_budget=True,max_tokens=24000)
        if getattr(self, 'only_model', None):
            meta.update(models=[self.only_model], backward_model='')
            if model is not None and model != self.only_model:
                raise ValueError('Repair model violates the single-model policy')
            atomic(job/'provider-policy.json', {'only_model':self.only_model,'scope':'new_generation_and_repairs'})
        (job/'job.json').write_text(json.dumps(meta,indent=2))
        if diagnosis is None:
            cmd=[sys.executable,'-m','flint_gpu.build',str(job)]
        else:
            path=job/'diagnosis.json';atomic(path,diagnosis)
            cmd=[sys.executable,'-m','flint_gpu.repair',str(job),str(path),'--model',model,'--max-usd',str(remaining)]
        from .terminal import Progress
        with Progress('Repairing and checking' if diagnosis else 'Generating and checking'), (job/'controller.log').open('w') as log:
            from .setup import provider_environment
            child=subprocess.Popen(cmd,env={**provider_environment(self.env),'MUSE_KERNEL_REMOTE_TIMEOUT':str(self.left())},
                                   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:returncode=child.wait(timeout=self.left())
            except BaseException:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
        state=json.loads((job/'status.json').read_text())
        if returncode:
            state.update(state='failed',reason=f'Candidate worker exited {returncode}; see controller.log')
            atomic(job/'status.json',state)
        self.event(event='candidate_checked',model=model,state=state.get('state'),reason=state.get('reason'),budget=self.budget())
        return state.get('state')=='ready' and state.get('compile_passed') and state.get('holdout_passed')


def run_loop(session, command, repeats=3, timeout=300, minimum_gain=.05, inputs=(), max_rounds=2,
             repair_models=('meta-models/muse-spark-open-1026','muse-wonder-1.0'),rehearsal_steps=None,require_unit_gain=False):
    """All model attempts and repairs share one persistent ledger; failed admission stays reference."""
    summary={'session':session.id,'accepted':False,'rounds':[],'command':command}
    owned=False
    try:
        session.worker('begin');owned=True
        session.event(event='capture_started',command=command)
        capture=session.worker('capture',command=command,timeout=min(timeout,session.left()),rehearsal_steps=rehearsal_steps)
        atomic(session.out/'capture.json',capture)
        if not capture['training'].get('success'):
            summary['reason']='baseline_capture_failed';return summary
        jobs={}
        for remote,files in capture['jobs'].items():
            job=session.out/'initial'/Path(remote).name;job.mkdir(parents=True)
            for name,content in files.items():
                if name not in ARTIFACT_FILES:raise ValueError('Unexpected captured artifact file')
                (job/name).write_text(content)
            state=json.loads(files['status.json'])
            if not (state.get('state')=='ready' and state.get('compile_passed') and state.get('holdout_passed')):
                if not session.generate(job):
                    summary['reason']='generation_failed_or_budget_exhausted';return summary
            session.worker('install',job=remote,round='initial',files=artifact_files(job))
            jobs[Path(remote).name]=(remote,job)
        if not jobs:
            failures=[{'name':d.get('name'),'errors':[k for k in d.get('calls',{}) if k.startswith('disabled:')]}
                      for d in capture['training'].get('dispatch',[]) if any(k.startswith('disabled:') for k in d.get('calls',{}))]
            summary.update(reason='runtime_capture_failed' if failures else 'no_supported_decorated_function_captured',
                           capture_failures=failures)
            return summary
        for round_index in range(max_rounds+1):
            session.event(event='qualification_started',round=round_index)
            receipt=session.worker('qualify',command=command,repeats=repeats,timeout=min(timeout,session.left()),
                                   minimum_gain=minimum_gain,inputs=list(inputs),round=round_index,rehearsal_steps=rehearsal_steps,require_unit_gain=require_unit_gain)
            atomic(session.out/f'qualification-{round_index}.json',receipt)
            diagnosis=diagnose(receipt)
            summary['rounds'].append({'round':round_index,'diagnosis':diagnosis,'accepted':receipt['accepted']})
            session.event(event='qualification',round=round_index,diagnosis=diagnosis)
            if not receipt['accepted'] and diagnosis.get('approval_available') and getattr(session,'review_memory_tradeoffs',False):
                choice=session.review_tradeoff(receipt) if hasattr(session,'review_tradeoff') else None
                if choice is not True:
                    summary['reason']='performance_tradeoff_declined' if choice is False else 'performance_tradeoff_needs_review'
                    return summary
                from .approval import evidence_digest
                approval=session.worker('approve',evidence_sha256=evidence_digest(receipt))
                if not approval.get('accepted'):raise RuntimeError('Tradeoff approval was not applied')
                atomic(session.out/f'approval-{round_index}.json',approval)
                summary['approval']=approval
            if receipt['accepted'] or summary.get('approval',{}).get('accepted'):
                normal=session.worker('verify',command=command,timeout=min(timeout,session.left()),rehearsal_steps=rehearsal_steps)
                atomic(session.out/'normal.json',normal)
                expected=receipt['pairs'][0]['benchmark']['artifacts']
                valid=normal.get('success') and normal.get('artifacts')==expected
                valid=valid and all(d.get('performance_qualified') for d in normal.get('dispatch',[])
                                    if Path(d['job']).name in expected)
                summary.update(accepted=bool(valid),reason='qualified_and_normal_run_verified' if valid else 'normal_run_did_not_activate')
                if not valid:session.worker('revoke',reason=summary['reason'])
                return summary
            if not diagnosis['repairable']:
                summary['reason']=diagnosis['kind'];return summary
            if round_index>=max_rounds:
                summary['reason']='repair_round_limit';return summary
            if session.remaining()<=0:
                summary['reason']='shared_budget_exhausted';return summary
            # Repair one observed artifact per round; the subsequent test measures the whole joint set.
            names=sorted(diagnosis.get('artifacts',{})) or sorted(jobs)
            name=names[round_index%len(names)]
            if name not in jobs:summary['reason']='unrecognized_dispatched_artifact';return summary
            remote,prior=jobs[name];job=session.out/f'repair-{round_index+1}'/name
            job.mkdir(parents=True)
            for n,c in artifact_files(prior).items():(job/n).write_text(c)
            model=repair_models[round_index%len(repair_models)]
            feedback={**diagnosis,'operator_memory':json.loads((job/'status.json').read_text()).get('memory')}
            if not session.generate(job,feedback,model):
                # Holdout is never supplied as repair feedback. Stop after any failed finalization.
                summary['reason']='repair_failed_checks_or_budget';return summary
            old=json.loads((prior/'status.json').read_text()).get('kernel_sha256')
            new=json.loads((job/'status.json').read_text()).get('kernel_sha256')
            if old is not None and new==old:
                summary['reason']='no_implementation_change';return summary
            session.worker('install',job=remote,round=round_index+1,files=artifact_files(job))
            jobs[name]=(remote,job)
    except (Exception,SystemExit,KeyboardInterrupt) as error:
        summary['reason']=type(error).__name__+': '+str(error)
        if owned:
            with contextlib.suppress(Exception):session.worker('revoke',reason='optimization interrupted or failed')
        return summary
    finally:
        summary['budget']=session.budget()
        atomic(session.out/'summary.json',summary)
        if owned:
            with contextlib.suppress(Exception):session.worker('finish',summary=summary)


def artifact_files(job):
    return {n:(job/n).read_text() for n in ARTIFACT_FILES if (job/n).exists()}


def optimize(command, project=None, remote=None, out=None, ledger=None, max_usd=None, minutes=None,
             repeats=3, timeout=300, minimum_gain=.05, inputs=(), max_rounds=2,rehearsal_steps=None,require_unit_gain=False,model=None):
    from .config import read_config
    config=read_config()
    max_usd=config['max_usd'] if max_usd is None else max_usd
    minutes=config['optimization_minutes'] if minutes is None else minutes
    if not command or not command[0].endswith('.py'):raise ValueError('Supply a Python script and its normal arguments')
    if not 0<max_usd<=100 or not 0<minutes<=240 or not 0<timeout or max_rounds<0 or repeats<3:
        raise ValueError('Invalid budget, deadline, timeout, repeats or attempt limit')
    if not .01<=minimum_gain<1:raise ValueError('Minimum gain must be between 1% and 100%')
    if rehearsal_steps is not None and not 1<=rehearsal_steps<=10000:raise ValueError('Use 1–10000 rehearsal optimizer steps')
    config=read_config();project=project or str(Path.cwd());remote=remote or config['remote']
    out=out or str(Path.cwd()/'.flint/optimizations'/uuid.uuid4().hex)
    ledger=str(Path(ledger or config['budget_ledger']).expanduser().resolve())
    session=Session(project,remote,out,ledger,max_usd,minutes,config['worker_dir'])
    if model is not None and (not isinstance(model,str) or not model.strip()):raise ValueError('model must be a nonempty model ID')
    session.only_model=model
    atomic(session.out/'provider-policy.json', {'only_model':model,'scope':'new_generation_and_repairs',
           'cached_artifacts_may_have_other_authors':True})
    models=[model] if model else list(dict.fromkeys(m for m in [*config['models'],config['backward_model']] if m))
    return run_loop(session,command,repeats,timeout,minimum_gain,inputs,max_rounds,models,rehearsal_steps,require_unit_gain)
