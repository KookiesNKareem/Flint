"""Managed, fail-closed mid-run admission; controller owns credentials and spending.

The application pauses only at an explicit completed work boundary. Qualification runs in
separate processes while it is paused, avoiding contention with its own GPU execution.
"""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from .observation import Observation
from .units import subscribe
from .optimize_worker import atomic

MARKER='FLINT_LIVE_REQUEST '

class ResumeBlocked(BaseException):
    pass


def activate(runtimes):
    """Check the entire promotion before publishing any callable; retain old autograd graphs."""
    from .qualification import activation
    receipt=json.loads(Path('.flint/qualification.json').read_text())
    if not receipt.get('require_unit_gain'):
        return {'activated':False,'reason':'live activation requires measured steady-state work-unit improvement'}
    candidates=[]
    for runtime in runtimes:
        if runtime.disabled or runtime.pin_missing or runtime.pin is not None:
            return {'activated':False,'reason':'disabled or pinned implementation'}
        try:
            state=json.loads((runtime.job/'status.json').read_text())
            from kernel_contract.cases import FUZZ_POLICY
            if state.get('fuzz_policy')!=FUZZ_POLICY:return {'activated':False,'reason':'current fuzz policy has not passed'}
            allowed,reason=activation(runtime.job,state,runtime.config,runtime.environment)
            if not allowed:return {'activated':False,'reason':reason}
            candidates.append(runtime)
        except (OSError,ValueError,KeyError) as error:return {'activated':False,'reason':str(error)}
    # Runtime state changes on the workload thread only. Operator modules are immutable;
    # already constructed forward/backward graphs retain their original callables.
    old=[(r,r.kernel,r.verified,r.native_loaded,r.performance_qualified,r.activation_reason,r.loaded_sha) for r in candidates]
    try:
        for runtime in candidates:
            import torch
            runtime._startup_activation=True
            try:
                from .device import preserve_rng
                with preserve_rng():runtime._load_if_ready()
            finally:runtime._startup_activation=False
            if not runtime.native_loaded or not runtime.performance_qualified:
                raise RuntimeError(runtime.activation_reason or 'candidate could not load')
    except Exception as error:
        for r,k,v,n,p,a,h in old:r.kernel,r.verified,r.native_loaded,r.performance_qualified,r.activation_reason,r.loaded_sha=k,v,n,p,a,h
        return {'activated':False,'reason':str(error)}
    return {'activated':bool(candidates),'artifacts':{r.job.name:r.status()['kernel_sha256'] for r in candidates}}


class Live:
    def __init__(self, folder, owner, config, exchange=None):
        self.folder=Path(folder);self.owner=owner;self.config=config;self.exchange=exchange or self.request
        self.observation=Observation(max_units=config['observation_max_units'],max_seconds=config['observation_seconds'])
        self.last=time.monotonic();self.remove=None;self.finished=False;self.events=[]
        self.coverage_counts=None;self.coverage_changed=False
        self.thread=threading.get_ident();self.kind=config['workload_kind'];self.phase='observing'

    def start(self):
        self.remove=subscribe(self.boundary,self.kind);return self

    def stop(self):
        if self.remove:self.remove();self.remove=None
        self.folder.mkdir(parents=True,exist_ok=True)
        atomic(self.folder/'live.json',self.report())

    def report(self):return {'phase':self.phase,'observation':self.observation.report(),'events':self.events}

    def boundary(self,kind,seconds=None):
        if threading.get_ident()!=self.thread:return
        if self.finished:
            if self.phase in ('active','additional_coverage_observing'):
                from .compilation import REGISTRY
                counts=tuple((r.registry_key,len(r.seen)) for r in REGISTRY.values())
                if self.coverage_counts is not None and counts!=self.coverage_counts:
                    self.coverage_changed=True;self.phase='additional_coverage_observing'
                    self.observation=Observation(max_units=self.config['observation_max_units'],max_seconds=self.config['observation_seconds'])
                    self.finished=False
                self.coverage_counts=counts
            if self.finished:return
        from .compilation import REGISTRY
        runtimes=[r for r in REGISTRY.values() if r.seen and not r.disabled]
        now=time.monotonic();elapsed=now-self.last if seconds is None else seconds;self.last=now
        signatures=[r.hash+hashlib.sha256(json.dumps(s['signature'],sort_keys=True).encode()).hexdigest()
                    for r in runtimes for s in r.seen]
        state=self.observation.add(elapsed,signatures)
        if state=='observing':return
        self.finished=True
        if state!='sufficient':self.phase='inconclusive';return
        if self.coverage_changed:
            self.phase='additional_coverage_requires_qualification';return
        if not self.config['replay_safe']:
            self.phase='replay_contract_required';return
        if int(os.environ.get('WORLD_SIZE','1'))!=1:
            self.phase='distributed_activation_unsupported';return
        if not runtimes or any(r.pin is not None or r.pin_missing for r in runtimes):
            self.phase='pinned';return
        if all(r.performance_qualified for r in runtimes):self.phase='already_qualified';return
        import torch
        from .device import available,synchronize
        if available()=='cpu':self.phase='gpu_required';return
        self.phase='qualifying';start=time.monotonic()
        try:
            synchronize()
            result=self.exchange({'observation':self.observation.report(),'kind':kind})
            if result.get('accepted'):
                result={**result,**activate(runtimes)}
                self.phase='active' if result['activated'] else 'reference_retained'
            else:self.phase='reference_retained'
            self.coverage_counts=tuple((r.registry_key,len(r.seen)) for r in REGISTRY.values())
            self.events.append({**result,'boundary':self.observation.units,'pause_seconds':time.monotonic()-start})
        except Exception as error:
            self.phase='reference_retained';self.events.append({'error':type(error).__name__+': '+str(error)})

    def request(self,evidence):
        from .leases import release,acquire
        self.folder.mkdir(parents=True,exist_ok=True)
        response=self.folder/'live-response.json';response.unlink(missing_ok=True)
        atomic(self.folder/'live-request.json',evidence)
        release(self.owner)
        try:
            print(MARKER+json.dumps({'owner':self.owner}),flush=True)
            deadline=time.monotonic()+self.config['optimization_minutes']*60+60
            while not response.exists():
                if time.monotonic()>deadline:raise TimeoutError('Live controller did not return before deadline')
                time.sleep(.1)
            return json.loads(response.read_text())
        finally:
            # Never resume GPU work while an abandoned optimizer still owns the worker.
            try:acquire(self.owner,'resumed workload')
            except Exception as error:raise ResumeBlocked('Worker still owns GPU; refusing unsafe resume: '+str(error)) from error
