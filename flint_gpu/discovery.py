"""A bounded profiler window, source attribution, and read-only source proposals."""
import argparse
import contextlib
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import runpy
import sys
import time

from .suggestions import functions,eligibility,propose_region,digest

PREFIX='flint.source:'
LIBRARY=('aten::mm','aten::addmm','aten::bmm','aten::convolution','aten::_scaled_dot_product',
         'aten::scaled_dot_product','aten::native_layer_norm','aten::cudnn','aten::linear')
MOVEMENT=('aten::copy_','aten::_to_copy','aten::to','aten::pin_memory')


def rank(records, inventory, total_device_us, total_cpu_us, minimum_share=.10):
    gpu=total_device_us>0;denominator=total_device_us if gpu else total_cpu_us
    ranked=[]
    for key,record in records.items():
        item=inventory.get(key)
        if not item:continue
        weight=record['device_us'] if gpu else record['cpu_us']
        share=weight/max(denominator,1e-9)
        library=sum(v for k,v in record['ops'].items() if k.startswith(LIBRARY))/max(weight,1e-9)
        movement=sum(v for k,v in record['ops'].items() if k.startswith(MOVEMENT))/max(weight,1e-9)
        why=eligibility(item)
        if not gpu:why='CPU-only evidence; GPU benefit is unmeasured'
        elif item.get('tensor_inputs') is False:why='no tensor arguments observed; input pipeline or control function'
        elif share<minimum_share:why='small measured contribution'
        elif movement>.6:why='dominated by data movement; inspect the input pipeline'
        elif library>.8:why='dominated by existing library operations'
        ranked.append({'key':key,'file':item['file'],'line':item['line'],'end_line':item['end_line'],
                       'function':item['function'],'device_us':record['device_us'],'cpu_us':record['cpu_us'],
                       'share':share,'basis':'attributed GPU kernel duration' if gpu else 'CPU operator time',
                       'forward_us':record['forward_us'],'backward_us':record['backward_us'],
                       'ops':dict(record['ops'].most_common(12)),'suggest':why is None,'reason':why,
                       'predicted_speedup':None})
    return sorted(ranked,key=lambda x:(x['suggest'],x['share']),reverse=True)


def analyze(events,inventory,minimum_share):
    """Use exclusive CPU operator work; correlate backward nodes to forward sequence numbers.

    Kernel durations are summed work, not wall time or additive speedup predictions. Nested
    source scopes receive exclusive ownership, preventing parent/child double counting.
    """
    records=defaultdict(lambda:{'device_us':0.,'cpu_us':0.,'forward_us':0.,'backward_us':0.,'ops':Counter()})
    sequence={};owners={};cpu=[e for e in events if str(e.device_type).endswith('CPU')]
    for e in cpu:
        parent=e;owner=None
        while parent is not None:
            if parent.name.startswith(PREFIX):owner=parent.name[len(PREFIX):];break
            parent=parent.cpu_parent
        if owner:
            owners[e.id]=owner
            if e.sequence_nr>=0 and e.scope!=1:sequence[(e.thread,e.sequence_nr)]=owner
    total_gpu=total_cpu=0.;unattributed_gpu=0.;categories=Counter()
    has_gpu=any(e.kernels for e in cpu)
    for e in cpu:
        if e.name.startswith(PREFIX):continue
        gpu=sum(k.duration for k in e.kernels)
        cpu_us=max(0.,e.self_cpu_time_total)
        if not e.name.startswith('aten::'):cpu_us=0.
        total_gpu+=gpu;total_cpu+=cpu_us
        owner=owners.get(e.id);parent=e;backward=False
        while parent is not None:
            if parent.scope==1 and parent.sequence_nr>=0:
                backward=True
                owner=sequence.get((parent.fwd_thread,parent.sequence_nr),owner);break
            parent=parent.cpu_parent
        if owner:
            r=records[owner];r['device_us']+=gpu;r['cpu_us']+=cpu_us
            r['backward_us' if backward else 'forward_us']+=gpu
            if gpu or cpu_us:r['ops'][e.name]+=gpu if has_gpu else cpu_us
        else:unattributed_gpu+=gpu
        if 'DataLoader' in e.name:categories['data_loader_cpu_us']+=e.self_cpu_time_total
        if 'Synchronize' in e.name or 'synchronize' in e.name:categories['synchronization_cpu_us']+=e.self_cpu_time_total
        if 'Compiled' in e.name:categories['compiled_region_cpu_us']+=e.self_cpu_time_total
    # Kineto also emits CUDA user-annotation ranges. They span gaps and nested
    # kernels; counting them as kernels dramatically inflates the denominator.
    # Kernels with no CPU correlation must not disappear from the attribution denominator.
    device_total=sum(e.self_device_time_total for e in events
                     if str(e.device_type).endswith('CUDA') and not e.is_user_annotation)
    if device_total>total_gpu:unattributed_gpu+=device_total-total_gpu;total_gpu=device_total
    return {'regions':rank(records,inventory,total_gpu,total_cpu,minimum_share),
            'device_work_us':total_gpu,'cpu_operator_us':total_cpu,'unattributed_device_us':unattributed_gpu,
            'categories':dict(categories),'measurement_note':'GPU shares sum attributed kernel durations, not training wall time. Overlapping streams and profiling overhead prevent interpreting shares as promised savings.'}


class Window:
    def __init__(self,root,out,warmup=3,steps=5,minimum_share=.10,max_scopes=20000):
        self.root=Path(root).resolve();self.out=Path(out).resolve()
        self.inventory_error=None
        try:self.inventory=functions(root)
        except Exception as error:
            self.inventory={};self.inventory_error=type(error).__name__+': '+str(error)
        self.code={}
        for k,v in self.inventory.items():
            first=min([v['line'],*[d.lineno for d in v['node'].decorator_list]])
            for name in (v['file'],str(self.root/v['file'])):self.code[(name,first,v['function'])]=k
        self.warmup=warmup;self.steps=steps;self.minimum_share=minimum_share;self.max_scopes=max_scopes
        from .config import read_config
        from .observation import Observation
        cfg=read_config()
        self.adaptive=Observation(max_units=cfg['observation_max_units'],max_seconds=cfg['observation_seconds']) if cfg['launch_policy']=='adaptive' else None
        self.observed_signatures=set()
        self.boundary_time=time.monotonic()
        self.count=0;self.active=False;self.finished=False;self.scopes={};self.scope_count=0
        self.profile=None;self.previous=None;self.hook=None;self.failure=None;self.started=None

    def trace(self,frame,event,arg):
        try:self._trace(frame,event,arg)
        except Exception as error:self.fail(error)

    def fail(self,error):
        self.failure=type(error).__name__+': '+str(error)
        sys.setprofile(self.previous)
        for scope in reversed(list(self.scopes.values())):
            with contextlib.suppress(Exception):scope.__exit__(None,None,None)
        self.scopes.clear()
        if self.active:
            with contextlib.suppress(Exception):self.profile.stop()
        self.active=False;self.finished=True
        self.profile=None;self.inventory.clear();self.code.clear()

    def _trace(self,frame,event,arg):
        if event=='call':
            c=frame.f_code;key=self.code.get((c.co_filename,c.co_firstlineno,c.co_name))
            if key and self.scope_count<self.max_scopes:
                import torch
                def tensor_input(value,depth=0):
                    if isinstance(value,torch.Tensor):return True
                    if depth>=3:return False
                    values=list(value.values())[:16] if isinstance(value,dict) else value[:16] if isinstance(value,(tuple,list)) else ()
                    return any(tensor_input(x,depth+1) for x in values)
                local=frame.f_locals
                observed=any(tensor_input(local.get(n)) for n in c.co_varnames[:c.co_argcount+c.co_kwonlyargcount])
                self.inventory[key]['tensor_inputs']=self.inventory[key].get('tensor_inputs',False) or observed
                if self.adaptive and len(self.observed_signatures)<256:
                    from .compilation import metadata
                    values=[local.get(n) for n in c.co_varnames[:c.co_argcount] if tensor_input(local.get(n))]
                    self.observed_signatures.add(key+json.dumps(metadata(values),sort_keys=True))
                scope=torch.profiler.record_function(PREFIX+key);scope.__enter__()
                self.scopes[id(frame)]=scope;self.scope_count+=1
        elif event=='return':
            scope=self.scopes.pop(id(frame),None)
            if scope:scope.__exit__(None,None,None)

    def start(self):
        import torch
        self.previous=sys.getprofile()
        if self.inventory_error or int(os.environ.get('WORLD_SIZE','1'))>1:
            self.failure=self.inventory_error or 'Distributed discovery is not supported yet'
            self.finished=True;return
        if self.previous is not None:
            self.failure='An existing Python profiler is active; discovery skipped';self.finished=True;return
        activities=[torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():activities.append(torch.profiler.ProfilerActivity.CUDA)
        # No tensor retention for shape recording, no full stack collection, no memory events.
        self.profile=torch.profiler.profile(activities=activities,record_shapes=False,with_stack=False,profile_memory=False)
        self.profile.start();self.started=time.monotonic();self.active=True;sys.setprofile(self.trace)

    def stop(self,reason='window_complete'):
        if self.finished:return
        self.finished=True
        if self.active:
            sys.setprofile(self.previous)
            for scope in reversed(list(self.scopes.values())):scope.__exit__(None,None,None)
            self.scopes.clear();self.profile.stop();self.active=False
            analysis=analyze(self.profile.events(),self.inventory,self.minimum_share)
        else:analysis={'regions':[],'device_work_us':0,'reason':'training ended before the profiling window'}
        import torch
        if torch.backends.mps.is_available() and not torch.cuda.is_available():
            analysis['device_timing_note']='MPS source-level GPU timing unavailable; add @flint.fast to the target function'
        self.out.mkdir(parents=True,exist_ok=True)
        suggestions=[];seen_proposals=set()
        for region in analysis['regions']:
            if not region['suggest']:continue
            proposal=propose_region(self.root,self.inventory[region['key']],region)
            if proposal['id'] in seen_proposals:continue
            seen_proposals.add(proposal['id'])
            proposal['workload_sources']={v['file']:digest(v['source']) for v in self.inventory.values()}
            (self.out/(proposal['id']+'.diff')).write_text(proposal['diff'])
            (self.out/(proposal['id']+'.json')).write_text(json.dumps(proposal,indent=2));suggestions.append(proposal)
            if len(suggestions)>=3:break
        self.result={**analysis,'schema':1,'root':str(self.root),'warmup_steps':self.warmup,
                     'profiled_steps':max(0,self.count-self.warmup) if self.adaptive else min(max(0,self.count-self.warmup),self.steps),'optimizer_steps_seen':self.count,
                     'source_scopes':self.scope_count,'scope_cap_reached':self.scope_count>=self.max_scopes,
                     'stop_reason':reason,'instrumentation_removed':sys.getprofile() is self.previous,
                     'observation':self.adaptive.report() if self.adaptive else None,'model_calls':0,'suggestions':suggestions,'profile_window_seconds':time.monotonic()-self.started if self.started else 0,
                     'source_hashes':{v['file']:digest(v['source']) for v in self.inventory.values()}}
        (self.out/'report.json').write_text(json.dumps(self.result,indent=2))
        # Long training runs retain the compact report, not the profiler event tree or ASTs.
        self.profile=None;self.inventory.clear();self.code.clear()

    def remove_hook_after_step(self,frame):
        # PyTorch iterates a live OrderedDict of global hooks. Removing a hook inside
        # that iteration can break other libraries' hooks. Detach on the wrapper's return.
        previous=sys.getprofile()
        def cleanup(current,event,arg):
            if previous is not None:previous(current,event,arg)
            if current is frame and event=='return':
                self.hook.remove();sys.setprofile(previous)
        sys.setprofile(cleanup)

    def step(self,optimizer,args,kwargs):
        self.count+=1
        try:
            if not self.finished:
                if self.count==self.warmup:self.start()
                elif self.active:
                    if self.adaptive:
                        now=time.monotonic()
                        state=self.adaptive.add(now-self.boundary_time,self.observed_signatures)
                        self.boundary_time=now
                        if state!='observing':self.stop('adaptive_'+state)
                    elif self.count>=self.warmup+self.steps:self.stop()
        except Exception as error:
            self.fail(error)
        if self.rehearsal.requested:
            self.rehearsal.completed+=1
            if self.rehearsal.completed>=self.rehearsal.requested:
                from .rehearsal import RehearsalComplete
                raise RehearsalComplete()
        elif self.finished:
            self.remove_hook_after_step(sys._getframe(1))

    def run(self,command,rehearsal_steps=None):
        from torch.optim.optimizer import register_optimizer_step_post_hook
        from .rehearsal import Rehearsal,RehearsalComplete
        rehearsal=self.rehearsal=Rehearsal(rehearsal_steps)
        from .config import read_config
        if read_config()['workload_kind']=='batch':
            from .units import subscribe
            from types import SimpleNamespace
            self.hook=SimpleNamespace(remove=subscribe(lambda *a:self.step(None,None,None),'batch'))
        else:self.hook=register_optimizer_step_post_hook(self.step)
        if self.warmup==0:
            try:self.start()
            except Exception as error:self.fail(error)
        old_argv=sys.argv;old_path=list(sys.path);success=False
        try:
            sys.argv=command;sys.path.insert(0,str(Path(command[0]).resolve().parent))
            runpy.run_path(command[0],run_name='__main__');success=True
        except RehearsalComplete:
            success=True
        except SystemExit as error:
            success=error.code in (None,0)
            if not success:raise
        finally:
            rehearsal.stop();success=success and rehearsal.report()['complete']
            self.hook.remove()
            if not self.finished:
                try:self.stop('training_ended')
                except Exception as error:self.fail(error)
            if self.failure:
                self.out.mkdir(parents=True,exist_ok=True)
                self.result={'schema':1,'suggestions':[],'profiling_error':self.failure,'model_calls':0,'instrumentation_removed':not self.active}
            self.result.update(training_success=success,command=command,rehearsal=rehearsal.report())
            (self.out/'report.json').write_text(json.dumps(self.result,indent=2))
            sys.argv=old_argv;sys.path[:]=old_path
        return self.result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--warmup',type=int,default=3)
    ap.add_argument('--steps',type=int,default=5);ap.add_argument('--minimum-share',type=float,default=.10)
    ap.add_argument('--rehearsal-steps',type=int)
    ap.add_argument('command',nargs=argparse.REMAINDER);a=ap.parse_args();command=a.command
    if command and command[0]=='--':command=command[1:]
    if not command or not command[0].endswith('.py') or a.warmup<0 or not 1<=a.steps<=50:ap.error('Use a Python script and 1–50 profiling steps')
    os.environ['FLINT_EXTERNAL_BUILDER']='1'
    Window(Path.cwd(),a.out,a.warmup,a.steps,a.minimum_share).run(command,a.rehearsal_steps)

if __name__=='__main__':main()
