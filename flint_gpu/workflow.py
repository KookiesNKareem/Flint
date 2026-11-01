"""One source proposal and optimization pipeline, with research and prepared launch policies."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid
from .optimize_worker import atomic

MARKER='FLINT_WORKFLOW '


def rejected_unchanged(receipt,command,config,env):
    from .qualification import snapshot,sha,SCHEMA
    if receipt.get('schema')!=SCHEMA or receipt.get('accepted') or receipt.get('environment')!=env:return False
    pairs=receipt.get('pairs',[])
    if len(pairs)<3 or not all(p.get(m,{}).get('success') for p in pairs for m in ('reference','benchmark')):return False
    old=receipt['workload']
    if snapshot(Path.cwd(),command,old['files'].keys())!=old:return False
    group=pairs[-1]['benchmark'].get('artifacts',{})
    if not group:return False
    for name,digest in group.items():
        if Path(name).name!=name:return False
        job=Path(config['cache_dir'])/name;state=json.loads((job/'status.json').read_text())
        if state.get('kernel_sha256')!=digest or sha(job/Path(state['kernel']).name)!=digest:return False
    return True


def ready(command,rehearsal_steps=None):
    from .qualification import activation
    from .config import read_config
    from .identity import environment
    config=read_config();env=environment();path=Path('.flint/qualification.json')
    try:
        receipt=json.loads(path.read_text())
        if not receipt.get('accepted'):
            from .approval import inspect_receipt
            unchanged=rejected_unchanged(receipt,command,config,env)
            return {'ready':False,'reason':receipt.get('reason','not qualified'),
                    'approval_available':unchanged and inspect_receipt(receipt)['approvable'],
                    'retry_allowed':not unchanged}
        group=receipt['pairs'][0]['benchmark']['artifacts']
        if not group:return {'ready':False,'reason':'no qualified artifacts'}
        old=sys.argv;sys.argv=command
        try:
            for name in group:
                job=Path(config['cache_dir'])/name
                state=json.loads((job/'status.json').read_text())
                allowed,reason=activation(job,state,config,env)
                if not allowed:return {'ready':False,'reason':reason}
        finally:sys.argv=old
        if rehearsal_steps and receipt.get('scope') in ('optimizer_step_rehearsal','batch_rehearsal') and receipt.get('rehearsal_steps',0)<rehearsal_steps:
            return {'ready':False,'reason':'Existing rehearsal is shorter than requested coverage'}
        return {'ready':True,'artifacts':group,'scope':receipt.get('scope'),
                'rehearsal_steps':receipt.get('rehearsal_steps')}
    except (OSError,KeyError,ValueError,TypeError) as error:
        return {'ready':False,'reason':'readiness evidence unavailable: '+type(error).__name__}


def execute(command,folder,discovery=None,required=False,timeout=None,live=False,owner=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'FLINT_LIVE':'1' if live else '0','FLINT_WORKFLOW_OWNER':owner or '', 'FLINT_EXTERNAL_BUILDER':'1','FLINT_REQUIRE_QUALIFIED':'1' if required else '0',
         'FLINT_MODE':'auto','FLINT_REHEARSAL_STEPS':'0','PYTHONUNBUFFERED':'1',
         'FLINT_REPORT':str(folder/'dispatch.json'),'FLINT_LAUNCH_REPORT':str(folder/'dispatch.json')}
    if discovery:
        args=[sys.executable,'-u','-m','flint_gpu.discovery','--out',str(folder),
              '--warmup',str(discovery['warmup']),'--steps',str(discovery['steps']),
              '--minimum-share',str(discovery['minimum_share'])]
        if discovery.get('rehearsal_steps'):args+=['--rehearsal-steps',str(discovery['rehearsal_steps'])]
        args+=['--',*command]
    else:args=[sys.executable,'-u','-m','flint_gpu.launch',*command]
    with (folder/'training.log').open('w') as log:
        proc=subprocess.Popen(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
        # Timeout is implemented by a watchdog, never by a blocking line read alone.
        import threading
        expired=threading.Event()
        def expire():
            expired.set()
            with contextlib.suppress(ProcessLookupError):os.killpg(proc.pid,signal.SIGKILL)
        timer=threading.Timer(timeout,expire) if timeout else None
        if timer:timer.start()
        try:
            for line in proc.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            code=proc.wait()
        except BaseException:
            with contextlib.suppress(ProcessLookupError):os.killpg(proc.pid,signal.SIGKILL)
            proc.wait();raise
        finally:
            if timer:timer.cancel()
    report=folder/('report.json' if discovery else 'dispatch.json')
    result=json.loads(report.read_text()) if report.exists() else {}
    result.update(exit_code=code,timed_out=expired.is_set(),log=str(folder/'training.log'))
    return result


def worker(request):
    from .leases import acquire,release
    root=Path(request['project']).resolve();os.chdir(root);action=request['action']
    if action=='ready':
        from .identity import environment
        return {**ready(request['command'],request.get('optimization',{}).get('rehearsal_steps')),
                'has_targets':has_targets(root),'cuda_available':bool(environment()['gpu'])}
    if action=='review_valid':
        from .suggestions import digest
        try:
            return {'valid':all((root/p).resolve().is_relative_to(root) and digest((root/p).read_text())==h
                                for p,h in request['sources'].items())}
        except OSError:return {'valid':False}
    if action=='resume':
        identifier=request['id']
        if not identifier.isalnum():raise ValueError('Invalid live request')
        atomic(root/'.flint/workflows'/identifier/'live-response.json',request['result'])
        return {'resumed':True}
    if action=='apply':
        from .suggestions import apply_proposal
        owner=uuid.uuid4().hex;acquire(owner,'source edit')
        try:return apply_proposal(request['proposal'],root)
        finally:release(owner)
    if action not in ('discover','run'):raise ValueError('Unknown workflow action')
    owner=request['id']
    if not isinstance(owner,str) or not owner.isalnum():raise ValueError('Invalid workflow identifier')
    acquire(owner,action)
    try:
        if action=='run' and request.get('required'):
            readiness=ready(request['command'],request.get('optimization',{}).get('rehearsal_steps'))
            if not readiness['ready']:return {'launched':False,'readiness':readiness}
        folder=root/'.flint/workflows'/owner
        cache=root/'.flint/discovery-cache.json';fingerprint=None
        if action=='discover':
            from .qualification import snapshot
            from .identity import environment
            fingerprint={'workload':snapshot(root,request['command'],request.get('optimization',{}).get('inputs',[])),
                         'environment':environment(),'window':request['discovery']}
            if cache.exists():
                previous=json.loads(cache.read_text())
                if previous.get('fingerprint')==fingerprint and not request['discovery'].get('rehearsal_steps'):
                    training=execute(request['command'],folder,timeout=request.get('run_timeout'))
                    report={**previous['report'],'cached_discovery':True,'profiling_steps_this_run':0,
                            'training_success':training.get('success',False),'exit_code':training['exit_code']}
                    return {'launched':True,'result':report,'worker_evidence':str(folder)}
        result=execute(request['command'],folder,request.get('discovery') if action=='discover' else None,
                       request.get('required',False),request.get('run_timeout'),live=request.get('policy')=='adaptive' and not request.get('retain_reference'),owner=owner)
        if action=='discover' and result.get('training_success') and not result.get('profiling_error'):
            # Never reuse an attribution after source/config/data changed during the observed run.
            from .qualification import snapshot
            if snapshot(root,request['command'],request.get('optimization',{}).get('inputs',[]))==fingerprint['workload']:
                atomic(cache,{'fingerprint':fingerprint,'report':result})
        return {'launched':True,'result':result,'worker_evidence':str(folder)}
    finally:release(owner)


def rpc(remote,request,log_path):
    python=shlex.quote(sys.executable) if remote=='bash -c' else 'python'
    args=shlex.split(remote)+[python+' -u -m flint_gpu.workflow worker']
    proc=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                          text=True,start_new_session=True)
    proc.stdin.write(json.dumps(request));proc.stdin.close();response=None
    with Path(log_path).open('w') as log:
        try:
            for line in proc.stdout:
                log.write(line);log.flush()
                if line.startswith('FLINT_LIVE_REQUEST ') and request.get('action')=='run' and request.get('policy')=='adaptive':
                    token=json.loads(line[len('FLINT_LIVE_REQUEST '):])
                    if token.get('owner')!=request['id']:raise ValueError('Live controller identity mismatch')
                    try:
                        print('Qualifying at a completed work boundary…',flush=True)
                        result=optimize_request(request)
                    except Exception as error:result={'accepted':False,'reason':str(error)}
                    rpc(remote,{**request,'action':'resume','result':result},Path(request['out'])/'resume.log')
                elif line.startswith(MARKER):response=json.loads(line[len(MARKER):])
                else:print(line,end='',flush=True)
            code=proc.wait()
        except BaseException:
            with contextlib.suppress(ProcessLookupError):os.killpg(proc.pid,signal.SIGTERM)
            proc.wait();raise
    if code or response is None or response.get('error'):
        raise RuntimeError((response or {}).get('error','Worker failed; see '+str(log_path)))
    return response


def optimize_request(request):
    from .optimization import optimize
    options=request['optimization']
    return optimize(request['command'],project=request['project'],remote=request['remote'],
                    out=str(Path(request['out'])/'optimization'),**options)


def background_job(path):
    request=json.loads(Path(path).read_text());out=Path(request['out'])
    def cancelled(signum,frame):raise KeyboardInterrupt('background optimization cancelled')
    signal.signal(signal.SIGTERM,cancelled)
    atomic(out/'state.json',{'state':'optimizing','pid':os.getpid()})
    try:
        result=optimize_request(request)
        state='ready' if result['accepted'] else 'review_required' if result.get('reason')=='performance_tradeoff_needs_review' else 'reference_retained'
        atomic(out/'state.json',{'state':state,'result':result})
    except BaseException as error:
        atomic(out/'state.json',{'state':'failed','reason':type(error).__name__+': '+str(error)})




def workload_key(request):
    import hashlib
    return hashlib.sha256(json.dumps([request['remote'],request['project'],request['command']]).encode()).hexdigest()[:24]


def controller_alive(task,pid):
    probe=subprocess.run(['ps','-p',str(pid),'-o','args='],capture_output=True,text=True)
    return str(task/'request.json') in probe.stdout and 'flint_gpu.workflow' in probe.stdout


def wait_for_previous(request,out):
    record=out.parent/('.active-'+workload_key(request)+'.json')
    if not record.exists():return
    task=Path(json.loads(record.read_text())['task']);announced=False
    while True:
        state=json.loads((task/'state.json').read_text())
        if state['state'] not in ('queued','optimizing'):return
        pid=state.get('pid') or json.loads((task/'background-pid.json').read_text())['pid']
        if not controller_alive(task,pid):
            raise RuntimeError('Previous Flint controller stopped unexpectedly; inspect its worker before retrying: '+str(task))
        if not announced:print('Waiting for the current optimization to finish…',flush=True);announced=True
        time.sleep(.5)

def has_targets(root):
    import ast
    from .suggestions import functions
    try:inventory=functions(root)
    except (OSError,ValueError):return None
    for item in inventory.values():
        aliases=set();direct=set()
        for n in item['tree'].body:
            if isinstance(n,ast.Import):aliases.update(a.asname or a.name for a in n.names if a.name in ('flint','flint_gpu','muse_kernel'))
            if isinstance(n,ast.ImportFrom) and n.module in ('flint','flint_gpu','muse_kernel'):
                direct.update(a.asname or a.name for a in n.names if a.name=='fast')
        for decorator in item['node'].decorator_list:
            d=decorator.func if isinstance(decorator,ast.Call) else decorator
            if isinstance(d,ast.Name) and d.id in direct:return True
            if isinstance(d,ast.Attribute) and d.attr=='fast' and isinstance(d.value,ast.Name) and d.value.id in aliases:return True
    return False


def save_state(out,state):
    state={**state,'out':str(out)};atomic(out/'state.json',state);return state


def start_background(request,out):
    # Persist before spawning so immediate completion cannot be overwritten by the parent.
    atomic(out/'state.json',{'state':'queued'})
    with (out/'background.log').open('w') as log:
        child=subprocess.Popen([sys.executable,'-m','flint_gpu.workflow','background',str(out/'request.json')],
                               stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic(out/'background-pid.json',{'pid':child.pid})
    atomic(out.parent/('.active-'+workload_key(request)+'.json'),{'task':str(out)})
    return {'state':'optimization_started','pid':child.pid,'out':str(out)}


def review_proposal(proposal):
    print(proposal['diff'],flush=True)
    if not sys.stdin.isatty():return None
    try:answer=input('Apply this change and continue? [y/N] ').strip().lower()
    except EOFError:return None
    return answer in ('y','yes')


def discover_and_review(request,out,warmup,steps,minimum_share,policy):
    key=workload_key(request)
    record_path=out.parent/('.review-'+key+'.json')
    record=json.loads(record_path.read_text()) if record_path.exists() else {}
    report=record.get('pending')
    if report:
        valid=rpc(request['remote'],{**request,'action':'review_valid','sources':report['source_hashes']},out/'review-valid.log')
        if not valid['valid']:report=None
    if report is None:
        window={'warmup':warmup,'steps':steps,'minimum_share':minimum_share}
        if policy=='prepared':
            count=request['optimization']['rehearsal_steps']
            window.update(warmup=min(warmup,count-1),steps=min(steps,count-min(warmup,count-1)),rehearsal_steps=count)
        result=rpc(request['remote'],{**request,'action':'discover','discovery':window},out/'discovery-worker.log')
        report=result['result']
    atomic(out/'discovery.json',report)
    if not report.get('training_success'):
        return save_state(out,{'state':'training_failed','reason':'Discovery workload did not complete successfully'})
    proposals=[p for p in report.get('suggestions',[]) if p['id'] not in record.get('declined',[])]
    if not proposals:
        record.pop('pending',None);atomic(record_path,record)
        return save_state(out,{'state':'not_ready' if policy=='prepared' else 'reference_retained',
                               'launched':False if policy=='prepared' else True,
                               'reason':'No supported source change to review'})
    proposal=proposals[0]
    atomic(out/(proposal['id']+'.json'),proposal);(out/(proposal['id']+'.diff')).write_text(proposal['diff'])
    record['pending']=report;atomic(record_path,record)
    answer=review_proposal(proposal)
    if answer is None:
        print('Source review pending. Run the same command in a terminal to review it.',flush=True)
        return save_state(out,{'state':'review_required','proposal':str(out/(proposal['id']+'.diff'))})
    if not answer:
        record.setdefault('declined',[]).append(proposal['id']);record.pop('pending',None);atomic(record_path,record)
        return save_state(out,{'state':'not_ready' if policy=='prepared' else 'reference_retained','reason':'Source change declined'})
    applied=rpc(request['remote'],{**request,'action':'apply','proposal':proposal},out/'apply.log')
    atomic(out/'applied.json',applied)
    record.pop('pending',None);atomic(record_path,record)
    return None

def run(command,policy=None,project=None,remote=None,out=None,ledger=None,max_usd=None,
        minutes=None,rehearsal_steps=None,warmup=3,steps=5,minimum_share=.10,timeout=300,
        run_timeout=None,inputs=(),max_rounds=2,prepare_only=False,model=None):
    from .config import read_config
    config=read_config()
    policy=policy or config['launch_policy']
    minutes=config['optimization_minutes'] if minutes is None else minutes
    rehearsal_steps=config['rehearsal_steps'] if rehearsal_steps is None else rehearsal_steps
    if policy not in ('observe','background','prepared','adaptive'):raise ValueError('Unknown launch policy')
    if prepare_only and policy!='prepared':raise ValueError('--prepare-only requires --policy prepared')
    if not command or not command[0].endswith('.py'):raise ValueError('Use the existing Python script and its arguments')
    if not 0<=warmup<=100 or not 1<=steps<=50 or not 0<minimum_share<=1:raise ValueError('Invalid profiling window or share')
    if not 1<=rehearsal_steps<=10000:raise ValueError('Use 1–10000 rehearsal optimizer steps')
    project=project or str(Path.cwd());remote=remote or config['remote']
    max_usd=config['max_usd'] if max_usd is None else max_usd
    identifier=uuid.uuid4().hex;out=Path(out or Path.cwd()/'.flint/workflows'/identifier).resolve();out.mkdir(parents=True,exist_ok=False)
    request={'id':identifier,'command':command,'project':project,'remote':remote,'out':str(out),
             'run_timeout':run_timeout,'optimization':{'ledger':str(Path(ledger or config['budget_ledger']).expanduser().resolve()),
             'max_usd':max_usd,'minutes':minutes,'rehearsal_steps':rehearsal_steps,'timeout':timeout,
             'inputs':list(inputs),'max_rounds':max_rounds,'require_unit_gain':policy=='adaptive','model':model}}
    request['policy']=policy
    wait_for_previous(request,out)
    atomic(out/'request.json',request)
    if policy=='observe':
        result=rpc(remote,{**request,'action':'discover','discovery':{'warmup':warmup,'steps':steps,'minimum_share':minimum_share}},out/'worker.log')
        report=result['result'];atomic(out/'discovery.json',report)
        if report.get('cached_discovery'):print('Reused discovery; no profiling this run.',flush=True)
        if not report.get('suggestions'):
            print('No supported source change identified.',flush=True)
        work=report.get('device_work_us',0)
        if work and report.get('unattributed_device_us',0)/work>.2:
            print(f"{report['unattributed_device_us']/work:.0%} of GPU kernel work has no source attribution.",flush=True)
        for proposal in report.get('suggestions',[]):
            atomic(out/(proposal['id']+'.json'),proposal)
            (out/(proposal['id']+'.diff')).write_text(proposal['diff'])
            print(f"{proposal['file']}:{proposal['line']} · {proposal['share']:.0%}" if 'share' in proposal else
                  f"{proposal['file']}:{proposal['line']} · {proposal['evidence']['share']:.0%} of GPU kernel work",flush=True)
            print(proposal['diff'],flush=True)
        atomic(out/'state.json',{'state':'observed','success':report.get('training_success'),
                               'suggestions':len(report.get('suggestions',[]))})
        return {'state':'observed','out':str(out),'report':report}
    readiness=rpc(remote,{**request,'action':'ready'},out/'readiness.log')
    if readiness.get('approval_available'):
        from .approval import review_on_terminal
        approval=review_on_terminal(remote,project)
        if isinstance(approval,dict) and approval.get('accepted'):
            readiness=rpc(remote,{**request,'action':'ready'},out/'readiness-after-approval.log')
        elif policy=='prepared':
            return save_state(out,{'state':'review_required','launched':False,
                'reason':'Performance tradeoff declined' if approval is False else 'Performance tradeoff awaits review'})
    if not readiness['ready'] and readiness.get('retry_allowed') is False:
        request['retain_reference']=True
    # Fresh undecorated projects discover and review inside the same command. A pending
    # review is resumed without rerunning the user's script merely to display its diff.
    if not readiness['ready'] and readiness.get('has_targets') is False:
        result=discover_and_review(request,out,warmup,steps,minimum_share,policy)
        if result is not None:return result
        readiness=rpc(remote,{**request,'action':'ready'},out/'after-review.log')
        if policy in ('background','adaptive'):
            if readiness.get('cuda_available') is False:
                return save_state(out,{'state':'reference_retained','reason':'Source change applied; a CUDA or Metal worker is needed for generation'})
            if policy=='background':return start_background(request,out)
            return save_state(out,{'state':'reference_retained','reason':'Source change applied. The next invocation can activate at a work boundary.'})
    if not readiness['ready'] and readiness.get('cuda_available') is False and policy=='prepared':
        return save_state(out,{'state':'not_ready','launched':False,'reason':'A CUDA or Metal worker is needed for preparation'})
    if policy=='prepared' and not readiness['ready'] and readiness.get('retry_allowed') is False:
        state={'state':'not_ready','launched':False,'reason':'Unchanged candidate was already rejected: '+readiness['reason'],'out':str(out)}
        atomic(out/'state.json',state);return state
    if policy=='prepared' and not readiness['ready']:
        atomic(out/'state.json',{'state':'preparing'})
        result=optimize_request(request)
        if not result['accepted']:
            state={'state':'not_ready','launched':False,'reason':result['reason'],'out':str(out)}
            atomic(out/'state.json',state);return state
    if policy=='prepared' and prepare_only:
        state={'state':'ready','launched':False,'out':str(out)};atomic(out/'state.json',state);return state
    atomic(out/'state.json',{'state':'training','policy':policy})
    result=rpc(remote,{**request,'action':'run','required':policy=='prepared'},out/'training-worker.log')
    atomic(out/'training.json',result)
    if not result.get('launched') or not result.get('result',{}).get('success'):
        state={'state':'training_failed','out':str(out),'result':result};atomic(out/'state.json',state);return state
    if policy=='background' and not readiness['ready']:
        if readiness.get('retry_allowed') is False:
            state={'state':'reference_retained','reason':'Unchanged candidate was already rejected; no repeated optimization','out':str(out)}
            atomic(out/'state.json',state);return state
        if not result['result'].get('dispatch'):
            state={'state':'reference_retained','reason':'No supported decorated target was captured','out':str(out)}
            atomic(out/'state.json',state);return state
        if readiness.get('cuda_available') is False:
            return save_state(out,{'state':'reference_retained','reason':'A CUDA or Metal worker is needed for generation'})
        return start_background(request,out)
    state={'state':'completed','out':str(out),'policy':policy};atomic(out/'state.json',state);return state


def main():
    if sys.argv[1]=='background':background_job(sys.argv[2]);return
    if sys.argv[1]!='worker':raise SystemExit('Expected worker or background')
    def terminate(signum,frame):raise KeyboardInterrupt('worker interrupted')
    signal.signal(signal.SIGTERM,terminate)
    try:result=worker(json.load(sys.stdin))
    except BaseException as error:result={'error':type(error).__name__+': '+str(error)}
    print(MARKER+json.dumps(result),flush=True)

if __name__=='__main__':main()
