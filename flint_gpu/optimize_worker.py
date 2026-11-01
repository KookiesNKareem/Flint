"""Credential-free GPU-side operations for a controller-owned optimization session."""
from pathlib import Path
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time

ARTIFACT_FILES = ('reference.py','spec.json','extra_specs.json','job.json','status.json',
                  'kernel.py','checks.json','holdout_protocol.json','holdout_results.json','revalidation.json')


def atomic(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def train(command, folder, mode, timeout, rehearsal_steps=None, required=False):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    report=folder/'dispatch.json';log=folder/'training.log'
    env={**os.environ,'FLINT_LIVE':'0','FLINT_MODE':mode,'FLINT_EXTERNAL_BUILDER':'1',
         'FLINT_REHEARSAL_STEPS':str(rehearsal_steps or 0),'FLINT_REQUIRE_QUALIFIED':'1' if required else '0',
         'FLINT_REPORT':str(report),'FLINT_LAUNCH_REPORT':str(report)}
    start=time.monotonic()
    with log.open('w') as stream:
        proc=subprocess.Popen([sys.executable,'-m','flint_gpu.launch',*command],env=env,
                              stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=proc.wait(timeout=timeout)
        except BaseException:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait();raise
    data=json.loads(report.read_text()) if report.exists() else {}
    data.update(exit_code=code,seconds=time.monotonic()-start,log=str(log))
    if code:data['failure_tail']=log.read_text(errors='replace')[-4000:]
    return data


def handle(request):
    project=Path(request['project']).resolve();os.chdir(project)
    if 'performance_policy' in request:
        os.environ['FLINT_MAX_MEMORY_GROWTH']=str(request['performance_policy']['max_memory_growth'])
    run=request['session']
    if not run.isalnum():raise ValueError('Invalid session ID')
    folder=project/'.flint/optimizations'/run;folder.mkdir(parents=True,exist_ok=True)
    owner=project/'.flint/optimization-owner.json'
    action=request['action']
    if action=='begin':
        from .leases import acquire,release
        if owner.exists():raise FileExistsError('Optimization session already owns this project')
        acquire(run,'optimization')
        try:
            with owner.open('x') as f:json.dump({'session':run,'pid':os.getpid(),'started':time.time()},f)
        except BaseException:
            release(run);raise
        receipt=project/'.flint/qualification.json'
        if receipt.exists():shutil.copy2(receipt,folder/'previous-qualification.json')
        atomic(receipt,{'accepted':False,'reason':'optimization running'})
        return {'folder':str(folder)}
    if not owner.exists() or json.loads(owner.read_text())['session']!=run:
        raise RuntimeError('Optimization session does not own this project')
    if action=='finish':
        from .leases import release
        atomic(folder/'summary.json',request['summary']);owner.unlink();release(run);return {'finished':True}
    if action=='capture':
        result=train(request['command'],folder/'capture','capture',request['timeout'],request.get('rehearsal_steps'))
        jobs={}
        for item in result.get('dispatch',[]):
            path=Path(item['job'])
            if path.joinpath('job.json').exists():
                state=json.loads((path/'status.json').read_text())
                if result.get('success') and state.get('state')=='needs_revalidation':
                    from .revalidate import revalidate
                    from kernel_contract.holdout import holdout_cases
                    protocol=holdout_cases(None);protocol['candidate_sha256']=state['kernel_sha256']
                    atomic(path/'holdout_protocol.json',protocol)
                    try:revalidate(path)
                    except Exception:
                        # Never expose held-out failures as repair feedback. Start a fresh
                        # generation if reuse cannot establish all observed coverage.
                        atomic(path/'status.json',{'state':'queued','reason':'additional coverage requires a new checked candidate'})
                jobs[str(path)]={n:(path/n).read_text() for n in ARTIFACT_FILES if (path/n).exists()}
        return {'training':result,'jobs':jobs}
    if action=='install':
        from .config import read_config
        from kernel_contract.evidence import sha256
        path=Path(request['job']).resolve()
        if path.parent!=Path(read_config()['cache_dir']).resolve():raise ValueError('Artifact outside configured cache')
        files=request['files']
        if set(files)-set(ARTIFACT_FILES):raise ValueError('Unexpected artifact file')
        state=json.loads(files['status.json']);meta=json.loads(files['job.json'])
        if state.get('state')!='ready' or not state.get('holdout_passed') or not state.get('compile_passed'):
            raise ValueError('Unchecked candidate')
        import hashlib
        if hashlib.sha256(files['kernel.py'].encode()).hexdigest()!=state['kernel_sha256']:
            raise ValueError('Candidate digest mismatch')
        from .identity import environment
        if meta['environment']!=environment():raise ValueError('Candidate environment changed')
        archive=folder/'artifacts'/str(request['round'])/path.name
        if path.exists():shutil.copytree(path,archive,ignore=shutil.ignore_patterns('__pycache__'))
        atomic(project/'.flint/qualification.json',{'accepted':False,'reason':'candidate changed; requalification required'})
        path.mkdir(parents=True,exist_ok=True)
        atomic(path/'status.json',{'state':'installing'})
        for name,content in files.items():
            if name!='status.json':
                temp=path/(name+'.tmp');temp.write_text(content);temp.replace(path/name)
        atomic(path/'status.json',state)
        return {'installed':state['kernel_sha256']}
    if action=='qualify':
        from .qualification import qualify
        result=qualify(request['command'],request['repeats'],request['timeout'],request['minimum_gain'],request['inputs'],request.get('rehearsal_steps'),request.get('require_unit_gain',False))
        archive=folder/('round-'+str(request['round']));archive.mkdir()
        for p in (project/'.flint').iterdir():
            if p.is_file() and (p.name=='qualification.json' or p.suffix in ('.json','.log')):
                shutil.copy2(p,archive/p.name)
        return result
    if action=='approve':
        from .approval import approve
        return approve(request['evidence_sha256'])
    if action=='verify':
        return train(request['command'],folder/'normal','auto',request['timeout'],request.get('rehearsal_steps'),required=True)
    if action=='revoke':
        atomic(project/'.flint/qualification.json',{'accepted':False,'reason':request['reason']})
        return {'revoked':True}
    raise ValueError('Unknown optimization action')


def main():
    def terminate(signum, frame):raise TimeoutError('Optimization deadline expired')
    signal.signal(signal.SIGTERM,terminate)
    request=json.load(sys.stdin)
    try:result=handle(request)
    except Exception as error:result={'worker_error':type(error).__name__+': '+str(error)}
    print('FLINT_RESULT '+json.dumps(result),flush=True)


if __name__=='__main__':main()
