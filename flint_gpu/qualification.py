"""Whole-command qualification, with source-bound activation and conservative admission."""
from pathlib import Path
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

SCHEMA = 4
EXCLUDED = {'.git','.flint','.venv','venv','__pycache__','node_modules','dist','build','runs','runs_public','outputs'}
SOURCE_SUFFIXES = {'.py','.toml','.yaml','.yml','.json'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def source_files(root):
    root=Path(root)
    result=[]
    for base, dirs, files in os.walk(root):
        dirs[:]=[d for d in dirs if d not in EXCLUDED and not d.startswith('.')]
        for name in files:
            p=Path(base)/name
            if p.suffix in SOURCE_SUFFIXES or p.name in ('flint.lock','muse.lock'):result.append(str(p.relative_to(root)))
        if len(result)>10000:raise ValueError('Workload source set too large; use an isolated project directory')
    return sorted(result)


def snapshot(root, argv, inputs=()):
    root=Path(root).resolve()
    files=source_files(root)
    extra=[str(Path(p).resolve()) for p in inputs]
    command=[str(Path(argv[0]).resolve()),*argv[1:]]
    for arg in argv:
        path=Path(arg)
        if path.is_file():extra.append(str(path.resolve()))
    all_files=sorted(set([str(root/p) for p in files]+extra))
    import importlib.metadata
    try: runtime_version=importlib.metadata.version('flint-gpu')
    except importlib.metadata.PackageNotFoundError: runtime_version='source'
    effective_env={k:v for k,v in os.environ.items() if k.startswith(('CUDA_','TORCH_','PYTORCH_','OMP_','MKL_'))}
    # torch.cuda._lazy_init adds this default; initialization is not a policy change.
    effective_env.setdefault('CUDA_MODULE_LOADING','LAZY')
    return {'runtime_version':runtime_version,
            'runtime_code':{p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
            'root':str(root),'command':command,'source_files':files,
            'files':{p:sha(p) for p in all_files},
            'environment':effective_env}


def decision(pairs, minimum_gain=.05, require_unit_gain=False, memory_growth=0.0, memory_ceiling=None):
    """Speed evidence is mandatory; memory is an explicit resource constraint."""
    import math
    if not isinstance(memory_growth,(int,float)) or not math.isfinite(memory_growth) or memory_growth<0:
        return False,'invalid memory allowance'
    if memory_ceiling is not None and (not isinstance(memory_ceiling,int) or memory_ceiling<=0):
        return False,'invalid memory ceiling'
    if len(pairs)<3:return False,'need at least three paired complete-command measurements'
    for pair in pairs:
        for mode in ('reference','benchmark'):
            r=pair[mode]
            if not r.get('success') or not 0<r.get('seconds',0)<float('inf'):
                return False,'failed or invalid command measurement'
            if not isinstance(r.get('peak_allocated_bytes'),int) or r['peak_allocated_bytes']<=0:
                return False,'missing CUDA peak memory evidence'
        if not pair['benchmark'].get('artifacts'):return False,'candidate execution not observed'
    if require_unit_gain:
        units={mode:[] for mode in ('reference','benchmark')}
        kinds=set()
        for pair in pairs:
            for mode in units:
                evidence=pair[mode].get('work_units') or {}
                value=evidence.get('median_seconds')
                if not evidence.get('synchronized') or not isinstance(value,(int,float)) or not 0<value<float('inf'):
                    return False,'missing steady-state work-unit evidence'
                kinds.add(evidence.get('kind'));units[mode].append(value)
        if len(kinds)!=1:return False,'work-unit adapters differ'
        if max(units['benchmark'])>=min(units['reference'])*(1-minimum_gain):
            return False,'steady-state work-unit improvement is absent or inconclusive'
    baseline=[p['reference']['seconds'] for p in pairs]
    candidate=[p['benchmark']['seconds'] for p in pairs]
    if max(candidate) >= min(baseline)*(1-minimum_gain):
        return False,'speed improvement is absent or inconclusive'
    kinds={(p[m].get('memory_measurement') or {}).get('kind','cuda_allocator_peak') for p in pairs for m in ('reference','benchmark')}
    if len(kinds)!=1 or not kinds <= {'cuda_allocator_peak','mps_sampled_allocated'}:
        return False,'incompatible memory measurements'
    if 'mps_sampled_allocated' in kinds and memory_ceiling is None:
        return False,'MPS peak memory is sampled; review required'
    baseline_memory=min(p['reference']['peak_allocated_bytes'] for p in pairs)
    candidate_memory=max(p['benchmark']['peak_allocated_bytes'] for p in pairs)
    ceiling=memory_ceiling if memory_ceiling is not None else baseline_memory*(1+memory_growth)
    if candidate_memory>ceiling:
        return False,'candidate increases peak allocated CUDA memory'
    groups=[p['benchmark']['artifacts'] for p in pairs]
    if any(g!=groups[0] for g in groups[1:]):return False,'candidate artifact set changed between repetitions'
    return True,('all measured commands faster beyond margin within the approved memory allowance'
                 if candidate_memory>baseline_memory else
                 'all measured commands faster beyond margin with no allocated-memory regression')


def activation(job, status, config, env, *, receipt_override=None, argv=None):
    """Read once at runtime construction; no qualification I/O in the function's hot path."""
    try:
        path=Path.cwd()/'.flint/qualification.json'
        receipt=receipt_override if receipt_override is not None else json.loads(path.read_text())
        if receipt.get('schema')!=SCHEMA or not receipt.get('accepted'):return False,'workload performance not qualified'
        old=receipt['workload']
        current=snapshot(Path.cwd(),sys.argv if argv is None else argv,old['files'].keys())
        if current!=old:return False,'workload changed since qualification'
        if receipt['environment']!=env:return False,'qualification environment changed'
        growth=receipt.get('performance_policy',{}).get('max_memory_growth',0.0)
        ceiling=None
        if receipt.get('approval'):
            from .approval import approval_ceiling
            ceiling=approval_ceiling(receipt)
        accepted,reason=decision(receipt['pairs'],receipt['minimum_gain'],receipt.get('require_unit_gain',False),growth,ceiling)
        if not accepted:return False,reason
        group=receipt['pairs'][0]['benchmark']['artifacts']
        prep=receipt.get('preparation',{})
        if receipt.get('cache_policy')!='one_complete_command_per_mode_before_fixed_pairs_v1' or not all(
            prep.get(m,{}).get('success') for m in ('reference','benchmark')):
            return False,'qualification preparation evidence missing'
        if prep['benchmark'].get('artifacts')!=group:return False,'prepared artifact set changed'
        if group.get(job.name)!=status['kernel_sha256']:return False,'artifact not qualified for this workload'
        # Promotion is joint: do not activate a partial set after another artifact changes.
        for name,digest in group.items():
            folder=Path(config['cache_dir'])/name
            state=json.loads((folder/'status.json').read_text())
            metadata=json.loads((folder/'job.json').read_text())
            if metadata.get('environment') != env or (metadata.get('require_holdout') and not state.get('holdout_passed')) or (metadata.get('native') and not state.get('compile_passed')):
                return False,'qualified artifact checks or environment changed'
            if state['state']!='ready' or state['kernel_sha256']!=digest or sha(folder/Path(state['kernel']).name)!=digest:
                return False,'qualified artifact set changed'
        return True,reason
    except (OSError,ValueError,KeyError,TypeError):
        return False,'workload performance not qualified'


def qualify(command, repeats=3, timeout=300, minimum_gain=.05, inputs=(), rehearsal_steps=None, require_unit_gain=False):
    """Run exactly the user's Python script in isolated comparison processes."""
    if repeats<3 or not .01<=minimum_gain<1:raise ValueError('Use >=3 repeats and a gain margin of at least 1%')
    if rehearsal_steps is not None and not 1<=rehearsal_steps<=10000:raise ValueError('Use 1–10000 rehearsal optimizer steps')
    if not command or not Path(command[0]).is_file() or Path(command[0]).suffix!='.py':
        raise ValueError('Pass a Python script and its normal arguments after --')
    root=Path.cwd(); folder=root/'.flint';folder.mkdir(exist_ok=True)
    workload=snapshot(root,command,inputs)
    output=folder/'qualification.json'
    # Revoke stale permission before starting another qualification.
    output.write_text(json.dumps({'schema':SCHEMA,'accepted':False,'reason':'qualification running'}))
    def run(mode, label):
        report=folder/f'{label}-{mode}.json';log=folder/f'{label}-{mode}.log'
        report.unlink(missing_ok=True)
        childenv={**os.environ,'FLINT_LIVE':'0','FLINT_MODE':mode,'MUSE_MODE':mode,
                  'FLINT_MEASURE_UNITS':'1' if require_unit_gain else '0','FLINT_REHEARSAL_STEPS':str(rehearsal_steps or 0),'FLINT_REQUIRE_QUALIFIED':'0',
                  'FLINT_REPORT':str(report),'MUSE_REPORT':str(report),
                  'FLINT_RUN_ID':f'{label}-{mode}','FLINT_LAUNCH_REPORT':str(report)}
        start=time.monotonic()
        with log.open('w') as stream:
            proc=subprocess.Popen([sys.executable,'-m','flint_gpu.launch',*command],env=childenv,
                                  stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(proc.pid,signal.SIGKILL);proc.wait();code=-9
            except BaseException:
                import signal
                os.killpg(proc.pid,signal.SIGKILL);proc.wait();raise
        elapsed=time.monotonic()-start
        try:data=json.loads(report.read_text())
        except (OSError,ValueError):data={}
        result={'success':code==0 and data.get('success',False),'seconds':elapsed,
                    'rehearsal':data.get('rehearsal'),'work_units':data.get('work_units'),
                    'exit_code':code,
                    'peak_allocated_bytes':data.get('peak_allocated_bytes'),
                    'memory_measurement':data.get('memory_measurement'),
                    'artifacts':data.get('artifacts',{}),'log':str(log),
                    'dispatch':data.get('dispatch',[]),
                    'failure_tail':log.read_text(errors='replace')[-4000:] if code else None}
        print(f'{label} {mode}: {elapsed:.3f}s, exit {code}',flush=True)
        return result
    # Capture already exercises the reference. Prepare both paths once, before any measured
    # pair, so a new candidate's cold compiler allocations are not compared to a warm reference.
    # These are real complete commands; their time and peak memory remain in the receipt.
    preparation={}
    for mode in ('reference','benchmark'):
        preparation[mode]=run(mode,'prepare')
        if not preparation[mode]['success']:break
    prepared=(len(preparation)==2 and all(r['success'] for r in preparation.values())
              and bool(preparation['benchmark']['artifacts']))
    pairs=[]
    for i in range(repeats if prepared else 0):
        pair={}
        for mode in (('reference','benchmark') if i%2==0 else ('benchmark','reference')):
            pair[mode]=run(mode,str(i))
        pairs.append(pair)
        if not all(p['success'] for p in pair.values()):break
    from .config import read_config
    policy={'max_memory_growth':read_config()['max_memory_growth']}
    accepted,reason=decision(pairs,minimum_gain,require_unit_gain,policy['max_memory_growth'])
    if not prepared:accepted,reason=False,'preparation command failed'
    if accepted and preparation['benchmark']['artifacts']!=pairs[0]['benchmark']['artifacts']:
        accepted,reason=False,'prepared artifact set changed'
    if snapshot(root,command,inputs)!=workload:accepted,reason=False,'workload changed during qualification'
    from .identity import environment
    preparation_seconds=sum(r['seconds'] for r in preparation.values())
    spent=preparation_seconds+sum(r['seconds'] for p in pairs for r in p.values())
    saving=(statistics.median(p['reference']['seconds'] for p in pairs)-
            statistics.median(p['benchmark']['seconds'] for p in pairs)) if pairs else 0
    import math
    receipt={'performance_policy':policy,'require_unit_gain':require_unit_gain,'qualification_seconds':spent,
             'scope':('batch_rehearsal' if __import__('flint_gpu.config',fromlist=['read_config']).read_config()['workload_kind']=='batch' else 'optimizer_step_rehearsal') if rehearsal_steps else 'full_command',
             'rehearsal_steps':rehearsal_steps,
             'preparation_seconds':preparation_seconds,'preparation':preparation,
             'cache_policy':'one_complete_command_per_mode_before_fixed_pairs_v1',
             'qualification_break_even_runs':math.ceil(spent/saving) if saving>0 else None,
             'break_even_note':'Includes both preparation commands and measured pairs; excludes capture, model generation and earlier setup. Admission covers prepared compiler caches, not a future cold-cache run.',
             'schema':SCHEMA,'accepted':accepted,'reason':reason,'minimum_gain':minimum_gain,
             'workload':workload,'environment':environment(),'pairs':pairs}
    temp=output.with_suffix('.tmp')
    receipt['approval_available']=not accepted and reason in ('candidate increases peak allocated CUDA memory','MPS peak memory is sampled; review required')
    temp.write_text(json.dumps(receipt,indent=2));temp.replace(output)
    print(('Accepted: ' if accepted else 'Ready for review: ' if receipt['approval_available'] else 'Kept default: ')+reason,flush=True)
    if receipt['approval_available']:print('Run flint approve to review this measured tradeoff.',flush=True)
    return receipt
