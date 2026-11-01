"""Review measured performance tradeoffs without overriding correctness or speed evidence."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import time


def evidence_digest(receipt):
    keys=('schema','workload','environment','pairs','preparation','cache_policy',
          'minimum_gain','require_unit_gain','performance_policy')
    data={k:receipt.get(k) for k in keys}
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def approval_ceiling(receipt):
    approval=receipt['approval']
    peak=max(p['benchmark']['peak_allocated_bytes'] for p in receipt['pairs'])
    if (approval.get('kind')!='measured_memory_tradeoff' or
        approval.get('evidence_sha256')!=evidence_digest(receipt) or
        approval.get('max_peak_allocated_bytes')!=peak):
        raise ValueError('Approved performance evidence changed')
    return peak


def inspect_receipt(receipt):
    from .qualification import decision,SCHEMA
    pairs=receipt['pairs']
    baseline=min(p['reference']['peak_allocated_bytes'] for p in pairs)
    peak=max(p['benchmark']['peak_allocated_bytes'] for p in pairs)
    eligible,reason=decision(pairs,receipt['minimum_gain'],receipt.get('require_unit_gain',False),memory_ceiling=peak)
    eligible=eligible and receipt.get('schema')==SCHEMA and receipt.get('reason') in ('candidate increases peak allocated CUDA memory','MPS peak memory is sampled; review required')
    units=[((p.get('reference',{}).get('work_units') or {}).get('median_seconds'),(p.get('benchmark',{}).get('work_units') or {}).get('median_seconds')) for p in pairs]
    unit_speedup=statistics.median(x for x,y in units)/statistics.median(y for x,y in units) if all(isinstance(x,(int,float)) and isinstance(y,(int,float)) and x>0 and y>0 for x,y in units) else None
    return {'unit_speedup':unit_speedup,'approvable':eligible,'reason':reason,'evidence_sha256':evidence_digest(receipt),
            'command_speedup':statistics.median(p['reference']['seconds'] for p in pairs)/statistics.median(p['benchmark']['seconds'] for p in pairs),
            'reference_peak_bytes':baseline,'candidate_peak_bytes':peak,
            'extra_bytes':peak-baseline,'memory_growth':max(0,peak/baseline-1),
            'memory_measurement':(pairs[0]['benchmark'].get('memory_measurement') or {}).get('kind','cuda_allocator_peak'),
            'already_accepted':bool(receipt.get('accepted'))}


def approve(expected_digest, remember=False):
    from .qualification import activation
    if remember:
        from .setup import config_path
        if config_path().resolve().is_relative_to(Path.cwd().resolve()):
            raise ValueError('Remember requires personal configuration outside the qualified project; approve without --remember or move FLINT_CONFIG')
    from .identity import environment
    from .config import read_config
    path=Path('.flint/qualification.json');receipt=json.loads(path.read_text())
    view=inspect_receipt(receipt)
    if view['evidence_sha256']!=expected_digest:raise ValueError('Evidence changed since review')
    if remember and view['memory_measurement']=='mps_sampled_allocated':raise ValueError('An MPS sampled-memory review cannot become an exact peak-memory allowance')
    if not view['approvable']:raise ValueError('Only a measured memory tradeoff can be approved; correctness and speed requirements remain mandatory')
    receipt['approval']={'kind':'measured_memory_tradeoff','evidence_sha256':expected_digest,
                         'max_peak_allocated_bytes':view['candidate_peak_bytes'],'approved_at':time.time()}
    receipt['accepted']=True
    config=read_config();env=environment()
    for name in receipt['pairs'][0]['benchmark']['artifacts']:
        job=Path(config['cache_dir'])/name;state=json.loads((job/'status.json').read_text())
        valid,reason=activation(job,state,config,env,receipt_override=receipt,argv=receipt['workload']['command'])
        if not valid:raise ValueError('Approval refused: '+reason)
    # Do not approve a receipt that changed while source/artifact checks ran.
    current=json.loads(path.read_text())
    if evidence_digest(current)!=expected_digest:raise ValueError('Evidence changed during approval')
    receipt['reason']='measured memory tradeoff explicitly approved for this workload and artifact set'
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(receipt,indent=2));temp.replace(path)
    if remember:
        from .setup import save_settings
        save_settings({'max_memory_growth':max(config['max_memory_growth'],view['memory_growth'])})
    return {**view,'accepted':True,'remembered':remember,'approval':receipt['approval']}


def handle(request):
    os.chdir(Path(request['project']).resolve())
    if request['action']=='review':return inspect_receipt(json.loads(Path('.flint/qualification.json').read_text()))
    if request['action']=='approve':return approve(request['evidence_sha256'],request.get('remember',False))
    raise ValueError('Unknown approval action')


def review_on_terminal(remote=None, project=None, *, yes=False, remember=False):
    """The foreground controller owns input; workers never read the user's terminal."""
    import shlex,subprocess
    from .terminal import Progress,confirm_tradeoff,show_tradeoff
    def call(action,**kwargs):
        request={'action':action,'project':project or str(Path.cwd()),**kwargs}
        if remote:
            import sys
            python=shlex.quote(sys.executable) if remote=='bash -c' else 'python'
            proc=subprocess.run(shlex.split(remote)+[python+' -m flint_gpu.approval'],input=json.dumps(request),text=True,capture_output=True,check=True,timeout=120)
            result=json.loads(next(line[15:] for line in proc.stdout.splitlines() if line.startswith('FLINT_APPROVAL ')))
        else:
            previous=Path.cwd()
            try:result=handle(request)
            finally:os.chdir(previous)
        if result.get('error'):raise RuntimeError(result['error'])
        return result
    with Progress('Reading measurements'):view=call('review')
    if view['already_accepted']:
        show_tradeoff(view);return {**view,'accepted':True}
    if not view['approvable']:raise ValueError('This result cannot be approved as a memory tradeoff.')
    if yes:show_tradeoff(view)
    else:
        choice=confirm_tradeoff(view)
        if choice is not True:return choice
    with Progress('Approving'):result=call('approve',evidence_sha256=view['evidence_sha256'],remember=remember)
    if remember and remote:
        from .setup import save_settings
        from .config import read_config
        save_settings({'max_memory_growth':max(read_config()['max_memory_growth'],view['memory_growth'])})
    return result


def main():
    import sys
    try:result=handle(json.load(sys.stdin))
    except Exception as e:result={'error':type(e).__name__+': '+str(e)}
    print('FLINT_APPROVAL '+json.dumps(result))


if __name__=='__main__':main()
