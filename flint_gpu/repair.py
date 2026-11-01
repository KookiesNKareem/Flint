"""Model-written revisions driven by measured workload failures, without holdout feedback."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from .build import stage, publish, remote_dir, remote_prefix, status
import shlex


def repair(job, diagnosis, model, max_usd, max_tokens=24000):
    from kernel_contract.paper_agent import Run, JOB_SIZES, job_prompt, backward_prompt
    from kernel_contract.evidence import sha256
    job=Path(job);J=json.loads((job/'job.json').read_text());JOB_SIZES.update(J['sizes'])
    original=json.loads((job/'status.json').read_text())
    source=job/Path(original['kernel']).name
    if sha256(source)!=original['kernel_sha256']:raise ValueError('Prior candidate digest mismatch')
    rjob=stage(job,J)
    args=SimpleNamespace(model=model,effort='medium',max_tokens=max_tokens,max_usd=max_usd,
                         out=str(job/'repair'),remote=shlex.join(remote_prefix()),remote_dir=remote_dir(),
                         remote_job=rjob,job=str(job),layer='gdr',opt_rounds=0,bwd_opt_rounds=0)
    r=Run(args)
    r.messages=[{'role':'user','content':job_prompt(J)+'\n'+backward_prompt({
        'signature':J['signature'],'grad_names':J['grad_names'],'native':True,'backend':J.get('backend'),'memory_policy':J.get('memory_policy')})},
        {'role':'assistant','content':'```python\n'+source.read_text()+'\n```'}]
    prompt=('The implementation above passed numerical/native checks but the actual training command failed '
            'performance qualification. Diagnose and repair using this measured evidence:\n'+json.dumps(diagnosis,indent=2)+
            '\nThe full command must be faster by the existing margin AND must not increase measured GPU memory. '
            'If memory regresses, reduce retained activations using checkpoint/recomputation and avoid redundant workspace. '
            'If latency regresses, reduce launches, allocations, redundant computation or layout conversions. '
            'Measurements may include startup/compilation; do not claim the operator alone is the cause. '
            'Preserve equations, input/output dtypes and registered native autograd/operator APIs. '
            'Do not change the training script, checker, tolerances, budget, or benchmark; never special-case test data. '
            'Return the complete module. No human kernel edits will be made.')
    r.log(event='workload_feedback',model=model,diagnosis=diagnosis,prior_sha256=original['kernel_sha256'])
    status(job,state='repairing',reason=None)
    try:
        path=r.ask(prompt,'workload_repair');result=r.evaluate(path,True)
        path,result=r.until_correct(path,result,True,2,'workload repair')
        if result['blocking']:
            status(job,state='failed',reason='Development checks failed',repair_usd=r.cost);return
        events=[json.loads(x) for x in (r.out/'events.jsonl').read_text().splitlines()]
        last=next(e for e in reversed(events) if e['event']=='eval')
        done={'backward':str(path),'fwd_bwd_ms':result.get('ms'),'memory':result.get('memory'),
              'reference_memory':result.get('reference_memory'),'native_checks':result.get('native_checks')}
        r.log(event='done',ok=True,model=model,**done)
        models=[*original.get('models_used',[]),model]
        publish(job,J,rjob,done,last,original.get('usd',0)+r.cost,models)
        status(job,repair_usd=r.cost)
    except BaseException as error:
        status(job,state='failed',reason=type(error).__name__+': '+str(error),repair_usd=r.cost)
        if isinstance(error,KeyboardInterrupt):raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('job');ap.add_argument('diagnosis');ap.add_argument('--model',required=True)
    ap.add_argument('--max-usd',type=float,required=True);ap.add_argument('--max-tokens',type=int,default=24000)
    a=ap.parse_args();repair(a.job,json.loads(Path(a.diagnosis).read_text()),a.model,a.max_usd,a.max_tokens)


if __name__=='__main__':main()
