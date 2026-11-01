"""No-model Metal integration proof using an explicitly hand-authored test artifact.

Run with the installed wheel, outside the source checkout. Writes a new isolated project,
checks the fixture using development and fresh held-out cases, compares the normal training
command, and leaves a faster result pending for the user's terminal memory review.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=60)
    ap.add_argument('--length',type=int,default=8)
    ap.add_argument('--eager',action='store_true')
    ap.add_argument('--exercise-approval',action='store_true',help='Test approval and normal dispatch, then restore pending review')
    args=ap.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    project=out/'project';shutil.copytree(ROOT/'examples/metal',project,ignore=shutil.ignore_patterns('__pycache__','.flint'))
    cache=out/'cache';cache.mkdir()
    # Personal config and artifact cache sit outside the qualified source tree.
    config=out/'config.toml'
    config.write_text('[flint]\nsetup_complete = true\nauto_build = false\nlaunch_policy = "prepared"\nremote = "bash -c"\n'
                      +'cache_dir = '+json.dumps(str(cache))+'\nworker_dir = '+json.dumps(str(out/'worker'))+'\n')
    os.environ['FLINT_CONFIG']=str(config);os.environ['FLINT_CACHE']=str(cache)
    os.environ['FLINT_EXTERNAL_BUILDER']='1'
    command=['train.py','--steps',str(args.steps),'--length',str(args.length)]+(['--eager'] if args.eager else [])
    report=out/'capture.json'
    env={**os.environ,'FLINT_MODE':'capture','FLINT_LAUNCH_REPORT':str(report)}
    with (out/'capture.log').open('w') as log:
        subprocess.run([sys.executable,'-m','flint_gpu.launch',*command],cwd=project,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    capture=json.loads(report.read_text());assert capture['success']
    jobs=[Path(d['job']) for d in capture['dispatch']]
    assert len(jobs)==1
    job=jobs[0]
    shutil.copy2(ROOT/'tests/package/fixtures/metal_recurrence.py',job/'kernel.py')
    from flint_gpu.qualification import sha
    from flint_gpu.revalidate import revalidate
    from flint_gpu.identity import environment
    from flint_gpu._contract.holdout import holdout_cases
    state={'state':'candidate','kernel':'kernel.py','kernel_sha256':sha(job/'kernel.py'),
           'models_used':[],'provenance':'hand-authored Metal integration fixture','usd':0}
    (job/'status.json').write_text(json.dumps(state))
    protocol=holdout_cases(None);protocol['candidate_sha256']=state['kernel_sha256']
    (job/'holdout_protocol.json').write_text(json.dumps(protocol))
    print('Checking the fixture and fresh holdouts…',flush=True)
    checked=revalidate(job)
    assert checked['compile_passed'] and checked['holdout_passed']
    os.chdir(project)
    from flint_gpu.qualification import qualify
    receipt=qualify(command,require_unit_gain=True,timeout=300)
    result={'environment':environment(),'provenance':state['provenance'],'model_calls':0,
            'job':str(job),'command':command,'checked':checked,'qualification':receipt}
    if args.exercise_approval and receipt.get('approval_available'):
        from flint_gpu.approval import approve,evidence_digest
        try:
            result['test_approval']=approve(evidence_digest(receipt))
            result['approval_scope']='isolated integration test; pending receipt restored afterward'
            normal_report=out/'normal.json'
            env={**os.environ,'FLINT_MODE':'auto','FLINT_REQUIRE_QUALIFIED':'1','FLINT_LAUNCH_REPORT':str(normal_report)}
            with (out/'normal.log').open('w') as log:
                subprocess.run([sys.executable,'-m','flint_gpu.launch',*command],cwd=project,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
            result['normal']=json.loads(normal_report.read_text())
            assert result['normal']['success'] and result['normal']['artifacts']==receipt['pairs'][0]['benchmark']['artifacts']
            assert all(d['performance_qualified'] for d in result['normal']['dispatch'])
        finally:(project/'.flint/qualification.json').write_text(json.dumps(receipt,indent=2))
    (out/'result.json').write_text(json.dumps(result,indent=2))
    print('Saved '+str(out/'result.json'))
    print('To review and run:' if receipt.get('approval_available') else 'Candidate retained as unqualified. To rerun this example:')
    import shlex
    print('cd '+shlex.quote(str(project)))
    print('FLINT_CONFIG='+shlex.quote(str(config))+' flint run -- '+shlex.join(command))

if __name__=='__main__':main()
