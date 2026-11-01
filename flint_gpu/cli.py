"""Small terminal interface for configuration, evidence, builds and reproducible pins."""
import argparse
import json
import os
from pathlib import Path
from .config import read_config


def artifacts(name=None):
    root=Path(read_config()['cache_dir'])
    result=[]
    for path in sorted(root.glob('*/status.json')):
        if name and name not in path.parent.name: continue
        result.append({'path':str(path.parent), **json.loads(path.read_text())})
    return result


def main():
    ap=argparse.ArgumentParser(prog='flint')
    sub=ap.add_subparsers(dest='command',required=True)
    setup=sub.add_parser('setup',help='Save personal launch policy, budget and provider setup once')
    setup.add_argument('--memory-growth-percent',type=float,help='Save the allowed peak-memory increase for faster checked implementations')
    setup.add_argument('--policy',choices=['observe','background','prepared','adaptive']);setup.add_argument('--max-usd',type=float)
    setup.add_argument('--budget-ledger');setup.add_argument('--remote');setup.add_argument('--save-credentials',action='store_true')
    inspect=sub.add_parser('inspect');inspect.add_argument('name',nargs='?');inspect.add_argument('--json',action='store_true')
    build=sub.add_parser('build');build.add_argument('job')
    recheck=sub.add_parser('revalidate');recheck.add_argument('job')
    freeze=sub.add_parser('freeze');freeze.add_argument('--out',default='flint.lock')
    config=sub.add_parser('config');config.add_argument('--init',action='store_true');config.add_argument('--path')
    qualify=sub.add_parser('qualify')
    qualify.add_argument('--repeats',type=int,default=3)
    qualify.add_argument('--timeout',type=float,default=300)
    qualify.add_argument('--minimum-gain',type=float,default=.05)
    qualify.add_argument('--input',action='append',default=[])
    qualify.add_argument('training_command',nargs=argparse.REMAINDER)
    optimize=sub.add_parser('optimize')
    optimize.add_argument('--model',help='Use only this model for new forward, backward and repairs')
    optimize.add_argument('--project',help='Training project directory on the worker (default: current directory)')
    optimize.add_argument('--remote',help='Configured worker command, e.g. an SSH wrapper')
    optimize.add_argument('--out',help='New local evidence directory')
    optimize.add_argument('--budget-ledger',help='Persistent ledger shared across models, projects and attempts')
    optimize.add_argument('--max-usd',type=float)
    optimize.add_argument('--minutes',type=float)
    optimize.add_argument('--max-rounds',type=int,default=2)
    optimize.add_argument('--repeats',type=int,default=3)
    optimize.add_argument('--timeout',type=float,default=300)
    optimize.add_argument('--minimum-gain',type=float,default=.05)
    optimize.add_argument('--rehearsal-steps',type=int)
    optimize.add_argument('--input',action='append',default=[])
    optimize.add_argument('training_command',nargs=argparse.REMAINDER)
    run=sub.add_parser('run',help='Observe, optimize after a research run, or prepare before launch')
    run.add_argument('--model',help='Use only this model for new forward, backward and repairs')
    run.add_argument('--policy',choices=['observe','background','prepared','adaptive'],default=None)
    run.add_argument('--project');run.add_argument('--remote');run.add_argument('--out')
    run.add_argument('--budget-ledger');run.add_argument('--max-usd',type=float)
    run.add_argument('--minutes',type=float);run.add_argument('--rehearsal-steps',type=int)
    run.add_argument('--warmup',type=int,default=3);run.add_argument('--steps',type=int,default=5)
    run.add_argument('--minimum-share',type=float,default=.10)
    run.add_argument('--timeout',type=float,default=300);run.add_argument('--run-timeout',type=float)
    run.add_argument('--max-rounds',type=int,default=2);run.add_argument('--input',action='append',default=[])
    run.add_argument('--json',action='store_true',help='Print the machine-readable outcome')
    run.add_argument('--prepare-only',action='store_true');run.add_argument('training_command',nargs=argparse.REMAINDER)
    apply=sub.add_parser('apply',help='Apply a reviewed, unchanged source proposal')
    apply.add_argument('proposal');apply.add_argument('--project');apply.add_argument('--remote')
    status=sub.add_parser('status',help='Read a workload/background task state');status.add_argument('task')
    cancel=sub.add_parser('cancel',help='Stop a background controller and revoke unfinished qualification');cancel.add_argument('task')
    approve=sub.add_parser('approve',help='Review and approve a measured memory tradeoff')
    approve.add_argument('--yes',action='store_true');approve.add_argument('--remember',action='store_true')
    approve.add_argument('--project');approve.add_argument('--remote')
    args=ap.parse_args()
    if args.command=='setup':
        from .setup import setup
        setup(args.policy,args.max_usd,args.budget_ledger,args.remote,args.save_credentials,memory_growth_percent=args.memory_growth_percent)
    elif args.command=='approve':
        from .approval import review_on_terminal
        result=review_on_terminal(args.remote,args.project,yes=args.yes,remember=args.remember)
        if result is None:raise SystemExit('Approval is pending. Run flint approve in a terminal, or use --yes after reviewing the evidence.')
        if result is False:print('Kept the reference.');return
        print('Already approved.' if result.get('already_accepted') else 'Approved. Your next normal run can use this kernel.')
        if args.remember and result.get('remembered'):print(f"Saved a {result['memory_growth']:.2%} memory allowance.")
    elif args.command=='run':
        if args.json:os.environ['FLINT_INTERACTIVE']='0'
        import sys
        if args.policy is None and not read_config()['setup_complete']:
            if not sys.stdin.isatty():ap.error('Run flint setup once before using the saved launch policy')
            from .setup import setup
            setup()
        from .workflow import run
        command=args.training_command
        if command and command[0]=='--':command=command[1:]
        result=run(command,args.policy,args.project,args.remote,args.out,args.budget_ledger,args.max_usd,
                   args.minutes,args.rehearsal_steps,args.warmup,args.steps,args.minimum_share,args.timeout,
                   args.run_timeout,args.input,args.max_rounds,args.prepare_only,model=args.model)
        if args.json:print(json.dumps({k:v for k,v in result.items() if k!='report'},indent=2))
        elif result['state']=='optimization_started':print('Optimizing for your next run.',flush=True)
        elif result.get('reason'):print(result['reason'],flush=True)
        if result['state'] in ('not_ready','training_failed','review_required') or (result.get('report') and not result['report'].get('training_success')):
            raise SystemExit(1)
    elif args.command=='apply':
        from .workflow import rpc
        proposal=json.loads(Path(args.proposal).read_text());config=read_config()
        folder=Path.cwd()/'.flint';folder.mkdir(exist_ok=True)
        print(json.dumps(rpc(args.remote or config['remote'],{'action':'apply','project':args.project or proposal['root'],
                                                            'proposal':proposal},folder/'apply.log'),indent=2))
    elif args.command in ('status','cancel'):
        path=Path(args.task);state=json.loads((path/'state.json').read_text())
        if args.command=='cancel' and state.get('state') in ('queued','optimizing'):
            import signal
            # Only signal the controller whose command line contains this exact task request.
            import subprocess
            pid=state.get('pid') or json.loads((path/'background-pid.json').read_text())['pid'];actual=subprocess.run(['ps','-p',str(pid),'-o','args='],capture_output=True,text=True).stdout
            if str((path/'request.json').resolve()) not in actual:raise RuntimeError('Controller identity no longer matches; no process signalled')
            os.kill(pid,signal.SIGTERM);print('Cancellation requested')
        else:print(json.dumps(state,indent=2))
    elif args.command=='optimize':
        from .optimization import optimize
        command=args.training_command
        if command and command[0]=='--':command=command[1:]
        result=optimize(command,args.project,args.remote,args.out,args.budget_ledger,args.max_usd,args.minutes,
                        args.repeats,args.timeout,args.minimum_gain,args.input,args.max_rounds,args.rehearsal_steps,model=args.model)
        print(json.dumps(result,indent=2))
    elif args.command=='qualify':
        from .qualification import qualify
        command=args.training_command
        if command and command[0]=='--':command=command[1:]
        qualify(command,args.repeats,args.timeout,args.minimum_gain,args.input)
    elif args.command=='inspect':
        results=artifacts(args.name)
        if args.json: print(json.dumps(results,indent=2))
        else:
            for r in results:
                print(f"{Path(r['path']).name}: {r['state']}" +
                      (f" · {r.get('fwd_bwd_ms',0):.3f} ms · ${r.get('usd',0):.2f}" if r['state']=='ready' else ''))
                if r['state']=='ready': print('  Correctness verified; activation requires a matching workload qualification.')
                if r.get('reason'): print('  '+r['reason'])
    elif args.command=='build':
        config=read_config()
        os.environ.setdefault('MUSE_KERNEL_REMOTE',config['remote'])
        os.environ.setdefault('MUSE_KERNEL_REMOTE_DIR',config['worker_dir'])
        from .build import main as build
        build(args.job)
        if json.loads((Path(args.job)/'status.json').read_text())['state']!='ready': raise SystemExit(1)
    elif args.command=='revalidate':
        from .revalidate import revalidate
        print(json.dumps(revalidate(args.job),indent=2))
    elif args.command=='freeze':
        entries={Path(r['path']).name:{'kernel_sha256':r['kernel_sha256']} for r in artifacts() if r['state']=='ready'}
        Path(args.out).write_text(json.dumps({'version':1,'artifacts':entries},indent=2)+'\n')
        print(f'Pinned {len(entries)} artifacts in {args.out}')
    elif args.init:
        path=Path(args.path or Path.home()/'.config/flint/config.toml')
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as f:
            f.write('[flint]\nmodels = ["meta-models/muse-spark-open-1026"]\nbackward_model = "muse-wonder-1.0"\nmax_usd = 4.0\nnative = true\n')
        print(path)
    else: print(json.dumps(read_config(),indent=2))

if __name__=='__main__':main()
