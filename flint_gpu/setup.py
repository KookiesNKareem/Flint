"""One-time personal preferences; provider secrets never enter project configuration."""
import getpass
import json
import os
from pathlib import Path
import re
import sys
from .config import read_config


def config_path():
    return Path(os.environ.get('FLINT_CONFIG',os.environ.get('MUSE_CONFIG',Path.home()/'.config/flint/config.toml'))).expanduser()


def save_settings(values,path=None):
    path=Path(path or config_path());path.parent.mkdir(parents=True,exist_ok=True)
    lines=path.read_text().splitlines(True) if path.exists() else []
    start=next((i+1 for i,s in enumerate(lines) if re.match(r'^\s*\[flint\]\s*(?:#.*)?$',s)),None)
    if start is None:
        if lines and not lines[-1].endswith('\n'):lines[-1]+='\n'
        lines.extend(['\n[flint]\n']);start=len(lines)
    end=next((i for i in range(start,len(lines)) if lines[i].lstrip().startswith('[')),len(lines))
    for key,value in values.items():
        encoded=json.dumps(value) if not isinstance(value,Path) else json.dumps(str(value))
        line=f'{key} = {encoded}\n'
        index=next((i for i in range(start,end) if re.match(r'^\s*'+re.escape(key)+r'\s*=',lines[i])),None)
        if index is None:lines.insert(end,line);end+=1
        else:lines[index]=line
    temporary=path.with_suffix('.tmp')
    with temporary.open('w') as f:f.write(''.join(lines))
    temporary.chmod(0o600);temporary.replace(path)
    return path


def credential_path():return config_path().with_name('providers.json')


def save_credentials(values):
    path=credential_path();path.parent.mkdir(parents=True,exist_ok=True)
    old=json.loads(path.read_text()) if path.exists() else {}
    old.update({k:v for k,v in values.items() if k in ('MODEL_API_KEY','BASETEN_API_KEY') and v})
    if not old:return
    temporary=path.with_suffix('.tmp')
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    os.fchmod(fd,0o600)
    with os.fdopen(fd,'w') as f:json.dump(old,f)
    temporary.chmod(0o600);temporary.replace(path)


def provider_environment(base=None):
    env=dict(os.environ if base is None else base);path=credential_path()
    if path.exists():
        if path.stat().st_mode & 0o077:raise PermissionError('Flint provider file must have permissions 0600: '+str(path))
        for key,value in json.loads(path.read_text()).items():
            if key in ('MODEL_API_KEY','BASETEN_API_KEY'):env.setdefault(key,value)
    return env


def setup(policy=None,max_usd=None,ledger=None,remote=None,save_keys=False,interactive=None,memory_growth_percent=None):
    config=read_config();interactive=sys.stdin.isatty() if interactive is None else interactive
    if interactive:
        if policy is None:policy=input(f'Launch policy [adaptive/background/prepared/observe] ({config["launch_policy"]}): ').strip() or config['launch_policy']
        if max_usd is None:max_usd=float(input(f'Shared model spending cap, USD ({config["max_usd"]:g}): ').strip() or config['max_usd'])
    policy=policy or config['launch_policy'];max_usd=config['max_usd'] if max_usd is None else max_usd
    if policy not in ('background','prepared','observe','adaptive') or not 0<float(max_usd)<=100:raise ValueError('Choose a valid policy and a model cap above $0 and at most $100')
    values={'launch_policy':policy,'max_usd':float(max_usd),'budget_ledger':str(Path(ledger or config['budget_ledger']).expanduser().resolve()),
            'auto_build':False,'setup_complete':True}
    if memory_growth_percent is not None:
        import math
        if not math.isfinite(memory_growth_percent) or memory_growth_percent<0:raise ValueError('Memory growth must be a finite nonnegative percentage')
        values['max_memory_growth']=memory_growth_percent/100
    if remote is not None:values['remote']=remote
    if interactive or save_keys:
        keys={}
        existing=provider_environment()
        for name in ('BASETEN_API_KEY','MODEL_API_KEY'):
            value=existing.get(name)
            if not value and interactive:value=getpass.getpass(f'{name} (optional; Enter to configure later): ')
            if value:keys[name]=value
        save_credentials(keys)
    path=save_settings(values)
    print(f'Saved {policy} policy and ${float(max_usd):g} shared model cap. Existing spending is retained.\n{path}',flush=True)
    return values
