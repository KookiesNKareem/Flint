"""User defaults, project overrides, then environment overrides. No credentials in project files."""
from pathlib import Path
import os
try:
    import tomllib
except ImportError:
    import tomli as tomllib

DEFAULTS = {
    'native': True,
    'models': ['meta-models/muse-spark-open-1026'],
    'backward_model': 'muse-wonder-1.0',
    'max_usd': 4.0,
    'launch_policy': 'background',
    'budget_ledger': str(Path.home()/'.cache/flint/budget.json'),
    'optimization_minutes': 20,
    'rehearsal_steps': 20,
    'setup_complete': False,
    'max_memory_growth': 0.0,
    'review_memory_tradeoffs': True,
    'observe_calls': 2,
    'workload_kind': 'training',
    'replay_safe': False,
    'observation_max_units': 64,
    'observation_seconds': 30.0,
    'infer_domains': False,
    'memory_budget_mb': None,
    'memory_opt_rounds': 1,
    'cache_dir': str(Path.home()/'.cache/flint'),
    'worker_dir': str(Path.home()/'.cache/flint/worker'),
    'remote': 'bash -c',
    'auto_build': True,
    'watch_interval': 2.0,
    'mode': 'auto',
}


def read_config(cwd=None):
    config = dict(DEFAULTS)
    user = Path(os.environ.get('FLINT_CONFIG', os.environ.get('MUSE_CONFIG', Path.home()/'.config/flint/config.toml'))).expanduser()
    if user.is_file():
        tables = tomllib.loads(user.read_text())
        config.update(tables.get('muse', {})); config.update(tables.get('flint', {}))
    personal_cap=config['max_usd'];personal_ledger=config['budget_ledger'];setup_complete=config['setup_complete']
    directory = Path(cwd or Path.cwd()).resolve()
    for parent in (directory, *directory.parents):
        path = parent/'pyproject.toml'
        if path.is_file():
            tables = tomllib.loads(path.read_text()).get('tool', {})
            config.update(tables.get('muse', {})); config.update(tables.get('flint', {}))
            break
    # A project may tighten the personal budget, but cannot reset or enlarge it.
    config['max_usd']=min(float(config['max_usd']),float(personal_cap))
    config['budget_ledger']=personal_ledger
    config['setup_complete']=setup_complete
    for env, field in [('MUSE_KERNEL_CACHE','cache_dir'),('MUSE_KERNEL_REMOTE','remote'),
                       ('MUSE_KERNEL_REMOTE_DIR','worker_dir'),('MUSE_MODE','mode')]:
        if env in os.environ: config[field] = os.environ[env]
    if 'MUSE_KERNEL_MODEL' in os.environ: config['models'] = os.environ['MUSE_KERNEL_MODEL'].split(',')
    if 'MUSE_KERNEL_MAX_USD' in os.environ: config['max_usd'] = float(os.environ['MUSE_KERNEL_MAX_USD'])
    for env, field in [('FLINT_CACHE','cache_dir'),('FLINT_REMOTE','remote'),
                       ('FLINT_WORKER_DIR','worker_dir'),('FLINT_MODE','mode')]:
        if env in os.environ:config[field]=os.environ[env]
    if 'FLINT_MAX_MEMORY_GROWTH' in os.environ:config['max_memory_growth']=float(os.environ['FLINT_MAX_MEMORY_GROWTH'])
    if 'FLINT_MODEL' in os.environ:config['models']=os.environ['FLINT_MODEL'].split(',')
    if 'FLINT_BACKWARD_MODEL' in os.environ:config['backward_model']=os.environ['FLINT_BACKWARD_MODEL']
    if 'FLINT_MAX_USD' in os.environ:config['max_usd']=float(os.environ['FLINT_MAX_USD'])
    if config['launch_policy'] not in ('observe','background','prepared','adaptive'):
        raise ValueError('Flint launch_policy must be observe, background, prepared, or adaptive')
    if config['mode'] not in ('auto','reference','capture','offline','benchmark'):
        raise ValueError('Flint mode must be auto, reference, capture, offline, or benchmark')
    if not 0 < float(config['max_usd']) <= 100:
        raise ValueError('Flint max_usd must be positive and <= 100')
    if not isinstance(config['models'], list) or not config['models']:
        raise ValueError('Flint models must be a nonempty list')
    if config['workload_kind'] not in ('training','batch'):raise ValueError('workload_kind must be training or batch')
    import math
    if not isinstance(config['max_memory_growth'],(int,float)) or not math.isfinite(config['max_memory_growth']) or config['max_memory_growth']<0:
        raise ValueError('Flint max_memory_growth must be a finite nonnegative fraction')
    if not isinstance(config['review_memory_tradeoffs'],bool):raise ValueError('review_memory_tradeoffs must be a boolean')
    if not isinstance(config['replay_safe'],bool):raise ValueError('replay_safe must be a boolean')
    config['cache_dir'] = str(Path(config['cache_dir']).expanduser().resolve())
    config['worker_dir'] = str(Path(config['worker_dir']).expanduser().resolve())
    config['budget_ledger']=str(Path(config['budget_ledger']).expanduser().resolve())
    return config
