"""Coordinate Flint's own profiling/training/optimization on a configured worker."""
import json
from pathlib import Path
import time
from .config import read_config


def lease_path():
    root=Path(read_config()['worker_dir']);root.mkdir(parents=True,exist_ok=True)
    return root/'flint-worker-owner.json'


def acquire(owner,kind):
    path=lease_path()
    try:
        with path.open('x') as f:json.dump({'owner':owner,'kind':kind,'started':time.time()},f)
    except FileExistsError:
        raise RuntimeError('Flint worker is busy: '+path.read_text()+'; wait for completion or cancel its controller') from None


def release(owner):
    path=lease_path()
    if path.exists() and json.loads(path.read_text())['owner']==owner:path.unlink()
