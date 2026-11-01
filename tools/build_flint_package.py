"""Assemble a standalone sdist and wheel from the tested canonical runtime/checker sources.

Internal checker and provider modules receive private package names. Consumers do not
install KernelAgent, add its checkout to PYTHONPATH, or import its generic utils package.
The resulting sdist contains every source and builds with ordinary PEP 517 tooling.
"""
from pathlib import Path
import argparse
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ['__init__','agent','cases','evidence','holdout','model','paper_agent','runner','spec','tree','worker','recur_bench']


def stage(destination):
    dest = Path(destination).resolve()
    dest.mkdir(parents=True, exist_ok=False)
    # The published sdist already has its private modules in place.
    # Only its build-system table differs from the source checkout.
    project = (ROOT / 'pyproject.toml').read_text().split('[project]', 1)[1]
    (dest / 'pyproject.toml').write_text(
        '[build-system]\nrequires = ["setuptools>=68", "wheel"]\n'
        'build-backend = "setuptools.build_meta"\n\n[project]' + project
    )
    shutil.copy2(ROOT/'README.md', dest/'README.md')
    shutil.copy2(ROOT/'LICENSE', dest/'LICENSE')
    groups = [(ROOT/'flint_gpu', dest/'flint_gpu', None),
              (ROOT/'muse_kernel', dest/'muse_kernel', None),
              (ROOT/'kernel_contract', dest/'flint_gpu/_contract', CONTRACT),
              (ROOT/'utils/providers', dest/'flint_gpu/_providers', None)]
    for source, target, names in groups:
        target.mkdir(parents=True, exist_ok=True)
        for path in source.glob('*.py'):
            if names is not None and path.stem not in names: continue
            text=path.read_text().replace('kernel_contract', 'flint_gpu._contract').replace('utils.providers', 'flint_gpu._providers')
            compile(text,str(target/path.name),'exec')
            (target/path.name).write_text(text)
    return dest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage', required=True);ap.add_argument('--dist')
    a=ap.parse_args();dest=stage(a.stage)
    if a.dist:
        subprocess.run([sys.executable,'-m','build',str(dest),'--outdir',str(Path(a.dist).resolve())],check=True)
    print(dest)

if __name__=='__main__':main()
