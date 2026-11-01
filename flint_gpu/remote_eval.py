"""GPU-host side of a build: judge a candidate against the user's own function on its observed contract, with
gradients, then time candidate and reference on the exact observed call. One JSON line out.

  python -m flint_gpu.remote_eval JOB_DIR CANDIDATE.py [--grads]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from kernel_contract.cases import build_cases
from kernel_contract.model import load_kernel
from .device import timeit as device_timeit
from kernel_contract.runner import run_contract
from kernel_contract.spec import OpSpec, materialize
from kernel_contract.tree import map_tree, descriptors
from .memory import clear_grads, loss_of_outputs, measure


def observed_inputs(spec: OpSpec, grads: bool):
    args, kwargs = materialize(spec, spec.sizes, 0, spec.flags.get('device','cuda'))
    if grads:
        args, kwargs = map_tree(lambda a: a.detach().requires_grad_(True) if isinstance(a, torch.Tensor) and a.is_floating_point() else a, (args, kwargs))
    return args, kwargs


def timeit(fn,**kwargs):
    from .device import available
    return device_timeit(fn,device=available(),**kwargs)


def step(fn, args, kwargs, grads: bool):
    def run():
        clear_grads(args, kwargs)
        out = fn(*args, **kwargs)
        if grads:
            loss_of_outputs(out).backward()
    return run


def main(job: str, path: str, grads: bool, holdout: str | None = None):
    job = Path(job)
    spec = OpSpec.from_json(json.loads((job / "spec.json").read_text()))
    spec.flags["check_grads"] = grads
    cases = json.loads(Path(holdout).read_text())["cases"] if holdout else build_cases(spec)
    domains = {t.get("domain") for t in descriptors(spec)}
    if not holdout and "logdecay" in domains:
        cases.append({"name": "compound", "axis": "values", "seed": 3, "layout": "transposed", "stress": "decay_strong"})
    results = list(run_contract(spec, Path(path), timeout=900, cases=cases,device=spec.flags.get('device','cuda')).results)
    extra_specs = json.loads((job / "extra_specs.json").read_text()) if (job / "extra_specs.json").exists() else []
    for i, extra in enumerate(extra_specs):
        es = OpSpec.from_json(extra)
        es.flags["check_grads"] = grads
        seeds = cases if holdout else build_cases(es)
        for r in run_contract(es, Path(path), timeout=900, cases=seeds,device=es.flags.get('device','cuda')).results:
            results.append({**r, "name": f"call{i + 2}/{r['name']}", "axis": "workload"})
    worst = max(results, key=lambda r: {"wrong": 3, "crash": 2, "refuse": 1}.get(r["outcome"], 0))["outcome"]
    out = {"outcome": worst if worst in ("wrong", "crash", "refuse") else "accept", "results": results}
    if all(r["outcome"] not in ("wrong", "crash") for r in results) and \
            any(r["name"] == "observed" and r["outcome"] == "accept" for r in results):
        try:
            kern = load_kernel(Path(path))
            args, kwargs = observed_inputs(spec, grads)
            out["ms"] = timeit(step(kern, args, kwargs, grads), reps=10, warm=2)
            ref_ms = job / ("ref_fwd_bwd_ms" if grads else "ref_fwd_ms")
            if not ref_ms.exists():
                ref_ms.write_text(str(timeit(step(spec.reference(), *observed_inputs(spec, grads), grads), reps=1, warm=1)))
            out["reference_ms"] = float(ref_ms.read_text())
            if grads:
                out['memory'] = measure(kern, args, kwargs)
                out['memory_by_signature'] = [out['memory']]
                for extra in extra_specs:
                    out['memory_by_signature'].append(measure(kern, *observed_inputs(OpSpec.from_json(extra), True)))
                out['memory'] = max(out['memory_by_signature'], key=lambda m: m.get('peak_extra_bytes',m.get('sampled_extra_bytes',0)))
                ref_memory = job / 'reference_memory.json'
                if not ref_memory.exists():
                    ref_memory.write_text(json.dumps(measure(spec.reference(), *observed_inputs(spec, True))))
                out['reference_memory'] = json.loads(ref_memory.read_text())
                budget = (spec.flags.get('memory_policy') or {}).get('max_peak_extra_bytes')
                if budget is not None and out['memory'].get('peak_extra_bytes',float('inf')) > budget:
                    results.append({'name': 'memory_budget', 'axis': 'resource', 'outcome': 'refuse',
                                    'reason': f"forward+backward peak extra allocated bytes {out['memory'].get('peak_extra_bytes','unknown')} exceeds budget {budget}; reduce retained states/workspace using checkpointing or recomputation"})
                    out['outcome'] = 'refuse'
                if spec.flags.get('native') and not holdout:
                    from .native_check import check
                    out['native_checks'] = check(spec, kern)
                    out['native_checks']['additional_signatures']=[check(OpSpec.from_json(extra),kern) for extra in extra_specs]
                    results.append({'name':'native_compile','axis':'compiler','outcome':'accept'})
        except Exception as e:
            results.append({"name": "observed/benchmark", "axis": "workload", "outcome": "refuse",
                            "reason": f"{type(e).__name__}: {e}"[:300]})
            out["outcome"] = "refuse"
    if spec.flags.get('device')=='mps' and not holdout:
        from .metal import argument_hint
        hint=argument_hint(Path(path).read_text())
        failed=next((r for r in results if r['outcome'] in ('wrong','refuse','crash')),None)
        if hint and failed is not None:
            failed['reason']=failed.get('reason','')+'\n'+hint
            out['api_diagnostic']=hint
    print("RESULT " + json.dumps(out, default=str), flush=True)


def time_reference(job: str):
    spec = OpSpec.from_json(json.loads((Path(job) / "spec.json").read_text()))
    args, kwargs = observed_inputs(spec, False)
    print(f"REFERENCE_MS {timeit(step(spec.reference(), args, kwargs, False), reps=1, warm=1)}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--reference":
        time_reference(sys.argv[2])
    else:
        main(sys.argv[1], sys.argv[2], grads="--grads" in sys.argv,
             holdout=sys.argv[sys.argv.index("--holdout")+1] if "--holdout" in sys.argv else None)
