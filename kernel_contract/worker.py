"""Subprocess side: run cases against one kernel, one JSON line per case.

  python -m kernel_contract.worker SPEC_JSON KERNEL_PY CASES_JSON OUT_JSONL START DEVICE

A wrong kernel can poison the CUDA context, so on a device error the worker writes a crash
line and exits; the parent restarts it at the next case.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import traceback

import torch
from torch.overrides import TorchFunctionMode

from .cases import VALUE_TRANSFORMS
from .spec import OpSpec, materialize
from .tree import leaves, map_tree, structure, at

def error_text(message):
    text = str(message)
    return text if len(text) <= 1400 else text[:500] + '\n…\n' + text[-850:]


TOL = {torch.bfloat16: (1e-2, 1e-2), torch.float16: (1e-3, 1e-3), torch.float32: (1e-4, 1e-4),
       torch.float64: (1e-7, 1e-7)}
DEVICE_ERRORS = ("CUDA error", "illegal memory access", "device-side assert", "misaligned address",
                 "unspecified launch failure")


def load_kernel(path: str):
    spec = importlib.util.spec_from_file_location("candidate_kernel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.kernel_function


def first_tensor(out):
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    raise TypeError(f"kernel returned {type(out).__name__}, expected a tensor")


def to_f64(x):
    return map_tree(lambda value: (value.cpu() if value.device.type=='mps' else value).double()
                    if isinstance(value, torch.Tensor) and value.is_floating_point()
                    else value.cpu() if isinstance(value,torch.Tensor) and value.device.type=='mps' else value, x)


def compare(out: torch.Tensor, ref: torch.Tensor, truth: torch.Tensor, any_float: bool = False,
            normwise: float | None = None) -> dict:
    if tuple(out.shape) != tuple(ref.shape):
        return {"outcome": "wrong", "reason": f"shape {tuple(out.shape)} != {tuple(ref.shape)}"}
    if out.dtype != ref.dtype and not (any_float and out.is_floating_point() and ref.is_floating_point()):
        return {"outcome": "wrong", "reason": f"dtype {out.dtype} != {ref.dtype}"}
    tol_dtype = out.dtype if TOL.get(out.dtype, (0,))[0] > TOL.get(ref.dtype, (0,))[0] else ref.dtype
    target='cpu' if out.device.type=='mps' else out.device
    o,r,t=(x.to(target).double() for x in (out,ref,truth))
    fin_t = torch.isfinite(t)
    if not torch.equal(torch.isfinite(o), fin_t):
        bad = int((torch.isfinite(o) != fin_t).sum())
        return {"outcome": "wrong", "reason": f"{bad} non-finite mismatches"}
    if not fin_t.any():
        return {"outcome": "accept", "err": 0.0, "torch_err": 0.0, "scale": 0.0}
    o, r, t = o[fin_t], r[fin_t], t[fin_t]
    atol, rtol = TOL.get(tol_dtype, (1e-2, 1e-2))
    scale = float(t.abs().max())
    err = float((o - t).abs().max())
    torch_err = float((r - t).abs().max())
    if normwise is not None:
        # long reductions in low precision: bound the error relative to the output's scale, not per element
        within = err <= normwise * scale
    else:
        within = bool(((o - t).abs() <= atol * min(1.0, scale) + rtol * t.abs()).all())
    ok = within or (torch_err > 0 and err <= 2 * torch_err)
    return {"outcome": "accept" if ok else "wrong", "err": err, "torch_err": torch_err, "scale": scale}


def run_case(spec: OpSpec, kernel, case: dict, device: str) -> dict:
    sizes = {**spec.sizes, **case.get("sizes", {})}
    vf = VALUE_TRANSFORMS.get(case.get("transform", ""))

    def fresh():
        # rebuilt from the seed for every call: a clone would not keep a gapped view's strides
        return materialize(spec, sizes, case["seed"], device, layout=case.get("layout"), value_fn=vf,
                           stress=case.get("stress"))

    args, kwargs = fresh()
    fn = spec.reference()
    try:
        ref = fn(*args, **kwargs)
        structure(ref)
    except Exception as e:
        return {"outcome": "skip", "reason": f"reference raised {type(e).__name__}: {e}"[:300]}
    args, kwargs = fresh()
    try:
        with KeepFloat64():
            truth = fn(*to_f64(args), **to_f64(kwargs))
    except Exception as error:
        if device.startswith('mps'):return {'outcome':'refuse','reason':'CPU float64 oracle unavailable: '+error_text(error)}
        truth = fn(*map_tree(lambda a: a.float() if isinstance(a, torch.Tensor) and a.is_floating_point() else a, args), **kwargs)
    try:
        kargs, kkwargs = fresh()
        snapshots = [(path, value.detach().clone()) for path, value in leaves((kargs, kkwargs))
                     if isinstance(value, torch.Tensor)]
        out = kernel(*kargs, **kkwargs)
        if device.startswith('mps'):torch.mps.synchronize()
        elif device.startswith('cuda'):torch.cuda.synchronize()
        if any(not torch.equal(at((kargs, kkwargs), path), value) for path, value in snapshots):
            return {'outcome': 'wrong', 'reason': 'candidate mutated an input tensor'}
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if any(s in msg for s in DEVICE_ERRORS):
            return {"outcome": "crash", "reason": error_text(msg)}
        return {"outcome": "refuse", "reason": error_text(msg)}
    try:
        if structure(out) != structure(ref):
            return {"outcome": "wrong", "reason": "output tensor-tree structure differs from reference"}
        checks = []
        for path, value in leaves(ref):
            check = compare(at(out, path), value, at(truth, path),
                            any_float=spec.flags.get("any_float_output", False) and isinstance(ref, torch.Tensor),
                            normwise=spec.flags.get("normwise_rtol"))
            check["output_path"] = list(path)
            check["truth_dtype"] = str(at(truth, path).dtype)
            checks.append(check)
            if check['outcome'] != 'accept':
                check['reason'] = f"output {path}: " + check.get('reason', f"error {check.get('err')}")
                return check
    except (TypeError, KeyError, IndexError) as error:
        return {"outcome": "wrong", "reason": f"output structure: {error}"}
    fwd = {**checks[0], 'outputs_checked': len(checks), 'output_checks': checks}
    if not spec.flags.get("check_grads"):
        return fwd
    return check_grads(spec, kernel, fresh, fn, case, device, fwd)



class KeepFloat64(TorchFunctionMode):
    """While computing the float64 truth, a reference's own casts to float32 (x.float(), x.to(torch.float32))
    keep float64, so user code written for fp32 still yields a float64 oracle."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in (torch.Tensor.float, torch.Tensor.to, torch.Tensor.type) and args and \
                isinstance(args[0], torch.Tensor) and args[0].dtype == torch.float64:
            target = kwargs.get("dtype", next((a for a in args[1:] if isinstance(a, torch.dtype)), None))
            if func is torch.Tensor.float or target == torch.float32 or target == "torch.FloatTensor":
                return args[0]
        return func(*args, **kwargs)


def check_grads(spec, kernel, fresh, fn, case, device, fwd: dict) -> dict:
    """Check every input gradient, including isolated cotangents for every returned state."""
    rtol = spec.flags.get("normwise_rtol", 2e-2)
    probe_args, probe_kwargs = fresh()
    ref_outputs = fn(*probe_args, **probe_kwargs)
    paths = [path for path, value in leaves(ref_outputs) if value.is_floating_point()]
    objectives = [None] + (paths if len(paths) > 1 else [])
    def evaluate(function, double, objective, weights):
        args, kwargs = fresh()
        if double and device.startswith('mps'):args,kwargs=map_tree(lambda x:x.cpu() if isinstance(x,torch.Tensor) else x,(args,kwargs))
        inputs = map_tree(lambda x: (x.double() if double else x).detach().requires_grad_(True)
                          if isinstance(x, torch.Tensor) and x.is_floating_point() else x,
                          {'args': args, 'kwargs': kwargs})
        indexed = [(path, value) for path, value in leaves(inputs)
                   if isinstance(value, torch.Tensor) and value.is_floating_point()]
        outputs = function(*inputs['args'], **inputs['kwargs'])
        def weighted(path):
            value=at(outputs,path);weight=weights[path].to(value.device)
            return (value.float()*weight.float()).sum() if value.device.type=='mps' else (value.double()*weight.double()).sum()
        terms = [weighted(path)
                 for path in paths if objective is None or path == objective]
        loss = sum(terms)
        grads = torch.autograd.grad(loss, [value for _, value in indexed], allow_unused=True) \
            if isinstance(loss, torch.Tensor) and loss.requires_grad else [None] * len(indexed)
        return indexed, grads
    gradient_checks = []
    for objective in objectives:
        gen = torch.Generator(device='cpu').manual_seed(1234 + case['seed'])
        weights = {path: torch.randn(at(ref_outputs, path).shape, generator=gen).to(device) for path in paths}
        try:
            indexed, grads = evaluate(kernel, False, objective, weights)
            if device.startswith('mps'):torch.mps.synchronize()
            elif device.startswith('cuda'):torch.cuda.synchronize()
        except Exception as error:
            msg = f"{type(error).__name__}: {error}"
            return {'outcome': 'crash' if any(s in msg for s in DEVICE_ERRORS) else 'refuse',
                    'reason': error_text('backward: ' + msg)}
        try:
            with KeepFloat64():truth_inputs,truth_grads=evaluate(fn,True,objective,weights)
        except Exception as error:
            return {'outcome':'refuse','reason':'Float64 gradient oracle unavailable: '+error_text(error)}
        bounded = spec.flags.get('reference_bounded_gradients', False)
        native_grads = evaluate(fn, False, objective, weights)[1] if bounded else [None] * len(grads)
        for (path, value), (_, truth_value), actual, truth, native in zip(indexed, truth_inputs, grads, truth_grads, native_grads):
            if truth is None:
                truth = torch.zeros_like(truth_value)
                actual = torch.zeros_like(value) if actual is None else actual
            prefix = f"gradient of argument {path} (output {objective or 'joint'})"
            if actual is None:
                return {'outcome': 'wrong', 'reason': prefix + ': no gradient'}
            if actual.shape != truth.shape or not torch.isfinite(actual).all():
                return {'outcome': 'wrong', 'reason': prefix + ': wrong shape or nonfinite gradient'}
            scale = float(truth.abs().max()) if truth.numel() else 0.
            error = float((actual.to(truth.device).double() - truth.double()).abs().max()) if truth.numel() else 0.
            relative_limit = rtol * max(scale, 1e-30)
            native_error = None
            if bounded:
                native = torch.zeros_like(truth) if native is None else native
                native_error = float((native.to(truth.device).double() - truth.double()).abs().max()) if truth.numel() else 0.
            limit = max(relative_limit, native_error) if native_error is not None and math.isfinite(native_error) else relative_limit
            detail = {'input':list(path), 'objective':objective, 'error':error, 'oracle_scale':scale,
                      'native_reference_error':native_error, 'limit':limit,
                      'used_reference_bound':error > relative_limit and error <= limit}
            gradient_checks.append(detail)
            if error > limit:
                return {'outcome': 'wrong', 'reason': f'{prefix}: max error {error:.3g} on gradients up to {scale:.3g}; allowed {limit:.3g}',
                        'err': error, 'scale': scale, 'grad_arg': list(path), 'cotangent_output': objective,
                        'gradient_check':detail}
    return {**fwd, 'gradient_objectives': len(objectives), 'grad_inputs_checked': len(indexed),
            'gradient_policy':'reference_bounded_v1' if spec.flags.get('reference_bounded_gradients') else 'strict_relative_v1',
            'gradient_checks':gradient_checks}


def main():
    spec_path, kernel_path, cases_path, out_path, start, device = sys.argv[1:7]
    spec = OpSpec.from_json(json.load(open(spec_path)))
    cases = json.load(open(cases_path))
    try:
        kernel = load_kernel(kernel_path)
    except Exception:
        with open(out_path, "a") as f:
            for c in cases[int(start):]:
                f.write(json.dumps({"name": c["name"], "axis": c["axis"], "outcome": "refuse",
                                    "reason": "kernel failed to import: " + traceback.format_exc()[-300:]}) + "\n")
        return
    for c in cases[int(start):]:
        r = run_case(spec, kernel, c, device)
        with open(out_path, "a") as f:
            f.write(json.dumps({"name": c["name"], "axis": c["axis"], **r}) + "\n")
        if r["outcome"] == "crash":
            sys.exit(3)


if __name__ == "__main__":
    main()
