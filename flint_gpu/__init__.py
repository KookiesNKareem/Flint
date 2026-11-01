"""One decorator on a slow PyTorch function; a configured model writes a verified Triton kernel for it in the background.

    import flint_gpu as flint

    @flint.fast
    def my_layer(q, k, v, g, beta, scale):
        ...  # the reference loop, unchanged

Native mode captures tensor metadata and builds checked forward/backward operators in the background.
Compatible signatures use an accepted torch.library.triton_op; other calls use the reference.
Source, environment, checks and artifact hashes bind cache reuse. Recreate the decorator after edits.
Configure providers, worker and budgets once with `flint config --init`; projects can override
settings in pyproject.toml. See packages/flint/README.md for the standalone interface.

"""

from __future__ import annotations

import functools
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import torch

from .identity import CACHE_VERSION, environment, signature
from kernel_contract.tree import leaves, map_tree, structure, at, key_path
from .recovery import train_step
from .units import workload
from .infer import NORM_TOL, describe_domains, merge_domains, observe, source_hash, standalone_module
from .config import read_config
from .compilation import REGISTRY, metadata, capture_trace, trace_event, matches, signature_from_metadata, meta_arguments, supported_inputs, tensor_outputs

__all__ = ["fast", "train_step", "workload"]

# the cheap model verifies as often as Wonder on our benchmark at ~1/100 the cost; Wonder takes over on failure
DEFAULT_MODELS = "meta-models/muse-spark-open-1026"
_MODULES = {}


def cache_root(fn) -> Path:
    """Use the configured shared artifact cache."""
    return Path(read_config()['cache_dir'])


class Fast:
    def __init__(self, fn, observe_calls: int, memory_budget_mb=None, models=None, backward_model=None, memory_opt_rounds=1, native=None):
        functools.update_wrapper(self, fn)
        self.config = read_config()
        self.native = self.config['native'] if native is None else native
        self.native_loaded = False
        self.loaded_sha = None
        self.activation_reason = None
        self.performance_qualified = False
        self.require_qualified = os.environ.get('FLINT_REQUIRE_QUALIFIED')=='1'
        self.live_observation = os.environ.get('FLINT_LIVE')=='1'
        self.capture_window = self.native and self.config['mode']=='capture' and os.environ.get('FLINT_EXTERNAL_BUILDER')=='1'
        self.capture_overflow = False
        self._watcher = None
        self.fn, self.n_observe = fn, observe_calls
        if observe_calls < 1:
            raise ValueError("observe_calls must be positive")
        if memory_budget_mb is not None and (not isinstance(memory_budget_mb, (int, float)) or not 0 < memory_budget_mb < float('inf')):
            raise ValueError('memory_budget_mb must be positive and finite')
        self.memory_policy = {'max_peak_extra_bytes': int(memory_budget_mb * 1024**2),
                              'opt_rounds': int(memory_opt_rounds)} if memory_budget_mb is not None else None
        if memory_opt_rounds < 0:
            raise ValueError('memory_opt_rounds must be nonnegative')
        if models is not None and (isinstance(models, str) or not models or any(not isinstance(m, str) or not m for m in models)):
            raise ValueError('models must be a nonempty sequence of model names')
        self.models = list(models) if models is not None else self.config['models']
        self.backward_model = self.config['backward_model'] if backward_model is None else backward_model
        try: parameters = tuple(inspect.signature(fn).parameters.values())
        except (TypeError, ValueError): parameters = ()
        self.positional_parameters = all(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in parameters)
        self.argument_names = tuple(p.name for p in parameters)
        self.positional_only = tuple(p.name for p in parameters if p.kind == p.POSITIONAL_ONLY)
        self.argument_defaults = {p.name: p.default for p in parameters if p.default is not p.empty}
        self.disabled = False
        try:
            self.hash = source_hash(fn)
        except (OSError, ValueError, TypeError):
            self.hash = 'unavailable'
            self.disabled = True
        self.environment = environment()
        self.job = cache_root(fn) / f"{fn.__name__}_{self.hash}"
        if self.native:
            self.job = self.job.with_name(self.job.name + '_native')
        if self.memory_policy:
            import hashlib
            policy_id = hashlib.sha256(json.dumps(self.memory_policy, sort_keys=True).encode()).hexdigest()[:10]
            self.job = self.job.with_name(self.job.name + '_' + policy_id)
        if self.environment.get('backend')=='mps':self.job=self.job.with_name(self.job.name+'_metal')
        self.pin = None
        self.pin_missing = False
        for parent in (Path.cwd(), *Path.cwd().parents):
            lockfile = parent/'flint.lock'
            if not lockfile.is_file(): lockfile = parent/'muse.lock'
            if lockfile.is_file():
                pins = json.loads(lockfile.read_text()).get('artifacts', {})
                self.pin = pins.get(self.job.name)
                self.pin_missing = self.pin is None
                break
        self.seen: list[dict] = []
        self.domains: dict[int, str | None] = {}
        self.kernel = None
        self.verified: list[tuple] = []
        self.stats = Counter()
        self.lock = threading.Lock()
        self._last_check = 0.0
        self.registry_key = str(id(self))
        REGISTRY[self.registry_key] = self
        self._startup_activation = True
        if self.disabled:
            self.stats['disabled: reference identity unavailable'] += 1
        else:
            self._load_if_ready()
        self._startup_activation = False
        if self.require_qualified and (not self.native_loaded or not self.performance_qualified):
            raise RuntimeError(f'Flint prepared launch refused {fn.__qualname__}: '+(self.activation_reason or 'qualified native artifact unavailable'))

    # ---- the guard -------------------------------------------------------------------------------
    def _signature(self, args) -> tuple:
        return signature(args)

    def _values_ok(self, args) -> bool:
        checks = []
        for i, d in self.domains.items():
            if not d:
                continue
            x = at(args, key_path(i)).detach()
            if d == "l2row":
                checks.append((x.float().norm(dim=-1) - 1).abs().max() < NORM_TOL)
            elif d == "unit":
                checks.append((x.min() >= 0) & (x.max() <= 1))
            elif d == "logdecay":
                checks.append(x.max() <= 0)
            elif d == "positive":
                checks.append(x.min() > 0)
        return bool(torch.stack(checks).all()) if checks else True

    def __call__(self, *args, **kwargs):
        # Argument metadata is cached once; Dynamo specializes this binding path.
        if self.positional_parameters and (kwargs or len(args) < len(self.argument_names)):
            if (len(args) > len(self.argument_names) or
                any(k not in self.argument_names or k in self.positional_only or
                    k in self.argument_names[:len(args)] for k in kwargs)):
                return self.fn(*args, **kwargs)  # Preserve Python's original TypeError.
            values = list(args)
            for name in self.argument_names[len(args):]:
                if name in kwargs: values.append(kwargs[name])
                elif name in self.argument_defaults: values.append(self.argument_defaults[name])
                else: return self.fn(*args, **kwargs)
            args, kwargs = tuple(values), {}
        if self.native:
            return self._native_call(*args, **kwargs)
        if self.disabled or self.activation_reason or self.config['mode']=='reference':
            self.stats["reference"] += 1
            return self.fn(*args, **kwargs)
        try:
            if source_hash(self.fn) != self.hash:
                self.disable("reference dependencies changed; recreate decorator")
        except (ValueError, OSError, TypeError):
            self.disable("reference identity unavailable")
        self._poll()
        if self.kernel is not None and not kwargs and self._signature(args) in self.verified:
            if self._values_ok(args):
                try:
                    out = self.kernel(*args)
                    self.stats["kernel"] += 1
                    return out
                except Exception as e:
                    if any(term in str(e).lower() for term in ('illegal memory access', 'device-side assert')):
                        raise
                    self.disable(type(e).__name__)
            else:
                self.stats["fallback: values outside the verified range"] += 1
        elif self.kernel is not None:
            self.stats["fallback: shape or dtype not verified"] += 1
        out = self.fn(*args, **kwargs)
        if self.kernel is None and not kwargs and not self.disabled and self.config['mode'] not in ('offline','benchmark'):
            try:
                self._observe(args, out)
            except Exception as error:
                self.disable(f"build setup: {type(error).__name__}")
        self.stats["reference"] += 1
        return out

    def _native_call(self, *args, **kwargs):
        if self.require_qualified:
            if self.disabled or self.pin_missing or not self.native_loaded or kwargs or not supported_inputs(args):
                raise RuntimeError('Flint prepared launch: qualified implementation unavailable for this call')
            if not any(matches(args,sig) for sig in self.verified):
                raise RuntimeError('Flint prepared launch: input signature outside verified coverage')
        if not supported_inputs(args):
            return self.fn(*args, **kwargs)
        if torch.compiler.is_compiling():
            if self.disabled or self.pin_missing or self.activation_reason or self.config['mode'] == 'reference':
                output=self.fn(*args, **kwargs)
                if self.live_observation and not kwargs and tensor_outputs(output):capture_trace(self.registry_key,metadata(args),structure(output))
                return output
            if self.native_loaded and not kwargs:
                for sig in self.verified:
                    if matches(args, sig):
                        trace_event(self.registry_key, 'compiled kernel graphs')
                        return self.kernel(*args)
                trace_event(self.registry_key, 'compiled reference graphs: unsupported signature')
                output=self.fn(*args, **kwargs)
                if self.live_observation and not kwargs and tensor_outputs(output):capture_trace(self.registry_key,metadata(args),structure(output))
                return output
            output = self.fn(*args, **kwargs)
            if not kwargs and self.config['mode'] not in ('offline', 'benchmark') and tensor_outputs(output):
                capture_trace(self.registry_key, metadata(args), structure(output))
            trace_event(self.registry_key, 'compiled reference graphs: awaiting artifact')
            return output
        if self.disabled or self.pin_missing or self.activation_reason or self.config['mode'] == 'reference':
            self.stats['reference'] += 1
            output=self.fn(*args, **kwargs)
            if self.live_observation and not kwargs and tensor_outputs(output):self._observe_metadata(metadata(args),structure(output))
            return output
        if self.native_loaded and not kwargs:
            for sig in self.verified:
                if matches(args, sig):
                    result = self.kernel(*args)
                    self.stats['kernel'] += 1
                    return result
            self.stats['fallback: unsupported signature'] += 1
        output = self.fn(*args, **kwargs)
        if (not self.native_loaded or self.live_observation) and not kwargs and self.config['mode'] not in ('offline', 'benchmark') and tensor_outputs(output):
            # Capture tensor metadata without reading values or synchronizing the GPU.
            self._observe_metadata(metadata(args), structure(output))
        self.stats['reference'] += 1
        return output

    def _observe_metadata(self, meta, output_structure, trace=False):
        if (self.capture_window or self.live_observation) and not self.disabled:
            sig=signature_from_metadata(meta)['tuple']
            with self.lock:
                if any(s['signature']==sig for s in self.seen):return
                if len(self.seen)>=32:
                    self.capture_overflow=True;return
                self.seen.append({'args':meta_arguments(meta),'signature':sig,'output_structure':output_structure})
            return
        if self.disabled or self.job.joinpath('status.json').exists():
            self._watch()
            return
        with self.lock:
            if len(self.seen) >= self.n_observe:
                return
            sig = signature_from_metadata(meta)['tuple']
            self.seen.append({'args': meta_arguments(meta), 'signature':sig,
                              'output_structure':output_structure})
            if trace or len(self.seen) >= self.n_observe:
                try:
                    self._start_build()
                except Exception as error:
                    self.disable('capture: '+type(error).__name__+': '+str(error))
        self._watch()

    def finish_capture(self):
        """Finalize metadata once at the end of a managed rehearsal, never in a hot graph."""
        if not self.capture_window:return
        if self.capture_overflow:raise RuntimeError('Rehearsal exceeded 32 distinct signatures; narrow the workload')
        if not self.seen:return
        old_state=old_job=None
        if (self.job/'status.json').exists() and (self.job/'job.json').exists():
            old_state=json.loads((self.job/'status.json').read_text())
            old_job=json.loads((self.job/'job.json').read_text())
        reusable=old_state and old_state.get('state')=='ready' and old_state.get('compile_passed') and old_state.get('holdout_passed')
        if reusable and old_job.get('environment')==self.environment and all(
                x['signature'] in old_job.get('verified_signatures',[]) for x in self.seen):return
        if reusable:
            from .optimize_worker import ARTIFACT_FILES
            import shutil
            archive=self.job/'coverage-history'/str(time.time_ns());archive.mkdir(parents=True)
            for name in ARTIFACT_FILES:
                if (self.job/name).exists():shutil.copy2(self.job/name,archive/name)
        self._start_build()
        if reusable:
            # Preserve the model-written candidate, but revoke its old coverage until all new
            # development/native cases and freshly sampled holdouts have passed on the worker.
            old_state.update(state='needs_revalidation',compile_passed=False,holdout_passed=False,
                             reason='rehearsal observed additional input coverage')
            (self.job/'status.json').write_text(json.dumps(old_state,indent=2))

    def _watch(self):
        if os.environ.get('FLINT_EXTERNAL_BUILDER')=='1':return
        if self._watcher is not None or self.disabled or self.config['mode'] in ('offline','reference','benchmark'):
            return
        def watch():
            while not self.disabled:
                time.sleep(self.config['watch_interval'])
                try:
                    if source_hash(self.fn) != self.hash:
                        self.disable('reference changed; recreate decorator')
                        return
                    if self.kernel is None:
                        self._load_if_ready()
                    if self.activation_reason:
                        return
                except (OSError, ValueError, TypeError):
                    return
        self._watcher = threading.Thread(target=watch, name='flint-artifacts', daemon=True)
        self._watcher.start()

    # ---- observe and build ------------------------------------------------------------------------
    def _observe(self, args, output):
        if self.job.joinpath("status.json").exists() or len(self.seen) >= self.n_observe:
            return
        obs = observe(args)
        self.domains = obs["domains"] if not self.seen else merge_domains(self.domains, obs["domains"])
        self.seen.append({"args": map_tree(lambda a: a.detach().requires_grad_(a.requires_grad) if isinstance(a, torch.Tensor) else a, args), "output_structure": structure(output), **obs})
        if len(self.seen) == self.n_observe:
            self._start_build()

    def _start_build(self):
        from kernel_contract.agent import describe_call
        from kernel_contract.spec import capture, with_domains
        self.job.mkdir(parents=True, exist_ok=True)
        (self.job / "reference.py").write_text(standalone_module(self.fn))
        op = f"muse_jobs.j_{self.hash}.{self.fn.__name__}"
        names = list(inspect.signature(self.fn).parameters)
        specs, sigs = [], []
        for s in self.seen:
            sig = s.get('signature') or self._signature(s["args"])
            if sig in sigs:
                continue
            sigs.append(sig)
            spec = capture(op, tuple(s["args"]), name=self.fn.__name__)
            from .device import available
            spec.flags.update(any_float_output=True, normwise_rtol=2e-2,device=available())
            if self.native:
                spec.flags.update(native=True, any_float_output=False, reference_bounded_gradients=True)
            if self.memory_policy:
                spec.flags['memory_policy'] = self.memory_policy
            specs.append(with_domains(spec, {i: d for i, d in self.domains.items() if d}))
        (self.job / "spec.json").write_text(json.dumps(specs[0].to_json()))
        (self.job / "extra_specs.json").write_text(json.dumps([s.to_json() for s in specs[1:]]))
        if self.native:
            # The native checker tests all-floating-input gradients and no-grad, plus the captured mask.
            import copy
            for enabled in (True, False):
                for original in list(sigs):
                    variant = copy.deepcopy(original)
                    def grad_mask(value):
                        if isinstance(value, dict):
                            if 'dtype' in value and any(t in value['dtype'] for t in ('float', 'double')):
                                value['requires_grad'] = enabled
                            for child in value.values(): grad_mask(child)
                        elif isinstance(value, list):
                            for child in value: grad_mask(child)
                    grad_mask(variant)
                    if variant not in sigs: sigs.append(variant)
        job = {"backend":self.environment.get("backend","cuda"), "hash": self.hash, "name": self.fn.__name__, "source": standalone_module(self.fn),
               "signature": ", ".join(names), "call": describe_call(specs[0]), "sizes": specs[0].sizes,
               "grad_names": ", ".join(str(path) for path, a in leaves(self.seen[0]["args"])
                                       if isinstance(a, torch.Tensor) and a.is_floating_point()),
               "output_structure": self.seen[0]["output_structure"], "memory_policy": self.memory_policy,
               "domains": describe_domains(self.domains, names), "verified_signatures": sigs, "environment": self.environment, "cache_version": CACHE_VERSION,
               "models": self.models or os.environ.get("MUSE_KERNEL_MODEL", DEFAULT_MODELS).split(","),
               "backward_model": self.backward_model,
               "max_usd": self.config['max_usd'], "require_holdout": True, 'native':self.native,
               'gradient_policy':'reference_bounded_v1' if self.native else 'strict_relative_v1',
               'total_budget':True, 'opt_rounds':0, 'bwd_opt_rounds':0}
        (self.job / "job.json").write_text(json.dumps(job, indent=1))
        (self.job / "status.json").write_text(json.dumps({"state": "queued", "updated": time.time()}))
        if os.environ.get("FLINT_EXTERNAL_BUILDER", os.environ.get("MUSE_KERNEL_EXTERNAL_BUILDER")) == "1" or self.config['mode'] == 'capture' or not self.config['auto_build']:
            print(f"[flint] queued {self.job}", flush=True)
            return
        with open(self.job / "builder.log", "w") as log:
            subprocess.Popen([sys.executable, "-m", "flint_gpu.build", str(self.job)], stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True,
                             env={**os.environ,'MUSE_KERNEL_REMOTE':self.config['remote'],
                                  'MUSE_KERNEL_REMOTE_DIR':self.config['worker_dir']})
        print(f"[flint] {self.fn.__name__}: building a kernel in the background ({self.job})", flush=True)

    def _poll(self):
        if self.disabled or time.time() - self._last_check < 5:
            return
        self._last_check = time.time()
        if source_hash(self.fn) != self.hash:
            self.disable("reference dependencies changed; redecorate to build a new cache")
            return
        if self.kernel is None:
            self._load_if_ready()

    def _load_if_ready(self):
        st = self.job / "status.json"
        if not st.exists():
            return
        try:
            s = json.loads(st.read_text())
        except (OSError, ValueError):
            return
        if s.get("state") != "ready":
            return
        if self.config['mode'] in ('reference','capture'):
            return
        try:
            job = json.loads((self.job / "job.json").read_text())
            from kernel_contract.evidence import sha256
            kernel_path = self.job / Path(s["kernel"]).name
            if s.get("kernel_sha256") != sha256(kernel_path):
                self.stats["cache hash mismatch"] += 1
                return
            if self.pin is not None and s['kernel_sha256'] != self.pin['kernel_sha256']:
                self.stats['cache differs from project lockfile'] += 1
                return
        except (OSError, ValueError, KeyError, TypeError):
            self.stats["incomplete cache"] += 1
            return
        if job.get("require_holdout") and not s.get("holdout_passed"):
            self.stats["cache missing fresh checks"] += 1
            return
        if self.native and (not job.get('native') or not s.get('compile_passed')):
            self.stats['cache missing native compiler checks'] += 1
            return
        if job.get("memory_policy") != self.memory_policy:
            self.stats["cache memory policy mismatch"] += 1
            return
        if self.memory_policy and not 0 <= (s.get('memory') or {}).get('peak_extra_bytes', float('inf')) <= self.memory_policy['max_peak_extra_bytes']:
            self.stats['cache missing passing memory measurement'] += 1
            return
        if job.get("cache_version") != CACHE_VERSION or job.get("environment") != self.environment:
            self.stats["cache requires revalidation on this environment"] += 1
            return
        if self.config['mode'] != 'benchmark':
            if not self._startup_activation:
                self.activation_reason = 'workload performance qualification activates on the next run'
                return
            from .qualification import activation
            allowed, reason = activation(self.job, s, self.config, self.environment)
            if not allowed:
                self.activation_reason = reason
                self.stats['candidate withheld: performance unqualified'] = 1
                return
            self.performance_qualified = True
        self.activation_reason = None
        try:  # the cache may be committed and loaded on a machine without a GPU or Triton
            digest = s['kernel_sha256']
            if digest not in _MODULES:
                spec_mod = importlib.util.spec_from_file_location(f"flint_generated_{digest[:24]}", kernel_path)
                mod = importlib.util.module_from_spec(spec_mod)
                sys.modules[spec_mod.name] = mod
                spec_mod.loader.exec_module(mod)
                _MODULES[digest] = mod
            mod = _MODULES[digest]
            kernel = mod.kernel_function
            verified = job["verified_signatures"]
            domains = {i: d for i, d in json.loads((self.job / "domains.json").read_text()).items()} if (self.job / "domains.json").exists() else self._domains_from_spec()
        except Exception as e:
            self.stats[f"cached kernel unavailable: {type(e).__name__}"] += 1
            return
        self.kernel, self.verified, self.domains = kernel, verified, domains
        self.native_loaded = self.native and bool(s.get('compile_passed'))
        self.loaded_sha = s['kernel_sha256']
        print(f"[flint] {self.fn.__name__}: verified kernel in use "
              f"({s.get('verified_cases')}/{s.get('cases')} checks, ${s.get('usd', 0):.2f} build)", flush=True)

    def _domains_from_spec(self) -> dict[int, str | None]:
        spec = json.loads((self.job / "spec.json").read_text())
        from kernel_contract.tree import path_key
        domains = {}
        def walk(value, path):
            if isinstance(value, dict):
                if 'tensor' in value:
                    domains[path_key(path)] = value['tensor'].get('domain')
                elif 'seq' in value:
                    for i, x in enumerate(value['seq']): walk(x, path + (i,))
                elif 'map' in value:
                    for key, x in value['map'].items(): walk(x, path + (key,))
        for i, value in enumerate(spec['args']): walk(value, (i,))
        return domains

    def disable(self, reason="disabled by caller"):
        self.disabled = True
        self.kernel = None
        self.native_loaded = False
        self.stats[f"disabled: {reason}"] += 1

    def status(self) -> dict:
        st = self.job / "status.json"
        artifact = json.loads(st.read_text()) if st.exists() else {"state": "observing"}
        return {**artifact, "loaded_sha256":self.loaded_sha, "artifact_state": artifact['state'],
                "state": "reference" if self.activation_reason else artifact['state'],
                "activation_reason": self.activation_reason, "performance_qualified": self.performance_qualified, "mode": self.config["mode"], "calls": dict(self.stats)}


def fast(fn=None, *, observe_calls=None, memory_budget_mb=None, models=None, backward_model=None, memory_opt_rounds=None, native=None):
    """Decorate a slow PyTorch function; see the module docstring."""
    if fn is None:
        return lambda f: fast(f, observe_calls=observe_calls, memory_budget_mb=memory_budget_mb, models=models,
                             backward_model=backward_model, memory_opt_rounds=memory_opt_rounds, native=native)
    config = read_config()
    factory = lambda f: Fast(f, observe_calls if observe_calls is not None else config['observe_calls'],
                            memory_budget_mb if memory_budget_mb is not None else config['memory_budget_mb'],
                            models, backward_model, memory_opt_rounds if memory_opt_rounds is not None else config['memory_opt_rounds'], native)
    try: first_parameter = next(iter(inspect.signature(fn).parameters), None)
    except (TypeError, ValueError): first_parameter = None
    if first_parameter == 'self':
        from .methods import MethodFast
        try:
            return MethodFast(fn, factory).bound
        except (OSError, TypeError, ValueError) as error:
            reason = str(error)
            fn.status = lambda: {'state':'reference','reason':reason}
            return fn
    return factory(fn)
