# Flint

Flint keeps the original Python implementation and uses configured model providers to build checked
GPU alternatives. The API is `import flint_gpu as flint` and `@flint.fast`. The standalone distribution is
`flint-gpu`; the local wheel is not a PyPI publication. Spark Open and Wonder remain the providers
used for this hackathon. The older `muse_kernel` import remains a compatibility alias.

```python
import flint_gpu as flint

@flint.fast
def recurrence(q, k, v, decay, beta, transfer, state, scale):
    # Ordinary PyTorch, including the complete model definition.
    ...
```

Configure providers and workers with `flint config --init`, using `[flint]` in
`~/.config/flint/config.toml`. Projects override settings with `[tool.flint]` in `pyproject.toml`.
`FLINT_CONFIG`, `FLINT_CACHE`, `FLINT_REMOTE`, `FLINT_WORKER_DIR`, `FLINT_MODEL`, `FLINT_MAX_USD`,
`FLINT_MODE` and `FLINT_REPORT` are the public environment settings. Provider credentials remain
provider-specific and are not embedded in artifacts. Existing legacy environment settings still work;
Flint settings take precedence without mutating the caller's environment.

## Qualification

A candidate that passes output, gradient, holdout and native compiler checks is eligible for a
controlled benchmark. Correctness alone does not activate it in automatic or offline mode.

```bash
flint qualify --input dataset.bin -- train.py config.py
python train.py config.py
```

Qualification first runs one complete preparation command per mode, then runs the Python script
unchanged in fresh processes, using the same arguments and alternating the default and candidate
order. This gives both paths prepared compiler caches. At least three pairs must complete. Every measured
candidate command must beat every measured default command by the admission margin (5% by default),
and the highest candidate peak allocated CUDA memory must not exceed the lowest default peak.
This is a conservative empirical rule, not a statistical guarantee. Failed, slower, inconclusive,
or memory-regressing candidates leave the default in place. The training script's existing
`torch.compile` configuration is preserved; no eager-only baseline is substituted silently.

Native gradient checks use the explicitly approved `reference_bounded_v1` policy. For each input and
output cotangent, the error against float64 must satisfy the existing 2% normwise bound **or** be no
larger than the original PyTorch implementation's error on exactly the same inputs. Nonfinite baseline
errors do not enlarge the bound. Every comparison records the oracle scale, candidate error, native
reference error, allowed limit and whether the reference bound was needed. This avoids rejecting a
candidate that is more accurate than the original function in a precision-limited case. Output checks,
input mutation checks, fresh holdout and compiler checks remain mandatory. Historical strict-relative
receipts are kept separately; cache schema 4 requires revalidation of earlier environments.

The receipt records complete-command latency, including startup and any compilation performed,
CUDA peak allocation, observed candidate artifacts, logs, and qualification cost in elapsed seconds.
It also estimates the number of future identical runs needed to repay qualification itself, including
both preparation commands. Model generation, capture and earlier setup are excluded and must be accounted
for separately. Warm-step measurements can explain the outcome but do not override the admission rule.
Preparation time and peak allocation remain visible. Qualification does not promise equivalent
performance after compiler caches are removed; it applies to the prepared state tested by the command.

The receipt is bound to command arguments, project Python/config files, explicit input files, runtime
source/version, GPU/software environment, relevant CUDA/Torch/threading environment settings, and the
exact joint artifact set. Changed or missing evidence invalidates permission. Include data and external
dependencies with `--input`; hidden external services, arbitrary environment-dependent behavior and
changing shared-GPU load cannot be guaranteed by a file-based fingerprint. Qualification is intended
for repeatable, single-process Python workloads; distributed qualification is not implemented.

Qualification lives in `.flint/qualification.json`. Activation occurs on the next matching normal run,
not midway through an existing training step. Changes to one artifact invalidate the joint approval,
so a partially updated set cannot silently reuse an old result. A project `flint.lock` pins artifact
hashes; it does not bypass performance qualification. Explicit `FLINT_MODE=benchmark` permits checked
cached candidates for comparisons, without bypassing numerical/compiler checks or starting generation.

The command runs repeatedly and may write checkpoints. Use disposable experiment outputs. Subprocesses
have a timeout, and a failed or interrupted qualification cannot leave an old approval active.

## Scope

`flint optimize` now automates capture, generation, qualification-failure diagnosis, budgeted
model-written repair, fresh checks and requalification. The model-dollar ledger is shared across
providers and projects. Accepted candidates must also activate in a final normal run. See the
[command guide](AUTONOMOUS_OPTIMIZATION.md) and [three-repository trial](../runs_public/flint-loop/final-results.json).
Those external trials verified the loop and fallback, but did not produce an admitted speedup.
The separate A40 recurrence replay did complete automatic memory repair and normal-run activation:
28.101 → 10.535 seconds median prepared full-command time, 29,079,552 → 28,417,024 bytes peak allocation.
The original candidate was rejected before Spark Open's repair. Full evidence and the harness
preparation-policy correction are recorded in the trial report.

Native operators use `torch.library.triton_op`, `wrap_triton`, and registered autograd. Parameters
lifted from decorated module methods stay live. Qualified steady-state calls do not inspect source,
poll files, call a model or read tensor values on the CPU. Unsupported signatures use the original
function. Compilation specializes to checked shapes and layouts.

This is an implementation tool, not an architecture-discovery claim. Numerical evidence is empirical.
A new recurrence must retain its semantics through generation and repair; the checker must not be
weakened to make a candidate pass. Performance claims must distinguish compile/setup latency,
whole-command cost, warm optimizer-inclusive steps, operator timings, and memory.

## Executed result — October 27, 2026

The standalone 0.3.0 wheel ran a two-state recurrent byte-language-model training command with
`torch.compile(fullgraph=True)`, BF16, AdamW and gradient accumulation. Three alternating pairs using
warm compiler caches measured **19.603 s → 7.222 s** median complete-command time (**2.71×**), with
peak allocated CUDA memory **29,079,552 → 28,855,296 bytes**. A subsequent ordinary command activated
the qualified artifact automatically. The first candidate was rejected for increased memory; Spark
Open's checkpointed backward repair reduced its saved activations by 80.4%. Wonder supplied the
backward implementation that this repair extended. The selected code passed 23 development/native
checks and 12 fresh holdouts, plus a full-model gradient and GPU-profiler audit.

The unrelated nanoGPT control retained its byte-identical upstream training script. Compilation,
BF16, accumulation and checkpoint resume worked, but its LayerNorm candidate failed the performance
gate and normal mode retained PyTorch. This is useful selectivity, not a speedup claim for nanoGPT.

[Historical qualification results](../runs_public/flint/results.json).
The bundle includes all failures and measured evidence. Local validation: 106 passed, one skipped.
This run used $3.5765 of model calls and an estimated $0.6929 of GPU rental; the L4 was deleted.

## Terminal review

Foreground optimization shows an activity bar with its phase and elapsed time. A faster, checked
candidate that exceeds the saved memory allowance shows its measurements and asks
`Use this kernel? [Y/n]`. Enter approves that exact measured implementation; `n` keeps the reference.
Approval continues to normal-dispatch verification. Background results are reviewed on the next
foreground `flint run`; unattended runs and EOF leave them pending. Correctness and repeated speed
requirements remain mandatory. This terminal change was tested locally without new GPU/model runs.

Metal/MPS support and local validation are documented in [METAL.md](METAL.md).
