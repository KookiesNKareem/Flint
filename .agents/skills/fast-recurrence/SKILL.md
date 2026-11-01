---
name: fast-recurrence
description: Make a slow PyTorch function (typically a recurrent layer written as a Python loop over time) fast and trainable with a verified Triton kernel, using the Flint decorator in this repository. Use when the user asks to speed up, write a kernel for, or make trainable a slow layer, recurrence, scan, or custom op in their PyTorch code.
---

# Fast recurrence

The user keeps their reference implementation. You add one decorator; a background build writes a forward and
backward Triton kernel, checks it against the reference in float64 (explicitly counted empirical cases and gradient checks), and swaps
it in only for checked signatures after workload qualification. Correctness alone never authorizes default activation. Never hand-write the kernel yourself and never weaken the checker.

## Steps

1. Find the slow function. It must take tensors (and plain scalars) positionally and return a tensor or a
   tuple/list/string-keyed dictionary of tensors. Initial states may use the same nested containers.
   Direct module methods with explicit positional arguments can be decorated; parameters remain live module attributes.
2. Add the decorator, and nothing else:

   ```python
   import flint

   @flint.fast
   def my_layer(q, k, v, g, beta, scale):
       ...  # unchanged
   ```

3. Run the user's training (or a few steps of it) on a CUDA machine. After two calls the decorator infers the
   contract from the real tensors (shapes, dtypes, unit-norm rows, ranges such as g <= 0 or beta in [0, 1])
   and starts a background build. Training keeps running on the reference meanwhile.
4. Use `flint inspect <function>` for cached status:
   `building` -> `ready` (kernel eligible for guarded use) or `failed` (reason given).
   `run/events.jsonl` has every attempt and the counterexamples the model was sent.
5. When it is `ready`, report accepted/unsupported/skipped/failed cases, forward+backward time versus a matched reference, and model cost separately from GPU cost.
   Call `my_layer.status()` for per-call routing counts (kernel vs reference and why).
6. A committed cache is reused only when source/dependencies, environment and kernel hashes match. A new
   GPU/software environment requires revalidation. Recreate the decorator after source edits.
7. For backward recovery, use `flint.train_step` with the decorated functions and a loss-only closure.
   Fatal CUDA context failures require restarting the worker; ordinary forward fallback cannot catch a later backward error.

## Gradient policy

The user approved Flint's `reference_bounded_v1` rule on Oct 27, 2026: retain the existing normwise
float64 tolerance, or accept a candidate no less accurate than native PyTorch on the same inputs and
cotangent. Record candidate/native errors and the bound used. Nonfinite baseline errors do not relax
the bound. This is now part of the fixed checker; do not further weaken it. Historical strict-relative
receipts retain their original policy and must not be relabeled.

## Workload performance qualification

Use `flint qualify --input DATA_FILE -- train.py NORMAL_ARGUMENTS` for a repeatable training command.
It runs at least three alternating whole-command comparisons, including startup and compilation, and
records peak allocated CUDA memory. A candidate must beat all reference samples by the margin. Memory admission uses the saved allowance
(default zero growth). Faster checked candidates above that allowance wait for review rather than
automatically spending more on repair. `flint approve` approves the measured memory tradeoff for the
exact workload/artifact; `--remember` saves the observed allowance. Explicit user approval in the current
conversation is sufficient authorization to apply that approval. It never overrides correctness,
missing speed evidence, or stale source/environment/artifact bindings. Include external data/config dependencies with `--input`.
The resulting receipt is bound to the command, source, environment and exact joint artifact set.
Automatic/offline activation occurs on a subsequent matching normal run; `FLINT_MODE=benchmark` is
an explicit measurement-only bypass of the performance gate, never of correctness checks.
Keep model IDs in provider configuration. Never rename historical receipts or edit generated Triton.

## Multiple states and memory

For a researcher-defined recurrence returning `(outputs, {'fast': fast, 'slow': slow})`, preserve all states
and their gradient connectivity across chunks. The checker evaluates every output, independent final-state
cotangents, and every floating input gradient. Multiple-output calls preserve output dtypes exactly.

Use `@flint.fast(models=['meta-models/muse-spark-open-1026'], backward_model='muse-wonder-1.0',
memory_budget_mb=4)` to explicitly hand a checked Spark Open forward to Wonder for backward generation.
The memory limit is peak extra CUDA allocation for the op's full forward/backward above resident inputs,
including gradients and workspace, on each captured signature. It is not a whole-model memory limit.
`memory_opt_rounds` defaults to one additional measured optimization attempt. Models write all Triton;
the checker can reject a numerically correct implementation for exceeding its memory limit.

See `examples/muse_kernel/two_memory.py` for BF16 training, connected sequence chunks, accumulated
gradients, and checkpoint resume. Normal PyTorch accumulation in that example does not use automatic
backward recovery. Report actual dispatch counts and measured results; do not infer model quality from
kernel correctness or infer general shape support from captured shapes.

## Settings (environment)

- `MUSE_KERNEL_MODEL` (default `muse-spark-1.3-contributor,muse-wonder-1.0`)
- `MUSE_KERNEL_MAX_USD` per model attempt (default 4). The workbench additionally enforces one shared experiment cap. Spend is tallied on the machine that runs the build.
- `MODEL_API_KEY` must be set (Meta Model API).
