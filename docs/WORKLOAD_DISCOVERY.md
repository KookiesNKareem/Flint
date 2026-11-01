# Find a useful kernel boundary

Flint 0.6 saves your personal setup once:

```bash
flint setup
```

Choose background or prepared launch and a cumulative model spending cap. Setup saves the shared
ledger location and optional provider keys outside your projects. Project overrides can tighten the
cap but cannot raise it or choose a fresh ledger. Existing spending counts against the cap. GPU rental
is separate; this launcher uses an existing CUDA environment or a configured worker.

After setup, run your normal command:

```bash
flint run -- train.py config.py
```

Flint shows a supported source diff. Accepting it applies the guarded edit and continues automatically:
background mode optimizes after the current run; prepared mode discovers in a bounded rehearsal and
qualifies before the full launch. Declines are remembered. Noninteractive runs leave source unchanged
and a review pending; repeat the command in a terminal to review it. Repeated launches wait for an
active optimization of the same workload. No separate apply/build/qualify commands are needed.

The detailed observation behavior below applies to background mode. For profiling and suggestions
without automatic continuation, use `flint run --policy observe -- train.py config.py`.

The first run executes the original command. After three optimizer updates, it profiles five updates and then removes the profiler and hooks while training continues. It records project function scopes, CPU operators, and CUDA kernels, including backward work correlated to forward operators. It does not retain tensor values, collect full Python stacks, call model providers, or edit source. An identical subsequent run reuses the discovery report without profiling. The cache binds source/configuration, command, environment, window settings and explicit `--input` files.

The output includes source locations and a diff. CUDA kernel duration is aggregate device work, **not a wall-time fraction or a predicted speedup**. Existing library operations and data movement are deprioritized. CPU-only evidence does not produce GPU optimization suggestions. Compiler-fused work without a source attribution remains unattributed.

For an explicitly managed workflow, the proposal is also available as a file:

```bash
flint apply .flint/workflows/TASK/PROPOSAL.json
```

A proposal adds an import and decorator. When a profiled helper sits inside a suitable existing conditional block, Flint can suggest extracting that block into a tensor helper, preserving its statements and passing its inputs explicitly. For example, a scan's coefficient construction can remain inside the generated contract instead of imposing an unsafe value assumption on its inputs. Extraction is deliberately limited to one live output and conservative statement forms. Review the diff: this is not a general purity proof or arbitrary graph partitioner.

Apply verifies the original source hashes and reconstructs the allowed edit before writing it. Changed source or modified proposal bodies are rejected. It saves the original source and revokes prior qualification. The subsequent kernel checks remain mandatory.

## Choose when optimization happens

```bash
# Finish this training run on the reference, then optimize for the next run.
flint run --policy background --budget-ledger ~/.cache/flint/budget.json \
  --max-usd 6 --minutes 15 --rehearsal-steps 20 --input dataset.bin -- train.py config.py

# Prepare and qualify before launching the original full command.
flint run --policy prepared --budget-ledger ~/.cache/flint/budget.json \
  --max-usd 6 --minutes 15 --rehearsal-steps 20 --input dataset.bin -- train.py config.py

# Prepare only; do not launch the full experiment yet.
flint run --policy prepared --prepare-only -- train.py config.py
```

Both policies use capture → generation → correctness/native checks → fresh holdouts → workload qualification → diagnosis/repair → recheck. The configured models and persistent spending ledger are shared. Spark Open can generate the forward implementation and Wonder the backward implementation; repair routing alternates the configured models. Discovery itself costs no model calls. A rejected candidate leaves the reference active in background mode and blocks the full launch in prepared mode. Matching rejected evidence is also cached: automatic launches do not repeatedly benchmark the same candidate until noise produces a pass. Change the source/candidate/environment, or explicitly use `flint optimize` to request a new investigation. A requested rehearsal longer than the accepted one requires new qualification.

Background optimization is a detached controller process, queued **after** successful foreground training on the configured worker. It does not contend with that training run. It is not yet a separate simultaneous GPU worker or an editor daemon. Inspect or cancel it with:

```bash
flint status .flint/workflows/TASK
flint cancel .flint/workflows/TASK
```

Preparation uses a bounded rehearsal of completed optimizer updates without changing script arguments. It collects up to 32 distinct decorated input signatures across that window, including validation or sampling reached by the script. A cached candidate encountering new observed coverage is withheld and rechecked, with fresh holdouts, before reuse; failed reuse requires a new checked candidate. Both reference and candidate receive one compiler-cache preparation run, followed by at least three fixed alternating measurements. Existing speed and memory admission gates are unchanged. The receipt labels its scope as `optimizer_step_rehearsal`: it covers that prefix, not later validation, checkpoint writes, input buckets, or a whole long experiment. Rehearsal ends after the optimizer post-hook, so multiple optimizers count as multiple updates. Use `flint optimize` without `--rehearsal-steps` for whole-command qualification.

Once ready, prepared mode launches the normal full command with strict qualified dispatch. A decorated function with missing evidence or an uncovered input signature fails instead of silently falling back. Matching accepted artifacts require no model calls. This is a first single-process launch policy; distributed coordination, comprehensive input coverage planning, and a guarantee about future system load remain outside its scope.

## Remote worker and cost controls

Pass `--project /worker/project --remote 'bash /path/to/worker.sh'` to each command, including `apply`. Install the same Flint wheel on controller and worker. The worker wrapper receives one shell command; it should select the existing CUDA environment. Keep provider credentials only on the controller.

Flint serializes its own training, profiling and optimization work through a shared lease in the configured `worker_dir`. Use the same directory for projects sharing a GPU. It cannot prevent contention from other applications. An unrecoverable transport failure may leave a lease; inspect the worker processes before removing a stale owner file.

The ledger cap counts all model attempts across participating projects; it is not a fresh allowance for each run. GPU rental is separate. `--minutes`, `--max-rounds`, `--timeout` and `--run-timeout` bound work, but do not terminate a rented host. Repeated qualification runs need disposable outputs and representative explicit data dependencies.

Profiler integration uses PyTorch's [profiler](https://docs.pytorch.org/docs/main/profiler) and optimizer post-hooks. The accepted implementation still uses native PyTorch operator/autograd registration, so the existing training loop and compiler integration remain in place.


## Scope

Automatic discovery and bounded preparation currently use PyTorch optimizer updates. They do not yet
provide request, simulation or pipeline adapters. Rehearsal truncates execution at an update boundary;
it does not isolate file or network effects. Use replayable experiments with disposable outputs.
The [workload contract proposal](WORKLOAD_CONTRACT.md) separates execution, correctness, effects and
performance so those other workloads can share the engine with appropriate controls.

The 0.6 setup/review orchestration is covered by local tests and installed-wheel CPU smoke runs.
GPU generation and performance evidence remains the separately frozen 0.5 report; no new GPU speedup
is claimed for this interface change.
