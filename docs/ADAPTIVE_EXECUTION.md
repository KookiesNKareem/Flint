# Adaptive observation and activation

Flint's adaptive policy observes completed work, qualifies a candidate in a controlled window, and
can switch implementations within the same process. It currently supports a single-process PyTorch
training adapter and an explicit batch-function adapter.

```bash
flint setup --policy adaptive
flint run -- train.py
```

A project must establish that its command can be replayed with disposable outputs before Flint
repeatedly executes it for qualification:

```toml
[tool.flint]
replay_safe = true
```

This declaration is not a filesystem/network sandbox. Unknown or side-effecting commands are not
automatically replayed. Training uses completed optimizer updates. Batch workloads identify their
outer prediction function once:

```python
@flint.workload
@torch.no_grad()
def predict(batch):
    return model(batch)  # model may already be compiled
```

Set `workload_kind = "batch"` in the project's `[tool.flint]` table. A kernel opportunity is still an
ordinary `@flint.fast` function inside that work unit. The wrapper identifies a completion boundary;
it neither changes model math nor grants replay permission. Failed and nested inner calls do not emit
additional completed work units.

## Observation

The controller compares successive bounded windows of host work-boundary intervals and captured
signatures. Two stable windows, with no newly discovered signatures, allow qualification for that
observed phase. Defaults bound observation to 64 units or 30 seconds after the first completed unit, checked at work boundaries.
A long individual work unit is not interrupted. Timing stability is not a correctness test, GPU
utilization measurement, proof of representativeness, or promised speedup. Unstable/unknown workloads
finish observation as inconclusive and retain the reference.

Detailed discovery profiling can use the same stopping rule. Metadata capture retains meta tensors,
not application tensor values. During an accepted compiled graph there is no source hashing, model
request, file polling or tensor-value inspection. New signatures take the reference and can reopen
bounded metadata observation. This release permits one optimization attempt per live launch; additional
coverage needs a new qualification on a later invocation. This prevents an unfamiliar input stream
from repeatedly spending the saved budget.

## Qualification and activation

On a single GPU, the application pauses at the completed work boundary. The controller uses the same
shared model ledger and repair pipeline in separate processes, then resumes the application. This
avoids concurrent benchmark/training contention; it is not zero-pause background optimization. The
pause and generation/checking costs are recorded separately from steady-state speed. Short jobs may
not recover these costs; prepared and next-run background policies remain available.

The application retains its live model/optimizer state while qualification replays the command.
Training/RNG/data reset and external effects must therefore be compatible with the project's replay
contract. Failed checks or inconclusive performance retain the reference. Live admission requires both
whole-command improvement and faster steady-state work units across three paired replays. CUDA
synchronization for these timings occurs only inside qualification, after initial units are discarded. If a lost controller leaves
another worker owning the GPU, the application refuses unsafe resume.

Admission checks the current receipt and entire artifact set. Callable versions are immutable;
in-flight autograd graphs retain their original operations. Loading preserves the application's RNG.
PyTorch guards retrace a compiled callable when its selected implementation changes; Flint does not
reset all compiler caches. Recompilation cost and unsupported runtime versions remain practical
constraints. Pinned artifacts and distributed runs do not receive mid-run updates.

## Verification

The shared checker adds compound layout/value/seed probes, generic independent holdouts, full
contract cases for every captured signature, and native compiler/autograd checks for additional
signatures. Failures retain exact seeded reproducers. Holdout failures are not returned to models as
repair feedback. Existing baseline-aware gradient tolerances remain unchanged.

These tests establish empirical coverage. They do not prove arbitrary numerical programs correct,
make mutation/aliasing semantics universal, or provide serving/distributed/pipeline adapters.
