# Bounded workload optimization

`flint optimize` owns capture, generation, whole-command qualification, measured failure feedback,
model-written repair, numerical/native rechecking, fresh holdout, and requalification. After admission,
it runs the original command in automatic mode and checks that the qualified artifacts were actually
selected. A failed run retains the reference implementation.

```bash
flint optimize --max-usd 6 --budget-ledger ~/.cache/flint/experiments/trial.json \
  --minutes 20 --max-rounds 2 --input dataset.bin -- train.py config.py
```

With an existing remote CUDA environment, the controller keeps the model credentials:

```bash
FLINT_WORKER_DIR=/workspace/flint-worker flint optimize \
  --remote 'bash /path/to/gpu.sh' --project /workspace/project \
  --max-usd 6 --budget-ledger /path/to/shared-budget.json \
  --minutes 20 --max-rounds 2 --input dataset.bin -- train.py config.py
```

The worker needs the same Flint runtime, the training project's dependencies, and GNU `timeout`.
The CLI does not rent GPUs or upload a repository. It operates on an existing project directory.
The remote command prefix must run one shell command, as in the existing build/revalidation interface.
Use a new `--out` directory for each controller run. Worker evidence is under
`.flint/optimizations/<session>/`; the current admission receipt remains `.flint/qualification.json`.

The model budget is the **total cap of the shared persistent ledger**, not a fresh allowance for
each repository or retry. Reusing the ledger reuses both settled spend and outstanding reservations.
Both providers reserve a conservative worst-case cost before each request; ambiguous API failures
retain their reservation. The wall-time limit is separate from model dollars and does not terminate
rented infrastructure. `--max-rounds` limits workload-repair revisions; each revision has a bounded
number of numerical/compiler repairs, all charged to the same ledger.

A broken reference command, missing candidate dispatch, changed workload/artifacts, absent memory
telemetry or insufficient measurements does not trigger speculative model spending. Successful paired
measurements distinguish latency/noise and memory regressions. This diagnosis reports observed facts;
it does not pretend to know which operator caused an end-to-end failure. The model receives the
original equation, checked implementation, whole-command measurements and operator memory evidence.
Only implementation changes are allowed. The training script, checks and thresholds remain fixed.

A new implementation must pass development, registered-autograd/compiler checks, and a freshly drawn
post-selection holdout before installation. Holdout failures are not returned to the model for repair.
Artifacts and receipts from earlier attempts remain archived. One complete command prepares each path
before the fixed measurements; preparation time and peak allocation are retained separately. This
avoids comparing a cold generated implementation against a reference already warmed during capture.
Admission is three alternating complete-command pairs with prepared compiler caches, every candidate
faster than every reference by the margin and no increase
in measured peak allocation. No numerical acceptance or operator microbenchmark bypasses this rule.
If final normal-mode activation fails, approval is revoked.
Preparation is included in the qualification cost and break-even estimate. Cold-cache behavior is
reported separately; a warm-cache qualification is not a guarantee after compiler caches are removed.
An unchanged repair digest stops the loop; Flint does not repeatedly benchmark the same implementation
until favorable noise produces an approval.

The current loop repairs one observed artifact per round and evaluates the complete joint artifact
set afterward. It is for repeatable single-process training commands. Source/data edits must happen
outside an optimization session. Training commands run repeatedly and may save checkpoints; use
disposable experiment output locations. It does not yet isolate arbitrary external side effects,
provide distributed coordination, or guarantee outcomes under unmeasured hardware/load/cache states.
An interrupted remote connection can leave an owner file; inspect the saved session and ensure its
processes have stopped before explicitly clearing `.flint/optimization-owner.json`.
