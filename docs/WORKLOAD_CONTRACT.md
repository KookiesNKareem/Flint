# One engine, explicit workload contracts

Design proposal; the current automatic adapter is PyTorch training with optimizer-update boundaries.
The other adapters described here are not implemented yet.

Flint should optimize useful work, with Python remaining the interface. The common engine should not
need to know whether that work is a training update, a request or a simulation trajectory. It does
need evidence that it can repeat the work safely and judge the result correctly.

## Two contracts

The **function contract** describes inputs, outputs, shapes, dtypes, layout, mutation, aliasing,
numerical semantics and required derivatives. It determines what a generated implementation must
preserve. A forward-only inference function need not pay for generating a backward implementation.

The **workload contract** describes:

- The useful unit of work and representative inputs, including rare shapes and long sequences.
- State that must be captured and reset, including RNG, buffers, caches and input position.
- Effects that may occur, and how replay isolates or suppresses them.
- Correctness checks and their fixed acceptance thresholds.
- Performance objectives: latency, throughput, peak memory, startup cost and expected run duration.
- A safe activation boundary, with a pinned implementation for each in-flight unit.

These are independent of the backend adapter. PyTorch, Julia or another runtime supplies capture,
compilation and dispatch; a workload adapter supplies replay and evaluation. Supporting a new
execution pattern should not require duplicating generation, repair, budget or artifact code.

## What changes between workloads

| Workload | Work unit and state | Safe experiment | Acceptance evidence |
|---|---|---|---|
| Training | Update; model, optimizer, RNG and data position | Isolated rehearsal with disposable outputs | Outputs, gradients, loss trajectory; update time and memory |
| Batch inference / evaluation | Batch or prompt set; RNG and caches | Replay fixed representative inputs | Output equivalence or declared quality checks; throughput, latency and memory |
| Simulation | Step and trajectory; physical state and RNG | Reset initial conditions in an isolated run | Local numerical checks plus invariants, convergence and long-horizon error; trajectory time and memory |
| Serving | Request or stream; session and KV state | Shadow traffic with external writes disabled | Output checks; tail latency and throughput at representative concurrency |
| Pipeline | Stage or partition; files, databases and upstream state | Fixtures, temporary destinations or transactional rollback | Schema and task-specific checks; stage and pipeline time, memory and I/O |

Observation is not permission to replay. Capturing arguments also does not prove a function is pure.
When the adapter cannot establish a controlled experiment, Flint can profile and suggest a source
change, but should not repeatedly execute the application to qualify it. Isolation must be enforced
by the worker or execution environment; an annotation alone is insufficient.

## One lifecycle, two schedules

Observe → rank opportunities → review a source diff → capture an isolated work unit → generate →
check → benchmark the real work unit → qualify → activate at a safe boundary.

Background policy schedules this after the current work and uses the result on a later invocation.
Prepared policy completes it before the full launch. They use the same acceptance checks, shared
spending ledger and artifact identity. Models may propose implementations and repairs; they cannot
change the acceptance contract. A failing or inconclusive candidate retains the original function.

Rank opportunities by measured impact on the useful work, coverage and expected payback after build
and qualification cost. Aggregate GPU duration is not an end-to-end speedup estimate. Keep profiling
bounded and reuse evidence only while its source, inputs, environment and contract remain valid.

## Keep the interface small

The normal path remains `flint run -- script.py`, with saved personal preferences and a reviewable
source diff. Recognized frameworks can supply adapters automatically. For an unfamiliar workflow,
ask only for missing facts: a representative fixture, reset mechanism, work boundary or task check.
Do not require every application to adopt a new training loop or a new language.

Start implementation by extracting the existing optimizer-update behavior behind this contract.
The first additional adapter should demonstrate forward-only batch work. A simulation adapter should
then test state reset and long-horizon validation; serving and pipeline adapters must add enforced
effect isolation before enabling automatic replay. This sequence tests whether the abstraction is
general without claiming that a single profiler hook already supports every workload.

Performance receipts cover the measured workload distribution and environment. They cannot promise
that any implementation is faster under every future input or system load. Outside qualified
coverage, retain the reference and record the reason.
