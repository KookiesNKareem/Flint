# Historical native-operator experiments

The commands and package names below refer to the historical 0.2 release. For the current install, use the [Flint README](../README.md).

## Default activation safeguard (0.2.1)

Correctness verification now grants eligibility for a controlled benchmark only. Automatic and offline
runs keep the original reference when workload performance is unqualified. `MUSE_MODE=benchmark`
explicitly permits a checked cached candidate for measurement. The status distinguishes an artifact
that is ready from a function still running its reference, and reports the withholding reason.

This is an interim fail-closed safeguard, not a completed performance qualification system. There is
currently no automatic promotion path. The planned qualification must compare matched whole workloads
against their best working default, including compilation, and bind results to the workload, artifact
and environment. Uncertain, slower or memory-regressing results should keep the default. Any optional
speed/memory tradeoff must be explicit. Finite measurements cannot guarantee every future execution is
faster. Generation/qualification cost and cold-start overhead also need amortization accounting.

The following GPU results and bundle are historical **0.2.0** evidence. Version 0.2.1's activation guard
is locally tested; no new GPU rental or model generation was used for this change.

## External training test

We installed only the wheel on an NVIDIA L4 and cloned **karpathy/nanoGPT**, commit
`3adf61e154c3fe3fca428ad6bc3818b27a3b8291`. Its `model.py` gained exactly an import and one decorator on
its existing `LayerNorm.forward`. Its `train.py`, optimizer, checkpoint code and data preparation code
were unchanged. The KernelAgent checkout was absent from the worker.

```bash
python train.py /workspace/training_config.py --out_dir=/workspace/native
python train.py /workspace/training_config.py --out_dir=/workspace/native --init_from=resume --max_iters=44 --eval_interval=2
```

The supplied config enabled `compile=True`, BF16 autocast, two gradient accumulation steps, fused
AdamW, and checkpoint saving. The 0.90M-parameter model used two layers, width 192, four attention
heads, sequence length 128 and batch size four on TinyShakespeare. This is a short integration test,
not a convergence study or a new-model result.

| Check | Observed result |
|---|---|
| Normal training command | Completed reference and generated paths, 40 iterations, then two 120-iteration repetitions each |
| Checkpoint resume | Both paths reached checkpoint iteration 44 with 28 optimizer state entries and finite parameters; pinned native artifact resumed again to 46 |
| Full model `torch.compile(fullgraph=True)` forward/backward | One graph, zero graph breaks; tested with BF16 autocast |
| Actual CUDA execution | Profiler recorded `_ln_fwd_kernel`, `_ln_bwd_dx_kernel`, `_ln_bwd_wb_kernel` |
| Same state/input numerical comparison | Loss absolute difference 0.0000129; maximum parameter-gradient relative L2 error 0.397% |
| Unsupported shape | Original reference executed within a complete graph |
| Shared cache and project pin | Second project directory trained with the same accepted artifact and no generation calls |
| Local suite | 95 passed, 1 skipped |
| Standalone distribution | Isolated wheel imports and wheel rebuild from standalone sdist passed |

The runtime reports **graph trace counts**, not executed-call counts. Device execution evidence comes
from the separate profiler audit. The cache was prebuilt in an isolated window for the normal-command
comparison; this does not establish distributed hot activation or seamless mid-step replacement.

## Model contributions and measured performance

Spark Open generated the forward implementation ($0.0349202). Wonder generated the native backward
($0.244756), which passed numerical checks but failed PyTorch's AOT dispatch check. Wonder repaired
it ($0.187202), replacing atomic parameter-gradient reductions with deterministic reductions. No
human-written Triton was substituted. The accepted artifact passed 23/23 development checks
(including the native compiler gate) and 12/12 fresh holdout checks.

The repeated normal-command measurements do **not** show an end-to-end speedup over compiled
PyTorch: pooled median reported iteration time was **5.65 ms reference vs 5.84 ms generated**, about
3.4% slower. Final validation loss was 2.5282–2.5286 for reference and 2.5285–2.5286 for generated.
Measurements used ABBA process order and logged iterations 20–110 excluding evaluation iteration 60;
these are short wall-clock samples, not a powered performance study. The isolated forward/backward
measurement varied between runs, so we make no speed claim from it. No memory benefit was established.

The baseline profiler records fused LayerNorm kernels that also contain residual additions, embedding
operations and backward work. This makes LayerNorm a useful integration and eventual no-regression
control, but a poor primary test of removing the implementation bottleneck for a new recurrence.
The 3.4% difference is descriptive of these short runs; statistical significance was not established.

The next performance experiment should use the researcher-defined two-state recurrence with connected
state gradients and a meaningful state-dependent update. Compare the same model and data under eager
PyTorch, `torch.compile`, and generated native forward/backward at increasing sequence lengths. Keep
optimizer, AMP, accumulation and token budget matched. Report whole-step time, peak allocated memory,
compile/build latency and amortization separately. The existing 21× result is against eager execution;
it does not establish a win over `torch.compile`. A useful stretch target is 2× faster whole training
steps than the best working PyTorch baseline, or materially lower memory that enables a longer sequence.
Keep the LayerNorm case as a control that an eventual performance gate should leave on the baseline
when substitution offers no measured benefit. This experiment has not yet been run.

The tested 0.2.0 runtime gated correctness and compatibility but did not gate activation on
end-to-end benefit over `torch.compile`. Version 0.2.1 now withholds unqualified candidates by default;
the workload measurement and promotion mechanism remains to be built. A shared cache
and native compiler integration make the tool usable in another repository; they do not guarantee
that replacing an already optimized LayerNorm is worthwhile.

Native capture specializes to checked shape/dtype/stride and gradient masks. Other signatures run
the reference. Direct module attributes and explicit positional method arguments are supported;
arbitrary object state, defaults and more complex method signatures may remain on the reference.
Source redefinition or a new process rebuilds identity; arbitrary mutation of captured Python globals
is not a supported hot-edit mechanism. DDP coordination, higher-order gradients, every training
framework and notebook source discovery have not been qualified by this test.

Model spend was **$0.4668782**. The L4 reservation was approximately **$0.259** at $0.59/hour, including
setup and tests: approximately **$0.73 total**, not a provider invoice. The pod was deleted after
retrieval, with a deletion receipt retained.

## Reproduce and inspect

The [evidence directory](../runs_public/portability/) contains the exact two-line patch, frozen data,
training config, wheel, sdist, accepted source, contract, model events, rejection/repair, checks,
compiler reports, training logs, cost receipt, lockfile and SHA-256 manifest.

On a provisioned CUDA environment, `bash replay.sh NEW_WORK_DIRECTORY` clones the pinned upstream
revision, installs the wheel, revalidates the cached implementation on that environment without model
calls, and runs the ordinary training and resume commands. The individual commands were exercised;
the combined replay wrapper was syntax-checked after the GPU was released. A different GPU must pass
revalidation before using the artifact. Compatible environments can run accepted artifacts offline.

The archived demo bundle additionally contains CPU-loadable
checkpoints, Chrome profiler traces and compiler-generated source in `gpu-evidence.tar.gz`.
