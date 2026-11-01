# Metal support in Flint 0.10.1

Flint can capture an MPS workload, target Metal Shading Language with the configured Spark Open/Wonder
route, run development and fresh held-out checks, qualify the normal command and load a checked
implementation through the existing decorator. No CUDA machine is needed for this path.

The runtime uses PyTorch's `torch.mps.compile_shader`, `torch.library.custom_op`, fake implementations
and registered backward operators. Device synchronization, cache identity, portable subprocess deadlines,
work-unit timing and RNG restoration include MPS. CUDA keeps its Triton path. Normal accepted calls do
not sample memory or invoke models; sampling happens in controlled launcher/qualification processes.

## Try it on Apple Silicon

Install Flint from the [repository root](../README.md). The standalone example is
`examples/metal`, using ordinary autograd and AdamW. Use MPS tensors in your own script; Flint does not
rewrite a CUDA training script into an MPS one.

```bash
cd examples/metal
flint run --policy prepared -- train.py --steps 60
```

Fresh generation needs configured providers and available model budget. For a no-model local integration
check, run the following from the root of this checkout with the installed wheel. The output folder must be new:

```bash
python3 tools/validate_flint_metal.py \
  --out /tmp/flint-metal-check --eager --length 64 --exercise-approval
```

This deliberately uses a hand-authored fixture in `tests/package/fixtures/metal_recurrence.py`. It does
not represent an agent-generated kernel. It runs actual checks, qualification and (if faster) isolated
approval/normal dispatch, then restores the receipt to pending review. It prints a command for reviewing
and launching the measured project interactively, without another model call.

## Evidence and limits

Executed on Apple M4, macOS 27.0, PyTorch 2.8.0, using the standalone wheel outside the source checkout:

| Workload | Result |
| --- | --- |
| 8-step recurrence, default `torch.compile`, 60 optimizer steps | 28/28 development and 16/16 fresh held-out cases passed. No reliable speed improvement; reference retained. |
| 64-step recurrence, eager PyTorch, 60 optimizer steps | 28/28 development and 16/16 fresh held-out cases passed. Median work-unit time 16.780 → 2.223 ms (7.55×); command 2.327 → 1.525 s (1.53×). |
| 64-step reference under default `torch.compile` | PyTorch Metal compiler buffer-argument limit; failed baseline, not a speedup. |

Three alternating complete-command pairs followed one preparation command per mode. The eager reference
and candidate final training losses both equal 0.06933704018592834; maximum loss-trace difference in the
first pair is 1.49e-8. This is a small synthetic regression integration test, not an ML-quality benchmark
or evidence of general acceleration. Model calls and rented GPU costs were zero for these fixtures.

The [compiled control](../runs_public/flint-metal/controls/compiled-8-step.json) and the comparison below are separate experiments.

Numerical truth uses CPU float64 because MPS has no float64. Unsupported CPU reference oracles cause
refusal. Actual MPS reference error remains the baseline-aware gradient bound. The checker tests all
returned states with separate cotangents and detects incorrect backward implementations.

Native checks pass schema, fake-tensor and AOT dispatch tests, default Inductor fullgraph execution,
partial gradient masks, and float16/bfloat16 autocast contexts. PyTorch 2.8's autograd-registration
`opcheck` explicitly does not support MPS; the receipt records that unsupported check rather than calling
it a pass, and compiled numerical gradient checks remain mandatory. The fixture's inputs are float32;
autocast-context success is not evidence of half/bfloat input support.

MPS has no exact peak-memory counter in the installed PyTorch. Flint samples live allocation every
1 ms in controlled processes and records `memory_measurement.kind=mps_sampled_allocated` and
`exact=false`. Samples can miss transient allocations. All MPS admissions require explicit review,
even with lower sampled memory; this cannot certify a hard memory ceiling. The eager candidate's
median sampled allocation was 398,848 bytes versus 366,592 bytes for reference. The approval prompt
uses conservative max-candidate/min-reference values, not these medians. Remembering a numeric memory
allowance cannot bypass this uncertainty.

Source-level GPU profiling is still CUDA-only: on MPS use the explicit decorator. Arbitrary models,
shapes/dtypes, serving adapters and distributed activation are not established by these tests.
The fixture results above use no model calls. Fresh model generation is documented separately below.

Primary APIs: [PyTorch Metal shader compilation](https://docs.pytorch.org/docs/main/generated/torch.mps.compile_shader.html),
[PyTorch custom operators](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html).

## Fresh Spark Open + Wonder generation

The separately authorized $1 experiment used a fresh cache and the same shared cumulative ledger.
Spark Open wrote the initial forward candidate and three attempted repairs. All four were rejected.
The controller identified grouped Metal shader arguments, clarified the prompt, and added an AST-based
API diagnostic to failed development checks in Flint 0.10.1. No candidate code was edited by hand;
no tolerance, numerical check, or held-out feedback policy changed.

Wonder repaired the forward implementation and wrote registered forward/backward custom operators.
The final artifact SHA-256 is
`2353c8fab100648a655a4c9cc52b512ba85036209ab0bc006d2f6a62e0912d27`.
It passed **28/28 development checks and 16/16 fresh held-out checks**, including compiled native
output/gradient checks. The installed PyTorch's unsupported MPS registration opcheck remains explicitly
recorded as described above.

For `train.py --steps 60 --length 64 --eager`, after preparation and three alternating pairs:

- Median optimizer-inclusive work unit: **15.011 → 1.137 ms (13.20×)**.
- Median complete command: **2.034 → 1.164 s (1.75×)**.
- Final loss: **0.06933704018592834** for both paths; maximum difference across the first paired
  loss traces: **1.49e-8**.
- Conservative maximum-candidate/minimum-reference sampled allocation difference:
  **+64,512 bytes (+63 KiB, +19.30%)**. This is sampled live allocation, not an exact peak.

This is an eager-PyTorch comparison on one synthetic workload. It is not a claim of beating
`torch.compile`, universal acceleration, or architecture discovery. The earlier short compiled
comparison and failed long compiled baseline are retained above.

Six model calls cost **$0.5117819** in total; rented GPU cost was **$0**. Cumulative ledger spending
rose from $6.09858705 to $6.61036895, with no outstanding reservations. The task-specific cap was
$7.09858705; the saved personal cap was not changed.

Both terminal choices were tested. Approval launched the verified kernel for the ordinary training command. The test restored the pending receipt afterward. [Generated run and control evidence](../runs_public/flint-metal/). Local receipts are bound to the original environment; a new installation requires fresh qualification.
