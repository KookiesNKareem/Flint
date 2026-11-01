# Compiled CUDA training

Both lanes train the [same byte language model](../../examples/flint/) in BF16 with accumulation and connected recurrent states. The baseline is default fullgraph torch.compile on an RTX 4090. Each workload had one cache preparation run in each mode followed by three paired commands. These are bounded measurements on this experiment.

| Build | Optimizer step | Whole command | Peak allocated memory |
| --- | ---: | ---: | ---: |
| [Mixed Spark Open/Wonder](mixed-summary.json), reused and revalidated | 4.47× faster | 2.62× faster | 2.28% lower |
| [Spark Open only](spark-summary.json), controller-assisted | 3.28× faster | 2.25× faster | 2.28% lower |

[Mixed paired receipt](results/mixed/qualification.json) · [Spark paired receipt](results/spark/qualification.json) · [generated kernels and checks](artifacts/) · [per-attempt events](generation/) · [Spark provenance](spark-artifact-proof.json) · [failed capture](failed-capture/summary.json) · [memory tradeoff](tradeoff-artifact-proof.json).

The original lane-specific project settings are in `projects/{mixed,spark}/pyproject.toml`. The matching Python/data files are stored once in `examples/flint/`. See [workload protocol](PROTOCOL.md) and [evidence index](../README.md).
