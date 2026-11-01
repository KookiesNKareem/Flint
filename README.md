<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/flint-logo-dark.svg">
    <img src="docs/assets/flint-logo.svg" alt="Flint" width="340">
  </picture>
</h1>

Flint turns a PyTorch reference function into a candidate GPU implementation. Add `@flint.fast`, then use your normal training loop. Flint captures the inputs, asks configured model providers for forward and backward kernels, checks outputs and gradients against the reference, and benchmarks the actual workload before allowing activation. Unsupported inputs and unqualified candidates use the Python reference.

## Install

From this repository, with Python 3.10+ and a compatible PyTorch installation:

```bash
python -m pip install .
flint --help
```

Use `import flint` and decorate your function with `@flint.fast`. The distribution and CLI are also named `flint`. Install from this checkout or with `python -m pip install git+https://github.com/KookiesNKareem/Flint.git` in a fresh virtual environment. The [PyPI project named flint](https://pypi.org/project/flint/) is a separate Fortran tool. For fresh model generation, install `python -m pip install '.[build]'` on the controller and run `flint setup` once to configure providers, a worker, launch policy, and budget. Older projects can still import `flint_gpu` or `muse_kernel`.

## Try the decorator

From the checkout, run the six-step CPU example. It needs no provider credentials or rented GPU.

```bash
cd examples/quickstart
flint run --policy observe -- train.py
```

The [example](examples/quickstart/train.py) calls a decorated recurrence, computes a loss, calls `loss.backward()`, and updates parameters with AdamW. The CPU run exercises reference dispatch. For GPU workloads, see the [compiled CUDA training example](examples/flint/) and [Apple Silicon example](examples/metal/). After setup, the normal launch is `flint run -- train.py`.

## Measured result

On one RTX 4090, the mixed Spark Open/Wonder implementation of a two-state byte language model ran optimizer updates **4.47× faster**, the whole training command **2.62× faster**, and used **2.28% less peak allocated memory** than the same model under default `torch.compile`. This was a previously generated kernel revalidated for the workload. A separate Spark Open-only build achieved **3.28×** and **2.25×** respectively. Both comparisons used three paired runs with prepared compiler caches. [CUDA evidence](runs_public/flint-decorator/)

On an Apple M4, a newly generated Metal kernel reduced median optimizer-step time from **15.01 to 1.14 ms** and whole-command time from **2.03 to 1.16 s** against eager PyTorch. It used **63 KiB more sampled memory**, which required explicit approval. [Metal evidence and controls](runs_public/flint-metal/)

## Limitations

Qualification applies to a specific command, input contract, environment, and artifact. It cannot guarantee a speedup on another workload. The Mac result uses an eager baseline; Metal peak allocation is sampled. Flint has also kept PyTorch after failed generation, incorrect gradients, and insufficient performance. The three external-repository trials did not produce three wins. [Retained outcomes](runs_public/flint-loop/final-results.json)

## Evidence and credits

- [Evidence index and integrity checks](runs_public/README.md)
- [Runtime and qualification guide](docs/FLINT.md)
- [Earlier experiments and release history](docs/HISTORY.md)
- [Hackathon contribution manifest](docs/HACKATHON_CONTRIBUTIONS.md)

Flint builds on [Meta’s KernelAgent](https://github.com/meta-pytorch/KernelAgent/tree/e0647170da36ef9b059ac0bd3d60103aa4ed378b). Upstream authors, copyright notices, and the [Apache 2.0 license](LICENSE) are retained.
