# Flint runtime

Build or install **flint** from the repository root with `python -m pip install .`. For a generation controller, install `python -m pip install '.[build]'`. Import `flint` and run `flint --help`. See the [quickstart](../../README.md) and [runtime guide](../../docs/FLINT.md).

For a redistributable wheel and source archive, run `python -m build` from the root. The source build stages the private checker and provider modules so the wheel can be installed without a KernelAgent checkout. The `flint_gpu` and `muse_kernel` compatibility imports remain available.
