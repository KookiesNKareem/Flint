# Contributing to Flint

Thanks for your interest in improving Flint! We aim to make
contributions straightforward, transparent, and welcoming.

## Code of Conduct
By participating, you agree to abide by our Code of Conduct. See `CODE_OF_CONDUCT.md`.

## Ways to Contribute
- Report bugs and performance issues
- Propose features and improvements
- Improve documentation and examples
- Add tests and fix bugs
- Optimize kernels or refine prompts/templates

## Development Setup

Use Python 3.10 or newer. Clone this repository and install the runtime and test dependencies in a virtual environment.

```bash
git clone https://github.com/KookiesNKareem/Flint.git
cd Flint
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[build,test]'
python -m pytest -q tests/package tests/kernel_contract tests/workbench
python tools/verify_submission_evidence.py
```

Tests requiring a GPU skip when the relevant backend is unavailable. Fresh generation uses provider credentials and a shared budget configured with `flint setup`. See the [Flint README](README.md) for the standalone wheel and CPU example.

## Linting and Style

We follow PEP 8 with type hints and docstrings.

- Indentation: 4 spaces
- Line length: aim for ~100 chars where practical
- Imports: standard library, third‑party, then local
- Logging: prefer `logging` over `print`
- Types: use annotations for public APIs

Ruff configuration is included in `pyproject.toml`. If you have `ruff` installed, run:

```bash
ruff check .
```

## Pull Requests
We actively welcome your pull requests. Small, focused PRs are easier to review.

1. Fork the repo and branch from `main`.
2. Make your change with tests where appropriate.
3. Update docs/README/examples if behavior or APIs change.
4. Run tests locally and ensure CI passes.
5. Optionally run `ruff check .` and address issues.
6. Submit a clear PR description with context and rationale.

### PR Checklist
- Tests added or updated when behavior changes
- Docs/examples updated when user‑facing behavior changes
- No unrelated formatting or drive‑by refactors
- CI green (see `.github/workflows/ci.yml`)

## Issues
Use GitHub Issues for bugs and feature requests.

When reporting a bug, please include:
- What you did (minimal reproducible example)
- What you expected to happen
- What actually happened (full error, logs, screenshots as relevant)
- Environment details (OS, Python, GPU/CUDA, package versions)

## Security
If you believe you’ve found a security vulnerability, do not open a public issue.
Please follow the process described in Meta’s [bug bounty program](https://bugbounty.meta.com/).

## Contributor License Agreement (CLA)
Some contributions may require signing a CLA before they can be merged. If
prompted by the bot during your PR, please complete the form here:
https://code.facebook.com/cla

## License
By contributing to Flint, you agree that your contributions will be
licensed under the `LICENSE` file in the root of this repository.

## Notes for Kernel/Template Contributions
- Changes to Jinja2 templates in [`triton_kernel_agent/templates/`](triton_kernel_agent/templates) should be validated with an
  end‑to‑end run (see e2e section) since they affect generation behavior.
- For GPU kernels or performance‑sensitive code, include a short note about
  observed correctness and performance if available.
