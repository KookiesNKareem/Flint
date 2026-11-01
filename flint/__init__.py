"""Public Python API for Flint."""
from importlib import import_module

__all__ = ["fast", "train_step", "workload"]


def __getattr__(name):
    # SymPy probes for an unrelated optional module named flint while PyTorch
    # initializes. Accessing our API, rather than importing its name, loads torch.
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("flint_gpu"), name)
    globals()[name] = value
    return value
