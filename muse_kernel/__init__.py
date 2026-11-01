"""Compatibility import for releases before the Flint name."""
import sys
import flint_gpu
sys.modules[__name__] = flint_gpu

# Share already-loaded runtime modules; duplicate config/registry modules split state.
for name,module in list(sys.modules.items()):
    if name.startswith("flint_gpu."):
        sys.modules.setdefault("muse_kernel"+name[len("flint_gpu"):],module)
