"""Cases a kernel must pass to honor the contract of the op it replaces.

Axes: seed (redraw the inputs), values (transform every floating input), sizes (perturb each
size symbol), layout (same logical values through a different memory layout). The dense
contract is seed + values + sizes at the observed layout; the strided contract adds layout.
"""

from __future__ import annotations

from .spec import OpSpec
from .tree import descriptors

VALUE_TRANSFORMS = {
    "scale_up": lambda x: x * 1e3,
    "scale_down": lambda x: x * 1e-3,
    "shift": lambda x: x + 100.0,
    "negative": lambda x: -x.abs() - 1.0,
    "half_zero": lambda x: x.clamp(min=0),
    "constant": lambda x: x * 0 + 0.5,
    "scale_mid": lambda x: x * 37.0,
    "shift_neg": lambda x: x - 100.0,
}

FUZZ_POLICY = 'contract_fuzz_v2'

LAYOUTS = ("contiguous", "strided2", "strided3_off1", "transposed", "leading_swap")


def build_cases(spec: OpSpec, contract: str = "strided") -> list[dict]:
    cases: list[dict] = [{"name": "observed", "axis": "seed", "seed": 0}]
    cases += [{"name": f"seed{s}", "axis": "seed", "seed": s} for s in (1, 2)]
    cases += [{"name": t, "axis": "values", "seed": 0, "transform": t} for t in VALUE_TRANSFORMS]
    for s, v in spec.sizes.items():
        for nv in sorted({v + 1, v // 2 + 1} - {v}):
            cases.append({"name": f"{s}={nv}", "axis": "sizes", "seed": 0, "sizes": {s: nv}})
    domains = {t.get("domain") for t in descriptors(spec)}
    if "logdecay" in domains:
        cases += [{"name": n, "axis": "values", "seed": 0, "stress": n} for n in ("decay_none", "decay_strong")]
    if "unit" in domains:
        cases += [{"name": n, "axis": "values", "seed": 0, "stress": n}
                  for n in ("gate_edges", "gate_typical", "gate_high", "gate_mid")]
    if contract == "strided":
        cases += [{"name": lay, "axis": "layout", "seed": 0, "layout": lay} for lay in LAYOUTS]
    if contract == 'strided':
        # Orthogonal axes alone miss candidates whose shortcuts fail only in combination.
        for i,(layout,transform) in enumerate(zip(LAYOUTS,('scale_down','scale_mid','negative','half_zero','constant'))):
            cases.append({'name':f'compound_{i}','axis':'compound','seed':31+i,'layout':layout,'transform':transform})
    return cases
