"""Fresh post-selection compound cases; never supplied as repair feedback."""
import secrets
from .cases import LAYOUTS,FUZZ_POLICY


def holdout_cases(spec, seed=None, count=16):
    import random
    seed=secrets.randbits(31) if seed is None else seed
    rng=random.Random(seed)
    transforms=('scale_up','scale_down','shift','negative','half_zero','constant','scale_mid','shift_neg')
    cases=[]
    for i in range(count):
        case={'name':f'holdout_{i}','axis':'holdout','seed':rng.randrange(10000,2**31),
              'layout':LAYOUTS[i%len(LAYOUTS)]}
        # Transforms respect declared domains in materialize. No inferred gate assumptions.
        if i%2:case['transform']=rng.choice(transforms)
        cases.append(case)
    return {'seed':seed,'fuzz_policy':FUZZ_POLICY,'policy':'sampled after candidate selection; no repair feedback','cases':cases}
