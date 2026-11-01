"""Trace-time capture only. The resulting graph contains tensor operations, not orchestration."""
import atexit
import json
import os
from pathlib import Path
import torch
from kernel_contract.tree import map_tree

REGISTRY = {}


def supported_inputs(tree):
    if isinstance(tree, (torch.Tensor, int, float, bool, str)) or tree is None: return True
    if isinstance(tree, (tuple,list)): return all(supported_inputs(v) for v in tree)
    if isinstance(tree, dict): return all(isinstance(k,str) and supported_inputs(v) for k,v in tree.items())
    return False


def tensor_outputs(tree):
    if isinstance(tree,torch.Tensor): return True
    if isinstance(tree,(tuple,list)): return len(tree)>0 and all(tensor_outputs(v) for v in tree)
    if isinstance(tree,dict): return len(tree)>0 and all(isinstance(k,str) and tensor_outputs(v) for k,v in tree.items())
    return False


def metadata(tree):
    if isinstance(tree, torch.Tensor):
        if torch.compiler.is_compiling():
            torch._dynamo.mark_static(tree)
        return {'tensor': {'shape': tuple(tree.shape), 'dtype': str(tree.dtype),
                           'device': str(tree.device), 'stride': tuple(tree.stride()),
                           'requires_grad': tree.requires_grad}}
    if isinstance(tree, tuple): return {'tuple': tuple(metadata(x) for x in tree)}
    if isinstance(tree, list): return {'list': tuple(metadata(x) for x in tree)}
    if isinstance(tree, dict): return {'dict': {k: metadata(v) for k,v in tree.items()}}
    return {'scalar': tree}


def signature_from_metadata(meta):
    if 'tensor' in meta:
        return {**meta['tensor'], 'shape': list(meta['tensor']['shape']),
                'stride': list(meta['tensor']['stride']), 'offset': 0}
    if 'tuple' in meta: return {'tuple': [signature_from_metadata(x) for x in meta['tuple']]}
    if 'list' in meta: return {'list': [signature_from_metadata(x) for x in meta['list']]}
    if 'dict' in meta: return {'dict': {k: signature_from_metadata(x) for k,x in meta['dict'].items()}}
    return meta


def meta_arguments(meta):
    if 'tensor' in meta:
        m = meta['tensor']
        return torch.empty_strided(m['shape'], m['stride'], device='meta',
                                  dtype=getattr(torch, m['dtype'].split('.')[-1]), requires_grad=m['requires_grad'])
    if 'tuple' in meta: return tuple(meta_arguments(x) for x in meta['tuple'])
    if 'list' in meta: return [meta_arguments(x) for x in meta['list']]
    if 'dict' in meta: return {k: meta_arguments(x) for k,x in meta['dict'].items()}
    return meta['scalar']


@torch.compiler.assume_constant_result
def capture_trace(key, meta, output_structure):
    runtime = REGISTRY[key]
    runtime._observe_metadata(meta, output_structure, trace=True)
    return True


@torch.compiler.assume_constant_result
def trace_event(key, kind):
    REGISTRY[key].stats[kind] += 1
    return True


def matches(args, signature):
    """Metadata guards; accepted native artifacts must pass all offset/layout probes."""
    def match(value, desc):
        if 'shape' in desc:
            return (isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(desc['shape'])
                    and str(value.dtype) == desc['dtype'] and str(value.device) == desc['device']
                    and tuple(value.stride()) == tuple(desc['stride']) and value.requires_grad == desc['requires_grad'])
        if 'tuple' in desc:
            return isinstance(value, tuple) and len(value)==len(desc['tuple']) and all(match(x,d) for x,d in zip(value,desc['tuple']))
        if 'list' in desc:
            return isinstance(value, list) and len(value)==len(desc['list']) and all(match(x,d) for x,d in zip(value,desc['list']))
        if 'dict' in desc:
            return isinstance(value, dict) and set(value)==set(desc['dict']) and all(match(value[k],d) for k,d in desc['dict'].items())
        return value == desc['scalar']
    return len(args)==len(signature) and all(match(v,d) for v,d in zip(args,signature))


def write_report():
    path = os.environ.get('FLINT_REPORT',os.environ.get('MUSE_REPORT'))
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps([v.status() | {'name':v.fn.__qualname__, 'hash':v.hash,
                                             'job':str(v.job), 'native_loaded':v.native_loaded}
                                          for v in REGISTRY.values()], indent=2))

atexit.register(write_report)
