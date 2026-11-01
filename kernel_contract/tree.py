"""Small, explicit tensor trees shared by capture, dispatch and the checker."""
import json
import torch


def leaves(tree, path=()):
    if isinstance(tree, (tuple, list)):
        for i, value in enumerate(tree):
            yield from leaves(value, path + (i,))
    elif isinstance(tree, dict):
        for key, value in tree.items():
            if not isinstance(key, str):
                raise TypeError('Tensor-tree dictionary keys must be strings')
            yield from leaves(value, path + (key,))
    else:
        yield path, tree


def map_tree(fn, tree):
    if isinstance(tree, tuple):
        return tuple(map_tree(fn, x) for x in tree)
    if isinstance(tree, list):
        return [map_tree(fn, x) for x in tree]
    if isinstance(tree, dict):
        return {k: map_tree(fn, v) for k, v in tree.items()}
    return fn(tree)


def structure(tree):
    if isinstance(tree, torch.Tensor):
        return 'tensor'
    if isinstance(tree, tuple):
        return {'tuple': [structure(x) for x in tree]}
    if isinstance(tree, list):
        return {'list': [structure(x) for x in tree]}
    if isinstance(tree, dict):
        return {'dict': {k: structure(v) for k, v in tree.items()}}
    raise TypeError('Outputs must be tensors or nested tuple/list/string-keyed dicts of tensors')


def path_key(path):
    return path[0] if len(path) == 1 else json.dumps(path, separators=(',', ':'))


def key_path(key):
    return tuple(json.loads(key)) if isinstance(key, str) and key.startswith('[') else (int(key),)


def at(tree, path):
    for key in path:
        tree = tree[key]
    return tree


def descriptors(spec):
    """Walk encoded arguments, without treating descriptor metadata as arguments."""
    def walk(value):
        if isinstance(value, dict):
            if 'tensor' in value:
                yield value['tensor']
            elif 'seq' in value:
                for x in value['seq']:
                    yield from walk(x)
            elif 'map' in value:
                for x in value['map'].values():
                    yield from walk(x)
    for value in list(spec.args) + list(spec.kwargs.values()):
        yield from walk(value)
