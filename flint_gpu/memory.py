"""Measure an actual differentiable call, including its backward workspace."""
import torch
from kernel_contract.tree import leaves


def loss_of_outputs(output):
    return sum(value.float().sum() for _, value in leaves(output)
               if isinstance(value, torch.Tensor) and value.is_floating_point())


def clear_grads(args, kwargs):
    for _, value in leaves((args, kwargs)):
        if isinstance(value, torch.Tensor) and value.is_leaf:
            value.grad = None


def measure(fn, args, kwargs):
    """Peak is allocated CUDA bytes above resident inputs, not reserved allocator space.

    Saved activation storage excludes input storage and counts aliases once. Peak memory
    also catches workspace or ctx attributes that bypass saved_tensors_hooks.
    """
    tensors = [x for _, x in leaves((args, kwargs)) if isinstance(x, torch.Tensor)]
    device = tensors[0].device
    input_storage = {x.untyped_storage().data_ptr() for x in tensors}
    saved = {}
    def pack(x):
        storage = x.untyped_storage()
        pointer = storage.data_ptr()
        if pointer not in input_storage:
            saved[pointer] = max(saved.get(pointer, 0), storage.nbytes())
        return x
    clear_grads(args, kwargs)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
    from .device import Memory
    monitor=Memory(device).start() if device.type=='mps' else None
    baseline_mps=torch.mps.current_allocated_memory() if monitor else 0
    try:
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            output = fn(*args, **kwargs)
            loss_of_outputs(output).backward()
    finally:
        sampled=monitor.finish() if monitor else None
    result = {'saved_activation_bytes': sum(saved.values()), 'saved_storages': len(saved)}
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        result['peak_extra_bytes'] = max(0, torch.cuda.max_memory_allocated(device) - baseline)
    if sampled:result.update(sampled_extra_bytes=max(0,sampled['peak_allocated_bytes']-baseline_mps),memory_measurement=sampled['memory_measurement'])
    clear_grads(args, kwargs)
    return result
