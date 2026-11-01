"""Measure ordinary PyTorch activation checkpointing using the exact training script.

python examples/muse_kernel/checkpoint_baseline.py PATH_TO_TWO_MEMORY_SCRIPT OUT.json
Run after kernel generation/training has finished, on an otherwise idle CUDA worker.
"""
import importlib.util
import json
from pathlib import Path
import sys
import torch
from torch.utils.checkpoint import checkpoint


def main():
    source = Path(sys.argv[1]).resolve()
    spec = importlib.util.spec_from_file_location('two_memory_baseline', source)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    tokens = example.data('cuda')
    split = int(len(tokens) * .9)

    def checkpointed(*args):
        return checkpoint(example.recurrence.fn, *args, use_reentrant=False)

    # Match the paired run's warmed CUDA/BLAS context before measuring resident memory.
    warm = example.model(checkpointed, 0)
    inputs = example.batch(tokens[:split], 2, 256, torch.Generator(device='cuda').manual_seed(1))
    loss = example.loss_of(warm, inputs)
    loss.backward()
    del loss, warm
    result = {'implementation': 'checkpointed_reference',
              **example.training(checkpointed, 0, tokens[:split], tokens[split:])}
    Path(sys.argv[2]).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'steps'}), flush=True)


if __name__ == '__main__':
    main()
