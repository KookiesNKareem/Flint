# Flint on a Mac

Requires Apple Silicon, working PyTorch MPS, and `torch.mps.compile_shader` (tested with PyTorch
2.8.0). The example trains two coupled recurrent states with ordinary autograd and AdamW;
its default command uses `torch.compile(fullgraph=True)` on MPS.

After installing Flint and configuring your providers/budget once:

```bash
flint setup --remote 'bash -c'
flint run --policy prepared -- train.py --steps 60
```

Flint targets Metal Shading Language on this host and registers native PyTorch custom operators,
fake implementations, and autograd. Spark Open and Wonder remain the configured generation route.
The script chooses MPS explicitly; Flint does not silently move a CUDA model or its tensors.

Generation still consumes the configured model budget. To test the backend without model calls,
run `tools/validate_flint_metal.py --out /absolute/new-directory` with the installed wheel.
That tool uses a clearly marked hand-authored fixture, performs real development/holdout/compiler
checks and repeated workload comparisons, and leaves any faster candidate awaiting memory review.
It is integration evidence, not proof of model-generated Metal code or general acceleration.

MPS memory is sampled live tensor allocation, not an exact allocator peak. Flint shows that limitation
and requires explicit `[Y/n]` review even when sampled memory decreases. Fixed memory ceilings cannot
be certified from sampling. Source-level GPU profiling/discovery remains CUDA-only; on a Mac add
`@flint.fast` to the function you want to investigate. Unsupported signatures use your reference.

The compiled example defaults to eight timesteps: PyTorch 2.8's default Metal compiler hit its
buffer-argument limit on the 64-step reference during validation. That baseline failure is not a
Flint speedup. For longer sequences use the script's explicit `--eager --length 64` option;
Flint never silently changes the compiler baseline.
