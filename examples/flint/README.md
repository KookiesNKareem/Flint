# Flint recurrence example

A byte language model with connected fast and slow recurrent memories. The transfer gate depends
on their agreement; changing that expression changes both forward computation and its gradients.
The ordinary training loop uses full-graph `torch.compile`, BF16 autocast, AdamW and two-microbatch
accumulation. The included Shakespeare text is a small experimental corpus; 60 updates are a
numerical/performance integration check, not evidence of language-model quality.

After installing Flint and running `flint setup` once, use a CUDA environment:

```bash
flint run -- train.py --steps 60 --data dataset.txt
```

The only acceleration-specific model code is `@flint.fast` in `recurrence.py`.
The saved launch policy determines whether preparation precedes training or uses a controlled
work boundary. Qualification replays this command and writes disposable files under `.flint/`.
Edit the Python recurrence and rerun; unsupported signatures retain the reference.

For a fresh single-model build, set a new shared-cache directory in your project config, then:

```bash
flint run --model meta-models/muse-spark-open-1026 -- train.py --steps 60 --data dataset.txt
```

That option applies to new generation/repair. Existing accepted cache entries keep their original
model provenance. Provider credentials remain on the controller; running an accepted artifact
on the CUDA host does not need them. See `runs_public/flint-decorator/` for the executed comparison
against default `torch.compile` and its scope.
