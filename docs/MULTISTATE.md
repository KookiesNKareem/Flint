# Multiple recurrent states

The completed L4 run is recorded in [the retained run summary](../runs_public/multistate/summary.json):
32 development and 24 fresh holdout cases passed; three-seed training used the generated implementation
without fallback. Median training steps were 21x faster than the eager loop with 48% lower total peak
allocated memory. A matched seed-0 activation-checkpointed reference was also measured.

`examples/muse_kernel/two_memory.py` changes the recurrence itself: a delta update writes a fast matrix
memory, and a gate depending on both memories consolidates a slow matrix memory. Both final states are
returned and feed the next sequence chunk without detaching. The model learns both initial states.
This is an implementation workflow example, not evidence that this architecture is better.

```python
@flint.fast(
    models=['meta-models/muse-spark-open-1026'],
    backward_model='muse-wonder-1.0',
    memory_budget_mb=4,
)
def recurrence(q, k, v, decay, beta, transfer, state, scale):
    fast, slow = state['fast'], state['slow']
    # Researcher's PyTorch recurrence.
    return outputs, {'fast': fast, 'slow': slow}
```

Inputs can contain nested tuples, lists, and dictionaries with string keys. Outputs must be tensors or
those containers of tensors. Exact captured input shape, stride, offset, dtype, device, scalar and grad
signatures guard dispatch. Captured value domains are checked again on each call. New signatures use
the reference until explicitly captured in a new build. Keyword-argument calls currently use the reference.

The checker compares every output against the reference and float64 computation, then checks input
gradients with joint and separate output cotangents. Final-state-only objectives catch detached state
outputs and incorrect initial-state derivatives. Multiple-output dtypes and container structure must match;
input mutation is rejected. Numerical checks remain empirical, not a proof of equivalence.

Spark Open writes the forward implementation; Wonder receives its verified source and builds the backward.
If Spark Open fails its checks, Wonder receives the candidate and counterexamples instead. Both providers
share the remote runner's model spending cap, and each call and candidate is recorded. This mixed-model
workflow should not be described as exclusively open-weight.

The memory budget is **peak extra CUDA allocated bytes for forward plus backward above resident inputs**,
including output tensors, retained activations, input gradients, and backward workspace. It is measured on
every captured signature and enforced as an acceptance condition. Saved activation storage is reported
separately, deduplicated by storage and excluding inputs. Neither metric means total model memory or CUDA
reserved memory. The backward writer chooses checkpointing/recomputation; one additional memory
optimization round is attempted by default. Only a passing candidate with lower measured peak replaces
the incumbent. A fresh, post-selection holdout runs before publication.

## Run the training example

Use a CUDA host with this repository installed and a shell wrapper accepting one remote command. Keep
`MODEL_API_KEY` and `BASETEN_API_KEY` on the controller, never in the uploaded training script.

```sh
python -m flint_gpu.remote_run examples/muse_kernel/two_memory.py \
  --remote 'bash /absolute/path/to/gpu-wrapper.sh' \
  --remote-dir /workspace/recurrence --out runs/two_memory \
  --max-usd 6 --minutes 90 --model meta-models/muse-spark-open-1026
```

The example captures training and inference signatures, waits for verification, compares every whole-model
parameter gradient, and runs matched reference/kernel byte-language-model training on Tiny Shakespeare.
It uses BF16 autocast, two 128-token connected chunks, two accumulated microbatches per optimizer step,
24 steps and three seeds, a fixed validation split, and an on-disk checkpoint/resume comparison. It asserts
actual kernel routing and fails on unexpected fallback. Accumulation uses ordinary PyTorch; automatic
backward recovery is not enabled in this example. Generation pauses training so checker timing is isolated.

The runner saves `training_results.json`, candidate source, source hashes, environment, check receipts,
model usage and logs locally. The checkpoint remains beside the remote script and should be retrieved
before deleting an ephemeral worker. Revalidate a saved cache on a different GPU/software environment
with `python -m flint_gpu.revalidate CACHE_DIRECTORY` before using it.

Cache schema 3 invalidates earlier caches because their checker did not verify every returned tensor.
Existing caches need revalidation before the decorator will dispatch to them.
