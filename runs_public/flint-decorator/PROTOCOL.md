# Flint 0.8 decorator qualification

One two-state byte language model: fast delta-rule memory, slow consolidated memory, and a state-dependent transfer gate. The model uses BF16 autocast, two accumulated microbatches per optimizer update, and two connected sequence chunks. The generated operator must preserve outputs, both final states and all input gradients. Fixed contract_fuzz_v2 / reference_bounded_v1 checks are unchanged.

Primary comparison: PyTorch 2.8 `torch.compile(model, fullgraph=True, dynamic=False)` with its default mode, with and without the generated recurrence implementation. Same data, seed, model, optimizer, 60 updates and validation batch. Dataset content is bound in the qualification receipt. Preparation is reported separately; three fixed alternating process pairs follow preparation. Report complete process latency, median steady optimizer-update latency, peak allocated memory, per-update losses and validation loss. Do not call default compile the best possible tuned implementation.

Admission retains the existing 5% all-samples margin on whole-command and steady-unit latency, with no measured peak-memory increase. Separately check finite training losses, <=1% relative per-update loss discrepancy, <=1% relative validation-loss discrepancy, and report final parameter differences. This short run is a numerical integration check, not a model-quality study.

Mixed-model lane reuses the model-written Spark Open/Wonder artifact from Flint 0.7 and freshly revalidates it. No generated Triton is hand-edited.

Spark-only lane starts from an empty artifact cache and supplies no prior generated candidate or handoff. Only `meta-models/muse-spark-open-1026` may author forward, backward, correctness repairs and performance repairs. All calls retain model IDs, usage, cost, outputs and checker results. The initial shared model cap was $6 (initial spend $4.6580873). After the remaining reservation blocked repairs, the user approved a cap of $8 specifically for Spark-only completion; budget-authorization.json records that change, and prior ledger spend was retained. GPU rental is separate; use one RTX 4090 and delete it after collecting evidence.

An initial package-02 screening run supplies the cold compile measurement. The final qualification uses the same prepared compiler caches and saves each run's checkpoint separately. Retain screening outcomes rather than silently discarding them.
