# Event contribution manifest

Upstream base: `e0647170da36ef9b059ac0bd3d60103aa4ed378b` from Meta's KernelAgent. The upstream agent,
Fuser, optimization pipeline, prior UI, oink, and their original tests remain credited to their authors.
They are not new hackathon features.

New event work is in:

- `kernel_contract/`: reference contracts, numerical/gradient checking, recurrence generation/repair,
  benchmark/training evaluation, event dashboard, and the research workbench.
- `flint_gpu/` (with a `muse_kernel` compatibility import): reference observation, native compiler integration, guarded dispatch, environment-bound caches, training recovery, and bounded qualification/repair orchestration. The three-repository 0.4 trial is recorded in `runs_public/flint-loop/`; rejected candidates remain explicitly labeled.
- `utils/providers/meta_provider.py` and `budget.py`: Meta model adapters and shared spend accounting.
- `examples/muse_kernel/`, `.agents/skills/fast-recurrence/`, and corresponding new tests.
- `runs_public/`: recorded model calls, candidates, checks and training evidence. Historical evidence is
  explicitly distinguished from newly executed workbench validation.
- New README material, workbench documentation and visualization assets. Upstream material remains labeled.

The official event page begins October 26, 2026; the participant guide separately lists the building period
as October 27–November 4. Commit history preserves the actual October 26 start of local event work. Do not
change timestamps or describe preexisting upstream features as event contributions. Submission materials
should identify the exact demonstrated commits and clarify event timing with organizers if required.

Triadic GDN is prior research with an official H100/sm90 implementation. This project evaluates generated
Triton implementations for its own reference on an L4. It does not claim invention of Triadic GDN, absence
of prior kernels, or equivalence to every feature of the authors' implementation.

Flint 0.8's decorator/compiler benchmark and single-model provenance work are recorded separately in
`runs_public/flint-decorator/`. The mixed-model kernel is reused from the 0.7 lineage and freshly
revalidated; it is not relabeled as a newly generated kernel. Spark-only calls retain their provider
response IDs, failures and spend. No generated Triton source is hand-edited.

Flint 0.9 separates measured resource tradeoffs from correctness failures. It adds receipt-bound
`flint approve`, remembered memory allowances, and review-before-repair spending for faster kernels
with higher memory. Correctness, speed evidence and workload/artifact guards remain mandatory.
See `runs_public/flint-approval/` for a fresh CUDA qualification, explicit approval and normal dispatch.

The Flint 0.10.1 Metal backend, repaired Metal generation, and compiled controls are recorded in [Metal evidence](../runs_public/flint-metal/). Packaging and curation for the submission are in 0.10.2; benchmark numbers retain the tested runtime versions.
