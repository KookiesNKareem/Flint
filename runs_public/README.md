# Submission evidence

| Evidence | What to inspect |
| --- | --- |
| [RTX 4090](flint-decorator/) | Matched default-compile baseline, mixed and Spark Open-only results, candidates, generation feedback, exact artifact identities and normal dispatch |
| [Apple M4](flint-metal/) | Eager reference, repaired Metal source, all 60 loss values and step times in three pairs, fresh holdout and native checks |
| [Negative controls](flint-metal/controls/) | Short compiled control and a different recurrence that failed fresh generation |
| [External training repositories](flint-loop/) | Pinned repositories, source patches, and rejection/repair outcomes |
| [Memory tradeoff](flint-approval/) | Explicitly approved performance tradeoff and recorded dispatch |
| [Discovery](flint-discovery/) and [adaptive](flint-adaptive/) | Training and batch inference outcomes including fallback |
| [Historical experiments](../docs/HISTORY.md) | Previous workbench, portability, and training experiments |

From the repository root, run `python tools/verify_submission_evidence.py`. This checks the retained files, workload and candidate hashes, paired speedups, application checks and the measured loss comparison without calling providers. These historical receipts identify the environment at collection time. Reusing a kernel on another machine requires compatibility checks and qualification.

This tree omits copied third-party training repositories, duplicated snapshots, model checkpoints, wheel builds and worker caches. The CUDA and Metal training sources used in the receipts are available once under [examples](../examples/). The full pre-cleanup evidence directory was separately archived before curation.
