# Earlier Flint experiments

The measurements in the submission were collected with the versions and hardware recorded in their receipts. The 0.10.3 package unifies the distribution, import, and CLI under `flint`. This packaging release did not regenerate the benchmark kernels.

| Stage | What was exercised | Retained evidence |
| --- | --- | --- |
| KernelAgent extension | Equation checking, forward/backward repairs, a fixed-budget recurrence workbench, and Triadic GDN on an L4 | [Triadic trials](../runs_public/triadic/), [training](../runs_public/train_triadic/), [workbench](../runs_public/workbench/), [benchmark trial outcomes](../runs_public/bench1/trials.json) |
| Early decorator, through 0.2 | Native operator integration, checkpoint resume, multiple states, and recomputation | [Portability](../runs_public/portability/), [multistate](../runs_public/multistate/) |
| 0.3–0.5 | Workload qualification, budgeted repair, discovery, and source suggestions | [Qualification](../runs_public/flint/results.json), [three external repos](../runs_public/flint-loop/final-results.json), [discovery](../runs_public/flint-discovery/results.json) |
| 0.6–0.7 | Personal setup, adaptive observation and safe activation at training or batch boundaries | [Adaptive results](../runs_public/flint-adaptive/results.json) |
| 0.8–0.9 | RTX 4090 compiled benchmark, Spark Open provenance, and explicit memory tradeoff review | [CUDA](../runs_public/flint-decorator/), [approval](../runs_public/flint-approval/summary.json) |
| 0.10.1 | Metal generation, held-out checks, measured approval and normal training dispatch | [Metal](../runs_public/flint-metal/) |

The original experiment README, dated logs and transcripts can be inspected in the [pre-cleanup tree](https://github.com/KookiesNKareem/Flint/tree/2d5e1e83b347b86d064298bc8333efc9981e861e). The [upstream README](https://github.com/meta-pytorch/KernelAgent/blob/e0647170da36ef9b059ac0bd3d60103aa4ed378b/README.md) documents KernelAgent before this work.
