# Generated Metal kernel on Apple M4

Spark Open’s first forward candidate failed the equation checks. Wonder used the diagnostic to repair its Metal buffer bindings and generate backward. See [per-attempt events and source](generation/), [controller changes](controller-changes.json), and the [checked kernel](artifact/kernel.py). The final code passed 28 development checks and 16 fresh holdouts. Six model calls cost $0.5117819.

The [qualification receipt](qualification/qualification.json) compares eager PyTorch and Flint in three pairs. All six 60-step traces are beside the receipt. Median optimizer steps were 13.20× faster; the entire command was 1.75× faster. At 100 ms of optimizer-step time, the median observed losses were 0.072 and 0.235 for Flint and PyTorch; both ended at 0.069. MPS allocation is sampled. An increase of 63 KiB needed review. See [normal dispatch](normal-verification.json), [summary](summary.json), and [failed 64-step compiler baseline](controls/compiled-64-step-error.txt) and [other controls](controls/).

The [example source](../../examples/metal/) is provided for reproduction. The historical receipt is not authorization to activate the kernel on a different host.
