"""Verify the curated submission without model calls, paid GPUs or external repos."""
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "runs_public"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, label):
    if not condition:
        raise ValueError(label)


def verify_files():
    for entry in (EVIDENCE / "SHA256SUMS").read_text().splitlines():
        expected, name = entry.split("  ", 1)
        check(digest(EVIDENCE / name) == expected, name)


def verify_lane(folder, lane=None):
    metal = lane is None
    results = folder / "qualification" if metal else folder / "results" / lane
    q = read(results / "qualification.json")
    pairs = q["pairs"]
    check(len(pairs) == 3, results)
    inputs = ROOT / "examples" / ("metal" if metal else "flint")
    for filename, expected in q["workload"]["files"].items():
        name = Path(filename).name
        if lane == "mixed" and name in {"._model.py", "._train.py", "._recurrence.py", "._pyproject.toml"}:
            continue  # Historic macOS sidecar hashes remain in the unmodified receipt.
        path = inputs / name
        if not metal and name == "pyproject.toml":
            path = folder / "projects" / lane / name
        check(digest(path) == expected, path)
    artifacts = pairs[0]["benchmark"]["artifacts"]
    check(len(artifacts) == 1, results)
    name, expected = next(iter(artifacts.items()))
    artifact = folder / "artifact" if metal else folder / "artifacts" / lane / name
    state = read(artifact / "status.json")
    check(digest(artifact / "kernel.py") == state["kernel_sha256"] == expected, artifact)
    check(state["compile_passed"] and state["holdout_passed"], artifact)
    for pair in pairs:
        check(pair["reference"]["success"] and pair["benchmark"]["success"], results)
        check(pair["benchmark"]["artifacts"] == artifacts, results)
    def speed(metric):
        def value(sample):
            return sample["seconds"] if metric == "command" else sample["work_units"]["median_seconds"]
        return median(value(p["reference"]) for p in pairs) / median(value(p["benchmark"]) for p in pairs)
    summary = read(folder / ("summary.json" if metal else lane + "-summary.json"))
    if metal:
        check(not q["accepted"] and q["approval_available"], results)
        dispatch = read(folder / "normal-verification.json")["result"]
        check(dispatch["artifacts"] == artifacts and dispatch["dispatch"][0]["calls"] == {"kernel": 60}, results)
        keys = ("command_speedup", "work_unit_speedup")
    else:
        check(q["accepted"] and summary["application_passed"], results)
        dispatch = read(results / "normal-direct.json")
        check(any(r.get("loaded_sha256") == expected and r.get("performance_qualified") for r in dispatch), results)
        keys = ("seconds_speedup", "steady_update_ms_speedup")
    for metric, key in zip(("command", "step"), keys):
        check(abs(speed(metric) - summary[key]) < 1e-5, key)
    print(f"{lane or 'Metal'}: {speed('step'):.2f}x per update, {speed('command'):.2f}x per command")


def verify_loss():
    measured = {}
    for mode in ("reference", "benchmark"):
        at_100, final = [], []
        for i in range(3):
            run = read(EVIDENCE / "flint-metal/qualification" / f"training-{i}-{mode}.json")
            elapsed, complete = 0, 0
            for duration in run["step_ms"]:
                elapsed += duration
                if elapsed <= 100:
                    complete += 1
            check(complete > 0 and len(run["losses"]) == len(run["step_ms"]) == 60, mode)
            at_100.append(run["losses"][complete-1])
            final.append(run["losses"][-1])
        measured[mode] = median(at_100), median(final)
    ref, fast = measured["reference"], measured["benchmark"]
    check((round(ref[0], 3), round(fast[0], 3)) == (0.235, 0.072) and ref[1] == fast[1], "loss traces")


if __name__ == "__main__":
    verify_files()
    for variant in ("mixed", "spark"):
        verify_lane(EVIDENCE / "flint-decorator", variant)
    verify_lane(EVIDENCE / "flint-metal")
    verify_loss()
