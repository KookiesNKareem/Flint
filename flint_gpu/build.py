"""Background build for one @fast function: stage it on the GPU host, have a configured model write and repair a
forward+backward kernel against the checker, and record the verified result in status.json.

  python -m flint_gpu.build JOB_DIR
"""

from __future__ import annotations

import json
import hashlib
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def status(job: Path, **kw):
    cur = json.loads((job / "status.json").read_text()) if (job / "status.json").exists() else {}
    cur.update(kw, updated=time.time())
    tmp = job / "status.json.tmp"
    tmp.write_text(json.dumps(cur, indent=1))
    tmp.replace(job / "status.json")


def remote_prefix() -> list[str]:
    from .config import read_config
    return shlex.split(read_config()["remote"])


def remote_dir() -> str:
    from .config import read_config
    return read_config()['worker_dir']


def run_remote(cmd: str, stdin: bytes | None = None) -> subprocess.CompletedProcess:
    limit = int(os.environ.get("MUSE_KERNEL_REMOTE_TIMEOUT", 7200))
    python=shlex.quote(sys.executable) if remote_prefix()==['bash','-c'] else 'python'
    if remote_prefix()==['bash','-c']:cmd=cmd.replace('&& python ', '&& '+python+' ')
    cmd = f"{python} -m flint_gpu.deadline {limit} bash -c {shlex.quote(cmd)}" if os.environ.get("MUSE_KERNEL_REMOTE_TIMEOUT") else cmd
    return subprocess.run(remote_prefix() + [cmd], input=stdin, capture_output=True, check=True, timeout=limit + 15)


def stage(job: Path, J: dict) -> str:
    """Put the function's module and the contract where the checker imports them; return the remote job dir."""
    rd, h = remote_dir(), J["hash"]
    rjob = f"{rd}/muse_jobs/{h}"
    run_remote(f"mkdir -p {shlex.quote(rjob)} && touch {shlex.quote(rd + '/muse_jobs/__init__.py')}")
    for local, dest in ((job / "reference.py", f"{rd}/muse_jobs/j_{h}.py"), (job / "spec.json", f"{rjob}/spec.json"),
                        (job / "extra_specs.json", f"{rjob}/extra_specs.json")):
        run_remote(f"cat > {shlex.quote(dest)}", local.read_bytes())
    return rjob


def publish(job, J, rjob, done, last, usd, models):
    """Publish only a hash-bound checked candidate after fresh post-selection checks."""
    from kernel_contract.evidence import sha256, eligible
    from kernel_contract.cases import FUZZ_POLICY
    source=Path(done['backward'])
    if last.get('sha256')!=sha256(source) or not eligible(last['results']):
        raise ValueError('Candidate changed or does not have passing development evidence')
    if J.get('native') and not (last.get('native_checks') or {}).get('passed'):
        raise ValueError('Native compiler checks missing')
    kernel = job / "kernel.py"
    kernel.write_text(Path(done["backward"]).read_text())
    from kernel_contract.evidence import coverage, sha256
    (job / "checks.json").write_text(json.dumps(last, indent=2))
    holdout_passed = False
    if J.get("require_holdout"):
        from kernel_contract.holdout import holdout_cases
        protocol = holdout_cases(None)
        protocol["candidate_sha256"] = sha256(kernel)
        (job / "holdout_protocol.json").write_text(json.dumps(protocol, indent=2))
        for name, content in (("selected.py", kernel.read_bytes()), ("holdout.json", json.dumps(protocol).encode())):
            run_remote(f"cat > {shlex.quote(rjob + '/' + name)}", content)
        check = run_remote(f"cd {shlex.quote(remote_dir())} && " + shlex.join([
            "python", "-m", "flint_gpu.remote_eval", rjob, rjob + "/selected.py", "--grads", "--holdout", rjob + "/holdout.json"]))
        result = json.loads(next(line[7:] for line in check.stdout.decode().splitlines() if line.startswith("RESULT ")))
        result.update(coverage=coverage(result["results"]), candidate_sha256=sha256(kernel))
        (job / "holdout_results.json").write_text(json.dumps(result, indent=2))
        if not result["coverage"]["all_cases_passed"]:
            return status(job, state="failed", usd=usd, reason="Fresh holdout failed; no repair feedback", holdout_passed=False)
        holdout_passed = True
    return status(job, state="ready", fuzz_policy=FUZZ_POLICY, gradient_policy=J.get("gradient_policy", "strict_relative_v1"), holdout_passed=holdout_passed, kernel=kernel.name, kernel_sha256=sha256(kernel),
                  coverage=coverage(last["results"]), model=models[-1], fwd_bwd_ms=done.get("fwd_bwd_ms"),
                  usd=usd, memory=done.get("memory"), reference_memory=done.get("reference_memory"), models_used=models, verified_cases=sum(r["outcome"] == "accept" for r in last["results"]),
                  cases=len(last["results"]), compile_passed=bool((done.get('native_checks') or {}).get('passed')),
                  native_checks=done.get('native_checks'))

def main(job: str):
    job = Path(job)
    J = json.loads((job / "job.json").read_text())
    status(job, state="building", started=time.time(), reason=None)
    try:
        rjob = stage(job, J)
        p = run_remote(f"cd {shlex.quote(remote_dir())} && python -m flint_gpu.remote_eval --reference {shlex.quote(rjob)}")
        J["eager_ms"] = float(next(x for x in p.stdout.decode().splitlines() if x.startswith("REFERENCE_MS")).split()[1])
        (job / "job.json").write_text(json.dumps(J, indent=1))
        status(job, reference_ms=J["eager_ms"])
        env_result = run_remote(f"cd {shlex.quote(remote_dir())} && python -c " + shlex.quote(
            "import json; from flint_gpu.identity import environment; print(json.dumps(environment()))"))
        J["target_environment"] = json.loads(env_result.stdout)
        J.setdefault('capture_environment', J['environment'])
        J['environment'] = J['target_environment']
        (job / "job.json").write_text(json.dumps(J, indent=1))
        usd, prior = 0.0, None
        models = J.get('models') or [J['model']]
        if J.get('backward_model'):
            models = [models[0], J['backward_model']]
        for i, model in enumerate(models):
            allowance = J["max_usd"] - usd if J.get("total_budget") else J["max_usd"]
            if allowance <= 0:
                break
            run = job / ("run" if i == 0 else f"run{i + 1}")
            status(job, model=model, attempt=i + 1)
            cmd = [sys.executable, "-m", "kernel_contract.paper_agent", "--job", str(job), "--remote-job", rjob,
                   "--model", model, "--max-usd", str(allowance), "--out", str(run),
                   "--remote", " ".join(shlex.quote(x) for x in remote_prefix()), "--remote-dir", remote_dir(),
                   "--opt-rounds", str(J.get("opt_rounds", 1)), "--bwd-opt-rounds", str(J.get("bwd_opt_rounds", 1)),
                   "--repairs", str(J.get("repairs", 3)), "--max-tokens", str(J.get('max_tokens', 32000))]
            if J.get('memory_policy'):
                cmd[cmd.index('--bwd-opt-rounds') + 1] = str(J['memory_policy'].get('opt_rounds', 1))
            if J.get('backward_model') and i == 0:
                cmd += ['--forward-only']
            if prior is not None:  # the next model inherits the verified forward or the counterexamples
                cmd += ["--handoff", str(prior)]
            with open(job / "build.log", "a") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            events = [json.loads(x) for x in (run / "events.jsonl").read_text().splitlines()] \
                if (run / "events.jsonl").exists() else []
            usd += sum(e.get("usd", 0) for e in events if e["event"] == "llm_done")
            done = next((e for e in reversed(events) if e["event"] == "done"), None)
            if done and done.get("ok") and done.get("backward"):
                last = next(e for e in reversed(events) if e["event"] == "eval" and e["path"] == done["backward"])
                return publish(job, J, rjob, done, last, usd, models[:i+1])
            if any(e["event"] == "eval" for e in events):
                prior = run
        status(job, state="failed", usd=usd, reason="no kernel passed the checker within the budget")
    except Exception as e:  # the training run must never be affected by a failed build
        status(job, state="failed", reason=f"{type(e).__name__}: {e}"[:500])


if __name__ == "__main__":
    main(sys.argv[1])
