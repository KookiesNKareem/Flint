"""Two coupled memories: generate kernels, then train a byte language model.

Run through `python -m flint_gpu.remote_run` on a CUDA host. Model API keys
stay on the controller. Results and a resumable checkpoint live beside this file.
"""
import hashlib
import gc
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import flint_gpu as flint
from kernel_contract.train_demo import data, batch, loss_of
from kernel_contract.tree import map_tree


@flint.fast(
    observe_calls=3,
    models=['meta-models/muse-spark-open-1026'],
    backward_model='muse-wonder-1.0',
    memory_budget_mb=4,
)
def recurrence(q, k, v, decay, beta, transfer, state, scale):
    """Delta writes to fast memory; a state-dependent gate consolidates slow memory."""
    q, k, v, decay, beta, transfer = (
        x.to(torch.promote_types(x.dtype, torch.float32))
        for x in (q, k, v, decay, beta, transfer)
    )
    fast, slow = state['fast'], state['slow']
    outputs = []
    for t in range(q.shape[2]):
        key, value = k[:, :, t], v[:, :, t]
        fast = fast * decay[:, :, t].exp()[..., None]
        prediction = (fast * key[..., None]).sum(-2)
        fast = fast + key[..., None] * (
            beta[:, :, t, None] * (value - prediction)
        )[..., None, :]
        agreement = (fast * slow).mean(dim=(-2, -1))
        gate = torch.sigmoid(transfer[:, :, t] + 0.05 * agreement)[..., None, None]
        slow = (1 - gate) * slow + gate * fast
        outputs.append(((fast + slow) * (0.5 * scale * q[:, :, t])[..., None]).sum(-2))
    return torch.stack(outputs, dim=2), {'fast': fast, 'slow': slow}


class TwoMemoryLM(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.emb = nn.Embedding(256, 128)
        self.norm = nn.RMSNorm(128)
        self.proj = nn.Linear(128, 4 * 128 + 8, bias=False)
        self.out = nn.Linear(128, 128, bias=False)
        self.norm2 = nn.RMSNorm(128)
        self.up = nn.Linear(128, 512, bias=False)
        self.down = nn.Linear(256, 128, bias=False)
        self.final = nn.RMSNorm(128)
        self.fast0 = nn.Parameter(torch.randn(1, 4, 32, 32) * .01)
        self.slow0 = nn.Parameter(torch.randn(1, 4, 32, 32) * .01)
        for p in self.parameters():
            if p.ndim == 2:
                nn.init.normal_(p, std=.02)

    def forward(self, ids):
        x = self.emb(ids)
        b, t, _ = x.shape
        q, k, v, a, gates = self.proj(self.norm(x)).split([128] * 4 + [8], dim=-1)
        q, k, v, a = (z.reshape(b, t, 4, 32).transpose(1, 2) for z in (q, k, v, a))
        k = F.normalize(k.float(), dim=-1).to(q.dtype)
        decay = F.logsigmoid(a.float() + 3)
        beta, transfer = gates.float().reshape(b, t, 2, 4).permute(2, 0, 3, 1)
        beta = beta.sigmoid()
        state = {'fast': self.fast0.expand(b, -1, -1, -1).contiguous(),
                 'slow': self.slow0.expand(b, -1, -1, -1).contiguous()}
        chunks = []
        for start in range(0, t, 128):
            inputs = [z[:, :, start:start + 128].contiguous() for z in (q, k, v, decay, beta, transfer)]
            y, state = self.fn(*inputs, state, 32 ** -.5)
            chunks.append(y)  # Keep the graph through both returned states.
        y = torch.cat(chunks, dim=2).transpose(1, 2).reshape(b, t, 128)
        x = x + self.out(y.to(x.dtype))
        a, b = self.up(self.norm2(x)).chunk(2, -1)
        x = x + self.down(F.silu(a) * b)
        return F.linear(self.final(x), self.emb.weight)


def model(fn, seed):
    torch.manual_seed(seed)
    return TwoMemoryLM(fn).cuda()


def update(net, optimizer, train, rng):
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for _ in range(2):
        loss = loss_of(net, batch(train, 2, 256, rng))
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        (loss / 2).backward()
        losses.append(float(loss.detach()))
    norm = nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return sum(losses) / 2, float(norm)


def validation(net, val):
    rng = torch.Generator(device='cuda').manual_seed(42)
    with torch.no_grad():
        return statistics.mean(float(loss_of(net, batch(val, 2, 256, rng))) for _ in range(8))


def training(fn, seed, train, val, resume=False):
    gc.collect()
    resident = torch.cuda.memory_allocated()
    net = model(fn, seed)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    rng = torch.Generator(device='cuda').manual_seed(100 + seed)
    records = []
    saved = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for step in range(24):
        started = time.perf_counter()
        loss, norm = update(net, optimizer, train, rng)
        torch.cuda.synchronize()
        records.append({'step': step, 'loss': loss, 'grad_norm': norm,
                        'ms': 1000 * (time.perf_counter() - started)})
        if resume and step == 11:
            saved = map_tree(lambda x: x.detach().cpu().clone() if isinstance(x, torch.Tensor) else x,
                                  {'model': net.state_dict(), 'optimizer': optimizer.state_dict(),
                                   'rng': rng.get_state(), 'cpu_rng': torch.get_rng_state(),
                                   'cuda_rng': torch.cuda.get_rng_state(), 'step': 12})
            torch.save(saved, Path(__file__).with_name('checkpoint.pt'))
    peak = torch.cuda.max_memory_allocated()
    result = {'seed': seed, 'validation_loss': validation(net, val), 'steps': records,
              'steady_step_ms': statistics.median(r['ms'] for r in records[2:]),
              'peak_allocated_bytes': peak, 'resident_before_model_bytes': resident,
              'peak_training_extra_bytes': peak - resident}
    if resume:
        expected = {k: v.detach().clone() for k, v in net.state_dict().items()}
        restored = torch.load(Path(__file__).with_name('checkpoint.pt'), weights_only=True)
        net.load_state_dict(restored['model'])
        optimizer.load_state_dict(restored['optimizer'])
        rng.set_state(restored['rng']); torch.set_rng_state(restored['cpu_rng'])
        torch.cuda.set_rng_state(restored['cuda_rng'])
        replay_losses = []
        for step in range(restored['step'], 24):
            replay_losses.append(update(net, optimizer, train, rng)[0])
        error = max(float((v - expected[k]).abs().max()) for k, v in net.state_dict().items())
        loss_error = max(abs(a - b['loss']) for a, b in zip(replay_losses, records[12:]))
        result['checkpoint_resume'] = {'max_parameter_abs_error': error, 'max_loss_abs_error': loss_error}
        if error > 2e-5 or loss_error > 2e-5:
            raise RuntimeError(f'Checkpoint resume diverged: {result["checkpoint_resume"]}')
    return result


def main():
    tokens = data('cuda')
    train, val = tokens[:int(len(tokens) * .9)], tokens[int(len(tokens) * .9):]
    warm = model(recurrence, 0)
    inputs = batch(train, 2, 256, torch.Generator(device='cuda').manual_seed(1))
    loss_of(warm, inputs).backward()  # Two chunks, including returned state as input.
    with torch.no_grad():
        loss_of(warm, inputs)  # Capture inference as a separate verified signature.
    del warm
    print(json.dumps({'phase': 'captured', 'hash': recurrence.hash, 'cache': str(recurrence.job)}), flush=True)
    deadline = time.monotonic() + 75 * 60
    while recurrence.status()['state'] not in ('ready', 'failed'):
        if time.monotonic() > deadline:
            raise TimeoutError('Kernel generation timed out')
        time.sleep(2)
    if recurrence.status()['state'] != 'ready':
        raise RuntimeError(recurrence.status())
    recurrence._load_if_ready()
    if recurrence.kernel is None:
        raise RuntimeError('Accepted kernel did not load')
    # Compare all model parameter gradients, including learned initial memories,
    # through two connected chunks under BF16 autocast.
    grads, losses = {}, {}
    for name, fn in [('reference', recurrence.fn), ('kernel', recurrence)]:
        net = model(fn, 7)
        loss = loss_of(net, inputs); loss.backward()
        grads[name] = {n: p.grad.detach().clone() for n, p in net.named_parameters()}
        losses[name] = float(loss.detach())
        del net
    errors = {n: float((g - grads['reference'][n]).norm() / grads['reference'][n].norm().clamp_min(1e-20))
              for n, g in grads['kernel'].items()}
    if max(errors.values()) > .03:
        raise RuntimeError(f'Whole-model gradient mismatch: {errors}')
    del grads, loss
    results = {'data_sha256': hashlib.sha256(Path('/tmp/tinyshakespeare.txt').read_bytes()).hexdigest(),
               'protocol': {'steps': 24, 'seeds': [0, 1, 2], 'batch': 2, 'accumulation': 2,
                            'sequence': 256, 'chunk': 128, 'precision': 'bf16 autocast',
                            'validation_batches': 8, 'train_fraction': .9},
               'whole_model_gradient_errors': errors, 'initial_losses': losses, 'runs': []}
    for seed in (0, 1, 2):
        for name, fn in [('reference', recurrence.fn), ('kernel', recurrence)]:
            run = {'implementation': name, **training(fn, seed, train, val, resume=name == 'kernel' and seed == 0)}
            results['runs'].append(run)
            print(json.dumps({'phase': 'trained', **{k: v for k, v in run.items() if k != 'steps'}}), flush=True)
    results['routing'] = recurrence.status()
    if results['routing']['calls'].get('kernel', 0) < 100:
        raise RuntimeError('Insufficient actual kernel dispatch')
    if any(k.startswith('fallback:') for k in results['routing']['calls']):
        raise RuntimeError(f'Unexpected fallback: {results["routing"]["calls"]}')
    Path(__file__).with_name('training_results.json').write_text(json.dumps(results, indent=2))
    print(json.dumps({'phase': 'complete', 'status': recurrence.status()}), flush=True)


if __name__ == '__main__':
    main()
