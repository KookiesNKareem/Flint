import json
import torch
import pytest
from kernel_contract.spec import capture, materialize
from kernel_contract.worker import run_case
from kernel_contract.tree import structure
from muse_kernel.identity import signature, CACHE_VERSION
from muse_kernel.memory import measure


def reference(x, state):
    fast = state['fast'] + x
    slow = state['slow'] * .8 + fast * .2
    return fast + slow, {'fast': fast, 'slow': slow}


def spec():
    s = capture(__name__ + '.reference', (torch.randn(2, 4), {'fast': torch.randn(2, 4), 'slow': torch.randn(2, 4)}))
    s.reference = lambda: reference
    s.flags['check_grads'] = True
    return s


def test_nested_state_capture_materialization_signature():
    s = spec()
    args, _ = materialize(s, s.sizes, 0, 'cpu')
    assert set(args[1]) == {'fast', 'slow'}
    assert args[1]['slow'].shape == (2, 4)
    before = signature(args)
    args[1]['slow'] = args[1]['slow'].T.contiguous().T
    assert before != signature(args)
    assert structure(reference(*args)) == {'tuple': ['tensor', {'dict': {'fast': 'tensor', 'slow': 'tensor'}}]}


def test_all_state_outputs_and_cotangents_are_checked():
    s = spec();case = {'name': 'observed', 'axis': 'seed', 'seed': 0}
    result = run_case(s, reference, case, 'cpu')
    assert result['outcome'] == 'accept'
    assert result['outputs_checked'] == 3 and result['gradient_objectives'] == 4
    assert result['grad_inputs_checked'] == 3
    def wrong_state(x, state):
        y, states = reference(x, state)
        return y, {**states, 'slow': states['slow'] + 1}
    assert run_case(s, wrong_state, case, 'cpu')['outcome'] == 'wrong'
    def detached_state(x, state):
        y, states = reference(x, state)
        return y, {k: v.detach() for k, v in states.items()}
    assert run_case(s, detached_state, case, 'cpu')['outcome'] == 'wrong'
    def missing_state(x, state):
        return reference(x, state)[0]
    assert run_case(s, missing_state, case, 'cpu')['outcome'] == 'wrong'
    def mutating_state(x, state):
        result = reference(x, state)
        state['fast'].zero_()
        return result
    assert run_case(s, mutating_state, case, 'cpu')['outcome'] == 'wrong'
    def wrong_dtype(x, state):
        y, states = reference(x, state)
        return y, {k: v.half() for k, v in states.items()}
    s.flags['any_float_output'] = True
    assert run_case(s, wrong_dtype, case, 'cpu')['outcome'] == 'wrong'


def test_two_memory_chunking_preserves_states_and_gradients():
    import importlib.util
    from pathlib import Path
    source = Path(__file__).resolve().parents[2] / 'examples/muse_kernel/two_memory.py'
    module_spec = importlib.util.spec_from_file_location('two_memory_test', source)
    module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
    fn = module.recurrence.fn
    torch.manual_seed(11)
    args = [torch.randn(1, 2, 8, 4, requires_grad=True) for _ in range(4)]
    args[1] = torch.nn.functional.normalize(args[1], dim=-1); args[3] = -args[3].abs()
    args += [torch.rand(1, 2, 8, requires_grad=True), torch.randn(1, 2, 8, requires_grad=True)]
    state = {k: torch.randn(1, 2, 4, 4, requires_grad=True) for k in ('fast', 'slow')}
    y, final = fn(*args, state, .5)
    first, mid = fn(*(v[:, :, :3] for v in args), state, .5)
    last, end = fn(*(v[:, :, 3:] for v in args), mid, .5)
    assert torch.equal(y, torch.cat([first, last], 2))
    assert all(torch.equal(final[k], end[k]) for k in final)
    g1 = torch.autograd.grad(y.sum() + sum(v.sum() for v in final.values()), tuple(state.values()), retain_graph=True)
    g2 = torch.autograd.grad(first.sum() + last.sum() + sum(v.sum() for v in end.values()), tuple(state.values()))
    assert all(torch.allclose(a, b) for a, b in zip(g1, g2))


def test_decorator_captures_nested_states_and_memory_policy(monkeypatch, tmp_path):
    import muse_kernel
    monkeypatch.setenv('MUSE_KERNEL_CACHE', str(tmp_path))
    monkeypatch.setenv('MUSE_KERNEL_EXTERNAL_BUILDER', '1')
    fast = muse_kernel.fast(reference, memory_budget_mb=4, models=['meta-models/muse-spark-open-1026'], backward_model='muse-wonder-1.0')
    for _ in range(2):
        fast(torch.randn(2, 4, requires_grad=True), {'fast': torch.randn(2, 4, requires_grad=True), 'slow': torch.randn(2, 4, requires_grad=True)})
    job = json.loads((fast.job/'job.json').read_text())
    assert job['cache_version'] == CACHE_VERSION
    assert job['memory_policy']['max_peak_extra_bytes'] == 4*1024**2
    assert job['backward_model'] == 'muse-wonder-1.0'
    assert job['output_structure']['tuple'][1]['dict']['slow'] == 'tensor'
    assert fast.job != muse_kernel.fast(reference, memory_budget_mb=8).job
    with pytest.raises(ValueError): muse_kernel.fast(reference, memory_budget_mb=-1)


def test_memory_counts_activation_storage_and_clears_gradients():
    x = torch.randn(4, 8, requires_grad=True)
    result = measure(lambda a: a.sin().square(), (x,), {})
    assert result['saved_activation_bytes'] >= x.numel()*x.element_size()
    assert x.grad is None
