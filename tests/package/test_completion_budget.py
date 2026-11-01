from types import SimpleNamespace
import pytest
from kernel_contract.paper_agent import Run
from utils.providers.base import LLMResponse


def test_truncated_no_code_increases_allowance_but_dollar_gate_still_blocks(tmp_path):
    run=Run.__new__(Run)
    run.a=SimpleNamespace(model='meta-models/muse-spark-open-1026',max_tokens=24000,max_usd=.21,effort='high')
    calls=[]
    def respond(*args,**kwargs):
        calls.append(kwargs['max_tokens'])
        return LLMResponse('',run.a.model,'baseten',{'prompt_tokens':1,'completion_tokens':24000})
    run.provider=SimpleNamespace(get_response=respond)
    run.out=tmp_path;run.messages=[];run.cost=0.;run.n=0;events=[]
    run.log=lambda **e:events.append(e)
    path=run.ask('write a kernel','bwd')
    assert run.a.max_tokens==48000 and 'model returned no code' in path.read_text()
    assert any(e['event']=='completion_budget_increased' for e in events)
    # The larger next response would cross the original dollar cap: no second call.
    with pytest.raises(SystemExit,match='budget'):run.ask('try again','repair_bwd')
    assert calls==[24000]


def test_import_failure_feedback_does_not_blame_tensor_inputs():
    from kernel_contract.paper_agent import repair_api_hint
    hint=repair_api_hint('@torch.library.wrap_triton(kernel)',[
        {'reason':'kernel failed to import: Triton kernels should always be called with a grid'}])
    assert 'before tensor inputs' in hint and '[grid]' in hint
    assert repair_api_hint('wrap_triton(kernel)[grid](x)',[{'reason':'gradient mismatch'}])==''


def test_repair_diagnostics_deduplicate_without_losing_failed_cases():
    from kernel_contract.paper_agent import report
    message=report([
        {'name':'observed','outcome':'refuse','reason':'module import failed'},
        {'name':'seed1','outcome':'refuse','reason':'module import failed'},
        {'name':'seed2','outcome':'wrong','reason':'gradient mismatch'},
        {'name':'seed3','outcome':'accept'},
    ])
    assert message.count('module import failed')==1
    assert all(name in message for name in ['observed','seed1','seed2','seed3'])
    assert 'gradient mismatch' in message and '2 cases' in message
