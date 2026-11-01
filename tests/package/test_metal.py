"""Real Metal integration; cleanly skipped on hosts without MPS."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import torch
import pytest
from flint_gpu.device import Memory
from kernel_contract.spec import capture
from kernel_contract.worker import run_case

FIXTURE=Path(__file__).parent/'fixtures/metal_recurrence.py'


def reference(x,h,c):
    outputs=[]
    for t in range(x.shape[1]):
        h=torch.tanh(x[:,t]+.8*h+.2*c);c=.9*c+.1*h;outputs.append(h)
    return torch.stack(outputs,dim=1),h,c


@pytest.fixture(scope='module')
def metal():
    if not torch.backends.mps.is_available() or not hasattr(torch.mps,'compile_shader'):pytest.skip('Metal shader backend unavailable')
    modspec=importlib.util.spec_from_file_location('flint_test_metal_fixture',FIXTURE)
    module=importlib.util.module_from_spec(modspec);modspec.loader.exec_module(module)
    return module


@pytest.fixture
def spec(monkeypatch):
    args=(torch.empty(2,9,7),torch.empty(2,7),torch.empty(2,7))
    result=capture('reference',args)
    result.flags.update(device='mps',native=True,check_grads=True,reference_bounded_gradients=True)
    monkeypatch.setattr(result,'reference',lambda:reference)
    return result


@pytest.mark.parametrize('layout',[None,'strided3_off1','transposed'])
def test_metal_forward_backward_multiple_outputs_and_cpu_double_oracle(metal,spec,layout):
    result=run_case(spec,metal.kernel_function,{'seed':123,'layout':layout},'mps')
    assert result['outcome']=='accept',result
    assert result['truth_dtype']=='torch.float64'
    assert result['gradient_objectives']==4 and result['grad_inputs_checked']==3


def test_metal_wrong_gradient_is_rejected(metal,spec):
    def wrong(x,h,c):
        outputs=metal.kernel_function(x,h,c)
        return tuple(v.detach()+.5*(v-v.detach()) for v in outputs)
    result=run_case(spec,wrong,{'seed':9},'mps')
    assert result['outcome']=='wrong' and 'gradient' in result['reason']


def test_metal_native_compile_and_partial_grad_masks(metal,spec):
    from flint_gpu.native_check import check
    spec.args[0]['tensor']['requires_grad']=True
    result=check(spec,metal.kernel_function)
    assert result['passed'] and result['fullgraph'] and result['graph_break_count']==0
    assert result['backend']=='mps'


def test_mps_memory_is_explicitly_sampled(metal):
    tracker=Memory('mps').start()
    x=torch.ones(10000,device='mps');torch.mps.synchronize()
    tracker.sample();result=tracker.finish()
    assert result['peak_allocated_bytes']>=x.numel()*4
    assert result['memory_measurement']['exact'] is False
    assert not tracker.thread.is_alive()


def test_metal_prompts_use_msl_and_custom_ops():
    from kernel_contract.paper_agent import job_prompt,backward_prompt
    job={'source':'def f(x): return x.sin()','domains':[],'call':'x: float32 MPS tensor',
         'signature':'x','backend':'mps','native':True,'grad_names':'x'}
    assert 'torch.mps.compile_shader' in job_prompt(job)
    text=backward_prompt(job)
    assert 'torch.library.custom_op' in text and 'register_fake' in text
    assert 'torch.library.triton_op' not in text


def test_mps_memory_requires_review_even_if_faster_and_smaller():
    from flint_gpu.qualification import decision
    pairs=[{'reference':{'success':True,'seconds':10,'peak_allocated_bytes':1000},
            'benchmark':{'success':True,'seconds':2,'peak_allocated_bytes':900,'artifacts':{'a':'sha'}}} for _ in range(3)]
    for p in pairs:
        for result in p.values():result['memory_measurement']={'kind':'mps_sampled_allocated','exact':False}
    assert decision(pairs)[0] is False
    assert 'review required' in decision(pairs)[1]
    assert decision(pairs,memory_ceiling=900)[0] is True
    pairs[0]['benchmark']['seconds']=20
    assert decision(pairs,memory_ceiling=900)[0] is False


def test_portable_deadline_times_out_child():
    p=subprocess.run([sys.executable,'-m','flint_gpu.deadline','.1',sys.executable,'-c','import time;time.sleep(30)'],capture_output=True,timeout=5)
    assert p.returncode!=0 and b'TimeoutExpired' in p.stderr


def test_real_mps_launcher_writes_memory_and_training_report(metal,tmp_path):
    import os
    script=tmp_path/'train.py'
    script.write_text("import torch\nx=torch.randn(100,device='mps',requires_grad=True)\nx.square().sum().backward()\n")
    report=tmp_path/'launch.json'
    env={**os.environ,'FLINT_LAUNCH_REPORT':str(report),'FLINT_LIVE':'0','FLINT_REHEARSAL_STEPS':'0',
         'PYTHONPATH':str(Path(__file__).resolve().parents[2])}
    subprocess.run([sys.executable,'-m','flint_gpu.launch',str(script)],cwd=tmp_path,env=env,check=True,capture_output=True,timeout=30)
    result=json.loads(report.read_text())
    assert result['success'] and result['memory_measurement']['kind']=='mps_sampled_allocated'


def test_metal_argument_diagnostic_is_specific_and_does_not_rewrite_source():
    from flint_gpu.metal import argument_hint
    grouped='lib = torch.mps.compile_shader(source)\nlib.scan([x,y,out],[N],threads=N)\n'
    assert 'separate' in argument_hint(grouped)
    assert not argument_hint('lib = torch.mps.compile_shader(source)\nlib.scan(x,y,out,N,threads=N)\n')
    assert not argument_hint('other.scan([x,y,out],[N],threads=N)\n')
    assert not argument_hint('this is not valid Python !!!')
