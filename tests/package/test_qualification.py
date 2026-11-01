import copy
import json
from pathlib import Path
import pytest
from flint_gpu.qualification import decision, snapshot, activation, SCHEMA, sha


def pairs():
 return [{'reference':{'success':True,'seconds':10+i*.1,'peak_allocated_bytes':100},
          'benchmark':{'success':True,'seconds':7+i*.1,'peak_allocated_bytes':90,'artifacts':{'job':'digest'}}} for i in range(3)]


def test_speed_memory_and_noise_are_admission_conditions():
 p=pairs();assert decision(p)[0]
 for key,value in [('seconds',11),('seconds',9.8),('peak_allocated_bytes',101),('success',False)]:
  q=copy.deepcopy(p);q[1]['benchmark'][key]=value;assert not decision(q)[0]
 assert not decision(p[:2])[0]
 q=copy.deepcopy(p);q[1]['benchmark']['artifacts']={};assert not decision(q)[0]


def test_activation_is_bound_to_source_arguments_data_and_artifact_set(monkeypatch,tmp_path):
 monkeypatch.chdir(tmp_path);monkeypatch.setattr('sys.argv',['train.py','--steps','20'])
 (tmp_path/'train.py').write_text('x=1\n');(tmp_path/'data.bin').write_bytes(b'fixed data')
 cache=tmp_path/'.flint/cache';job=cache/'job';job.mkdir(parents=True)
 (job/'kernel.py').write_text('x=2\n');digest=sha(job/'kernel.py')
 status={'state':'ready','kernel':'kernel.py','kernel_sha256':digest}
 (job/'status.json').write_text(json.dumps(status))
 (job/'job.json').write_text(json.dumps({'environment':{'test':1}}))
 p=pairs()
 for pair in p:pair['benchmark']['artifacts']={'job':digest}
 receipt={'schema':SCHEMA,'accepted':True,'minimum_gain':.05,'pairs':p,'environment':{'test':1},
          'preparation':copy.deepcopy(p[0]),'cache_policy':'one_complete_command_per_mode_before_fixed_pairs_v1',
          'workload':snapshot(tmp_path,['train.py','--steps','20'],['data.bin'])}
 path=tmp_path/'.flint/qualification.json';path.write_text(json.dumps(receipt))
 config={'cache_dir':str(cache)}
 assert activation(job,status,config,{'test':1})[0]
 saved=receipt.pop('preparation');path.write_text(json.dumps(receipt))
 assert not activation(job,status,config,{'test':1})[0]
 receipt['preparation']=saved;path.write_text(json.dumps(receipt))
 assert not activation(job,status,config,{'test':2})[0]
 (tmp_path/'data.bin').write_bytes(b'changed');assert not activation(job,status,config,{'test':1})[0]
 (tmp_path/'data.bin').write_bytes(b'fixed data')
 (tmp_path/'new_helper.py').write_text('x=1');assert not activation(job,status,config,{'test':1})[0]
 (tmp_path/'new_helper.py').unlink()
 (job/'kernel.py').write_text('x=3');assert not activation(job,status,config,{'test':1})[0]


def test_runtime_version_and_command_are_bound(monkeypatch,tmp_path):
 monkeypatch.chdir(tmp_path);(tmp_path/'train.py').write_text('pass')
 a=snapshot(tmp_path,['train.py','--steps','10'])
 b=snapshot(tmp_path,['train.py','--steps','11'])
 assert a!=b and 'runtime_version' in a


def test_flint_settings_override_legacy_without_mutating_environment(monkeypatch,tmp_path):
 from flint_gpu.config import read_config
 import os
 monkeypatch.setenv('MUSE_MODE','reference');monkeypatch.setenv('FLINT_MODE','benchmark')
 monkeypatch.setenv('FLINT_CACHE',str(tmp_path/'cache'))
 assert read_config(tmp_path)['mode']=='benchmark'
 assert read_config(tmp_path)['cache_dir']==str(tmp_path/'cache')
 assert os.environ['MUSE_MODE']=='reference'


def test_baseline_bounded_gradients_accept_reference_but_reject_worse_candidate(monkeypatch):
 import torch
 from kernel_contract.spec import capture
 from kernel_contract.worker import run_case, VALUE_TRANSFORMS
 spec=capture('torch.sigmoid',(torch.ones(3,requires_grad=True),),name='sigmoid')
 spec.flags.update(check_grads=True,normwise_rtol=.02)
 monkeypatch.setitem(VALUE_TRANSFORMS,'saturated20',lambda x:x*0+20)
 case={'name':'saturated20','axis':'values','transform':'saturated20','seed':0}
 strict=run_case(spec,torch.sigmoid,case,'cpu')
 assert strict['outcome']=='wrong'
 spec.flags['reference_bounded_gradients']=True
 bounded=run_case(spec,torch.sigmoid,case,'cpu')
 assert bounded['outcome']=='accept'
 assert any(g['used_reference_bound'] for g in bounded['gradient_checks'])
 def worse(x):return torch.sigmoid(x).detach()+3*(x-x.detach())
 assert run_case(spec,worse,case,'cpu')['outcome']=='wrong'


def test_torch_compiler_can_import_before_flint(tmp_path):
 import subprocess,sys,os
 root=Path(__file__).resolve().parents[2]
 p=subprocess.run([sys.executable,'-c','import torch._dynamo; import flint_gpu; assert callable(flint_gpu.fast)'],
                  cwd=tmp_path,env={**os.environ,'PYTHONPATH':str(root)},capture_output=True,text=True)
 assert p.returncode==0,p.stderr


def test_cuda_initialization_default_does_not_invalidate_qualification(monkeypatch,tmp_path):
 monkeypatch.chdir(tmp_path);(tmp_path/'train.py').write_text('pass')
 monkeypatch.delenv('CUDA_MODULE_LOADING',raising=False)
 before=snapshot(tmp_path,['train.py'])
 monkeypatch.setenv('CUDA_MODULE_LOADING','LAZY')
 assert snapshot(tmp_path,['train.py'])==before
 monkeypatch.setenv('CUDA_MODULE_LOADING','EAGER')
 assert snapshot(tmp_path,['train.py'])!=before


def test_qualification_prepares_both_paths_once_before_fixed_pairs(monkeypatch,tmp_path):
 import flint_gpu.qualification as q
 monkeypatch.chdir(tmp_path);(tmp_path/'train.py').write_text('pass')
 calls=[];elapsed=[]
 class Process:
  def __init__(self,command,env,**kwargs):
   mode=env['FLINT_MODE'];label=env['FLINT_RUN_ID'];calls.append((label,mode))
   prep=label.startswith('prepare')
   data={'success':True,'peak_allocated_bytes':900 if prep else (100 if mode=='reference' else 90),
         'artifacts':{'job':'digest'} if mode=='benchmark' else {},'dispatch':[]}
   Path(env['FLINT_LAUNCH_REPORT']).write_text(json.dumps(data))
   elapsed.append(30 if prep else (10 if mode=='reference' else 7))
  def wait(self,timeout):return 0
 ticks=[0]
 def clock():
  if elapsed:ticks[0]+=elapsed.pop(0)
  return ticks[0]
 monkeypatch.setattr(q.subprocess,'Popen',Process)
 monkeypatch.setattr(q.time,'monotonic',clock)
 monkeypatch.setattr('flint_gpu.identity.environment',lambda:{'gpu':'test'})
 result=q.qualify(['train.py'])
 assert result['accepted'] and len(result['pairs'])==3
 assert calls==[('prepare-reference','reference'),('prepare-benchmark','benchmark'),
                ('0-reference','reference'),('0-benchmark','benchmark'),
                ('1-benchmark','benchmark'),('1-reference','reference'),
                ('2-reference','reference'),('2-benchmark','benchmark')]
 assert result['preparation_seconds']==60 and result['qualification_seconds']==111
