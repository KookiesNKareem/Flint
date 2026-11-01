"""Acceptance checks for generated native operators, using the real compiler and autograd."""
import torch
from kernel_contract.tree import leaves, map_tree
from kernel_contract.worker import run_case
from .memory import loss_of_outputs
import math


def compare_gradients(actual, native, oracle=None):
    """Retain the original elementwise test; only the approved oracle bound can rescue it."""
    evidence=[]
    for index,(a,b) in enumerate(zip(actual,native)):
        if b is None:
            if a is not None and a.count_nonzero():raise ValueError('Unexpected partial-mask gradient')
            continue
        try:
            torch.testing.assert_close(a,b,rtol=2e-2,atol=1e-5)
            evidence.append({'input':index,'used_reference_bound':False})
        except AssertionError:
            if oracle is None or a is None or oracle[index] is None:raise
            truth=oracle[index]
            if a.shape!=truth.shape or b.shape!=truth.shape or not torch.isfinite(a).all():raise
            error=float((a.to(truth.device).double()-truth.double()).abs().max()) if truth.numel() else 0.
            baseline=float((b.to(truth.device).double()-truth.double()).abs().max()) if truth.numel() else 0.
            if not math.isfinite(error) or not math.isfinite(baseline) or error>baseline:raise
            evidence.append({'input':index,'used_reference_bound':True,'error':error,
                             'native_reference_error':baseline,'limit':baseline})
    return evidence


def check(spec, kernel):
    from .remote_eval import observed_inputs
    device=spec.flags.get('device','cuda')
    namespace = kernel.__globals__
    if not (namespace.get('FLINT_NATIVE') or namespace.get('MUSE_NATIVE')) or 'opcheck_arguments' not in namespace:
        raise ValueError('Native artifact must export FLINT_NATIVE and opcheck_arguments')
    args, kwargs = observed_inputs(spec, True)
    op, flat = namespace['opcheck_arguments'](*args, **kwargs)
    if not isinstance(op, torch.library.CustomOpDef):
        raise TypeError('opcheck_arguments must expose the actual registered forward operator')
    try:op_result=torch.library.opcheck(op,flat,raise_exception=True)
    except Exception as error:
        # PyTorch 2.8 explicitly lacks this one registration test for MPS. Do not
        # suppress numerical/registration failures, or claim the test passed.
        if device!='mps' or 'autograd_registration_check: NYI devices other than CPU/CUDA' not in str(error):raise
        op_result=torch.library.opcheck(op,flat,test_utils=('test_schema','test_faketensor','test_aot_dispatch_dynamic'),raise_exception=True)
        op_result['test_autograd_registration']='unsupported by installed PyTorch for MPS; compiled gradient cases required'

    explanation = torch._dynamo.explain(kernel)(*args, **kwargs)
    if explanation.graph_count != 1 or explanation.graph_break_count != 0:
        raise ValueError(f'Native wrapper graph breaks: {explanation.break_reasons}')
    names = [str(n.target) for graph in explanation.graphs for n in graph.graph.nodes if n.op=='call_function']
    if not any(op._name.replace('::','.') in name for name in names):
        raise ValueError('The public kernel_function did not dispatch the op exposed for checking')
    compiled = torch.compile(kernel, fullgraph=True, dynamic=False)
    result = run_case(spec, compiled, {'name':'compiled/observed','axis':'compiler','seed':173}, device)
    if result['outcome'] != 'accept':
        raise ValueError('Compiled output/gradient check: '+str(result))
    # Compare the captured partial grad mask; all-input and inference paths are tested above.
    from kernel_contract.spec import materialize
    def captured_mask(function,double=False):
        args, kwargs = materialize(spec, spec.sizes, 174, 'cpu' if double and device=='mps' else device)
        def mask(value, desc):
            if not isinstance(desc,dict):return value
            if 'tensor' in desc:
                return (value.double() if double and value.is_floating_point() else value).detach().requires_grad_(desc['tensor'].get('requires_grad', False))
            if 'seq' in desc:
                values=[mask(v,d) for v,d in zip(value,desc['seq'])]
                return tuple(values) if desc.get('tuple') else values
            if 'map' in desc: return {k:mask(value[k],d) for k,d in desc['map'].items()}
            return value
        args=[mask(v,d) for v,d in zip(args,spec.args)]
        inputs=[v for _,v in leaves(args) if isinstance(v,torch.Tensor) and v.requires_grad]
        out=function(*args,**kwargs)
        gradients=torch.autograd.grad(loss_of_outputs(out),inputs,allow_unused=True) if inputs else []
        return out, gradients
    reference, rg = captured_mask(spec.reference())
    actual, ag = captured_mask(compiled)
    for (_, a), (_, b) in zip(leaves(actual),leaves(reference)):
        torch.testing.assert_close(a,b,rtol=2e-2,atol=1e-5)
    oracle=None
    if spec.flags.get('reference_bounded_gradients'):
        from kernel_contract.worker import KeepFloat64
        with torch.autocast(device,enabled=False),KeepFloat64():
            _,oracle=captured_mask(spec.reference(),double=True)
    partial_checks=compare_gradients(ag,rg,oracle)
    amp_checks=[]
    for dtype in (torch.bfloat16, torch.float16):
        with torch.autocast(device, dtype=dtype):
            reference, rg = captured_mask(spec.reference())
            actual, ag = captured_mask(compiled)
        for (_,a),(_,b) in zip(leaves(actual),leaves(reference)):
            torch.testing.assert_close(a,b,rtol=2e-2,atol=1e-5)
        details=compare_gradients(ag,rg,oracle)
        amp_checks.append({'dtype':str(dtype),'gradient_checks':details})
    return {'passed':True,'opcheck':op_result,'graph_count':explanation.graph_count,
            'graph_break_count':explanation.graph_break_count,'operators':names,
            'compiled_case':result,'captured_grad_mask':True,'partial_gradient_checks':partial_checks,
            'fullgraph':True,'backend':device,'compiler':'inductor','autocast':amp_checks}
