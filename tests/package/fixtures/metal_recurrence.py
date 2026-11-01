"""Hand-authored backend integration fixture, not model-generated benchmark evidence."""
import torch
FLINT_NATIVE=True
FLINT_BACKEND='mps'
lib=torch.mps.compile_shader(r'''
#include <metal_stdlib>
using namespace metal;
kernel void forward_scan(device const float* x, device const float* hi, device const float* ci,
                         device float* y, device float* hf, device float* cf,
                         constant long& T, constant long& D, uint i [[thread_position_in_grid]]) {
    uint b=i/D,d=i%D;
    float h=hi[i],c=ci[i];
    for(long t=0;t<T;t++) {
        h=tanh(x[(b*T+t)*D+d]+0.8f*h+0.2f*c);
        c=0.9f*c+0.1f*h;
        y[(b*T+t)*D+d]=h;
    }
    hf[i]=h;cf[i]=c;
}
kernel void backward_scan(device const float* y, device const float* dy,
                          device const float* dhf, device const float* dcf,
                          device float* dx, device float* dhi, device float* dci,
                          constant long& T, constant long& D, uint i [[thread_position_in_grid]]) {
    uint b=i/D,d=i%D;
    float dh=dhf[i],dc=dcf[i];
    for(long t=T-1;t>=0;t--) {
        long j=(b*T+t)*D+d;
        float h=y[j];
        float dp=(dh+dy[j]+0.1f*dc)*(1.0f-h*h);
        dx[j]=dp;dh=0.8f*dp;dc=0.9f*dc+0.2f*dp;
    }
    dhi[i]=dh;dci[i]=dc;
}
''')
namespace=__name__.replace('.','_')

@torch.library.custom_op(namespace+'::scan',mutates_args=(),device_types='mps')
def forward(x:torch.Tensor,h:torch.Tensor,c:torch.Tensor)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    if x.dtype!=torch.float32 or h.dtype!=torch.float32 or c.dtype!=torch.float32:
        raise ValueError('This fixture supports float32 only')
    b,t,d=x.shape
    y=torch.empty((b,t,d),device=x.device,dtype=x.dtype)
    hf=torch.empty((b,d),device=x.device,dtype=x.dtype);cf=torch.empty_like(hf)
    lib.forward_scan(x.contiguous(),h.contiguous(),c.contiguous(),y,hf,cf,t,d,threads=b*d)
    return y,hf,cf

@forward.register_fake
def _(x,h,c):
    b,t,d=x.shape
    return (torch.empty((b,t,d),device=x.device,dtype=x.dtype),
            torch.empty((b,d),device=x.device,dtype=x.dtype),torch.empty((b,d),device=x.device,dtype=x.dtype))

@torch.library.custom_op(namespace+'::scan_backward',mutates_args=(),device_types='mps')
def backward_op(y:torch.Tensor,dy:torch.Tensor,dh:torch.Tensor,dc:torch.Tensor)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    b,t,d=y.shape
    dx=torch.empty((b,t,d),device=y.device,dtype=y.dtype)
    dhi=torch.empty((b,d),device=y.device,dtype=y.dtype);dci=torch.empty_like(dhi)
    lib.backward_scan(y.contiguous(),dy.contiguous(),dh.contiguous(),dc.contiguous(),dx,dhi,dci,t,d,threads=b*d)
    return dx,dhi,dci

@backward_op.register_fake
def _(y,dy,dh,dc):
    b,t,d=y.shape
    return (torch.empty((b,t,d),device=y.device,dtype=y.dtype),
            torch.empty((b,d),device=y.device,dtype=y.dtype),torch.empty((b,d),device=y.device,dtype=y.dtype))

def setup_context(ctx,inputs,output):ctx.save_for_backward(output[0])

def backward(ctx,dy,dh,dc):
    y,=ctx.saved_tensors;b,t,d=y.shape
    if dy is None:dy=torch.zeros_like(y)
    if dh is None:dh=torch.zeros((b,d),device=y.device,dtype=y.dtype)
    if dc is None:dc=torch.zeros((b,d),device=y.device,dtype=y.dtype)
    return backward_op(y,dy,dh,dc)

forward.register_autograd(backward,setup_context=setup_context)

def kernel_function(x,h,c):return forward(x,h,c)
def opcheck_arguments(x,h,c):return forward,(x,h,c)
