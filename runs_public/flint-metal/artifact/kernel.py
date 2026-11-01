from __future__ import annotations
import torch

FLINT_BACKEND = 'mps'
FLINT_NATIVE = True

namespace = __name__.replace('.', '_')

_MSL = r"""
#include <metal_stdlib>
using namespace metal;

kernel void recurrence_fwd(
    device float* x      [[buffer(0)]],
    device float* h_in   [[buffer(1)]],
    device float* c_in   [[buffer(2)]],
    device float* out    [[buffer(3)]],
    device float* h_out  [[buffer(4)]],
    device float* c_out  [[buffer(5)]],
    constant long& B     [[buffer(6)]],
    constant long& T     [[buffer(7)]],
    constant long& D     [[buffer(8)]],
    constant long& x_s0  [[buffer(9)]],
    constant long& x_s1  [[buffer(10)]],
    constant long& x_s2  [[buffer(11)]],
    constant long& h_s0  [[buffer(12)]],
    constant long& h_s1  [[buffer(13)]],
    constant long& c_s0  [[buffer(14)]],
    constant long& c_s1  [[buffer(15)]],
    uint gid [[thread_position_in_grid]]
) {
    long total = B * D;
    if ((long)gid >= total) return;
    long b = (long)gid / D;
    long d = (long)gid % D;
    float h = h_in[b * h_s0 + d * h_s1];
    float c = c_in[b * c_s0 + d * c_s1];
    for (long t = 0; t < T; ++t) {
        float xv = x[b * x_s0 + t * x_s1 + d * x_s2];
        float hn = tanh(xv + 0.8f * h + 0.2f * c);
        float cn = 0.9f * c + 0.1f * hn;
        h = hn;
        c = cn;
        out[b * T * D + t * D + d] = h;
    }
    long ho = b * D + d;
    h_out[ho] = h;
    c_out[ho] = c;
}

kernel void recurrence_bwd(
    device float* go     [[buffer(0)]],
    device float* ghf    [[buffer(1)]],
    device float* gcf    [[buffer(2)]],
    device float* saved  [[buffer(3)]],
    device float* gx     [[buffer(4)]],
    device float* gh0    [[buffer(5)]],
    device float* gc0    [[buffer(6)]],
    constant long& B     [[buffer(7)]],
    constant long& T     [[buffer(8)]],
    constant long& D     [[buffer(9)]],
    constant long& go_s0 [[buffer(10)]],
    constant long& go_s1 [[buffer(11)]],
    constant long& go_s2 [[buffer(12)]],
    constant long& gh_s0 [[buffer(13)]],
    constant long& gh_s1 [[buffer(14)]],
    constant long& gc_s0 [[buffer(15)]],
    constant long& gc_s1 [[buffer(16)]],
    constant long& sv_s0 [[buffer(17)]],
    constant long& sv_s1 [[buffer(18)]],
    constant long& sv_s2 [[buffer(19)]],
    uint gid [[thread_position_in_grid]]
) {
    long total = B * D;
    if ((long)gid >= total) return;
    long b = (long)gid / D;
    long d = (long)gid % D;
    float carry_dh = ghf[b * gh_s0 + d * gh_s1];
    float carry_dc = gcf[b * gc_s0 + d * gc_s1];
    for (long t = T - 1; t >= 0; --t) {
        float ht = saved[b * sv_s0 + t * sv_s1 + d * sv_s2];
        float gy = go[b * go_s0 + t * go_s1 + d * go_s2];
        float dc_total = carry_dc;
        float dh_eff = gy + carry_dh + 0.1f * dc_total;
        float da = dh_eff * (1.0f - ht * ht);
        gx[b * T * D + t * D + d] = da;
        carry_dh = 0.8f * da;
        carry_dc = 0.2f * da + 0.9f * dc_total;
    }
    long ho = b * D + d;
    gh0[ho] = carry_dh;
    gc0[ho] = carry_dc;
}
"""

_lib = torch.mps.compile_shader(_MSL)


@torch.library.custom_op(f"{namespace}::recurrence_fwd", mutates_args=(), device_types="mps")
def forward_op(x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B = int(x.shape[0])
    T = int(x.shape[1])
    D = int(x.shape[2])
    # Work in float32 (float accumulation); cast back to preserve dtypes.
    x_f = x.to(torch.float32).contiguous() if x.dtype != torch.float32 else x.contiguous()
    h_f = h.to(torch.float32).contiguous() if h.dtype != torch.float32 else h.contiguous()
    c_f = c.to(torch.float32).contiguous() if c.dtype != torch.float32 else c.contiguous()
    out_f = torch.empty((B, T, D), dtype=torch.float32, device=x.device)
    h_out_f = torch.empty((B, D), dtype=torch.float32, device=x.device)
    c_out_f = torch.empty((B, D), dtype=torch.float32, device=x.device)
    total = B * D
    if total > 0:
        x_s0, x_s1, x_s2 = map(int, x_f.stride())
        h_s0, h_s1 = map(int, h_f.stride())
        c_s0, c_s1 = map(int, c_f.stride())
        _lib.recurrence_fwd(x_f, h_f, c_f, out_f, h_out_f, c_out_f, B, T, D, x_s0, x_s1, x_s2, h_s0, h_s1, c_s0, c_s1, threads=total)
    else:
        if T == 0:
            h_out_f.copy_(h_f)
            c_out_f.copy_(c_f)
    out = out_f if x.dtype == torch.float32 else out_f.to(x.dtype)
    h_out = h_out_f if h.dtype == torch.float32 else h_out_f.to(h.dtype)
    c_out = c_out_f if c.dtype == torch.float32 else c_out_f.to(c.dtype)
    return (out, h_out, c_out)


@forward_op.register_fake
def _forward_fake(x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out = torch.empty((int(x.shape[0]), int(x.shape[1]), int(x.shape[2])), dtype=x.dtype, device=x.device)
    h_out = torch.empty((int(h.shape[0]), int(h.shape[1])), dtype=h.dtype, device=h.device)
    c_out = torch.empty((int(c.shape[0]), int(c.shape[1])), dtype=c.dtype, device=c.device)
    return (out, h_out, c_out)


@torch.library.custom_op(f"{namespace}::recurrence_bwd", mutates_args=(), device_types="mps")
def backward_op(grad_out: torch.Tensor, grad_h: torch.Tensor, grad_c: torch.Tensor, saved: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B = int(saved.shape[0])
    T = int(saved.shape[1])
    D = int(saved.shape[2])
    go_f = grad_out.to(torch.float32).contiguous() if grad_out.dtype != torch.float32 else grad_out.contiguous()
    gh_f = grad_h.to(torch.float32).contiguous() if grad_h.dtype != torch.float32 else grad_h.contiguous()
    gc_f = grad_c.to(torch.float32).contiguous() if grad_c.dtype != torch.float32 else grad_c.contiguous()
    sv_f = saved.to(torch.float32).contiguous() if saved.dtype != torch.float32 else saved.contiguous()
    gx_f = torch.empty((B, T, D), dtype=torch.float32, device=saved.device)
    gh0_f = torch.empty((B, D), dtype=torch.float32, device=saved.device)
    gc0_f = torch.empty((B, D), dtype=torch.float32, device=saved.device)
    total = B * D
    if total > 0:
        go_s0, go_s1, go_s2 = map(int, go_f.stride())
        gh_s0, gh_s1 = map(int, gh_f.stride())
        gc_s0, gc_s1 = map(int, gc_f.stride())
        sv_s0, sv_s1, sv_s2 = map(int, sv_f.stride())
        _lib.recurrence_bwd(go_f, gh_f, gc_f, sv_f, gx_f, gh0_f, gc0_f, B, T, D, go_s0, go_s1, go_s2, gh_s0, gh_s1, gc_s0, gc_s1, sv_s0, sv_s1, sv_s2, threads=total)
    gx = gx_f if grad_out.dtype == torch.float32 else gx_f.to(grad_out.dtype)
    gh0 = gh0_f if grad_h.dtype == torch.float32 else gh0_f.to(grad_h.dtype)
    gc0 = gc0_f if grad_c.dtype == torch.float32 else gc0_f.to(grad_c.dtype)
    return (gx, gh0, gc0)


@backward_op.register_fake
def _backward_fake(grad_out: torch.Tensor, grad_h: torch.Tensor, grad_c: torch.Tensor, saved: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gx = torch.empty((int(saved.shape[0]), int(saved.shape[1]), int(saved.shape[2])), dtype=grad_out.dtype, device=grad_out.device)
    gh0 = torch.empty((int(grad_h.shape[0]), int(grad_h.shape[1])), dtype=grad_h.dtype, device=grad_h.device)
    gc0 = torch.empty((int(grad_c.shape[0]), int(grad_c.shape[1])), dtype=grad_c.dtype, device=grad_c.device)
    return (gx, gh0, gc0)


def _setup_context(ctx, inputs, output):
    x, h, c = inputs
    out, h_out, c_out = output
    # `out` already holds every h_t, which is all backward needs (1 - h_t^2).
    # Save h/c as well for correct shapes/dtypes of absent final-state grads.
    ctx.save_for_backward(out, h, c)
    ctx.set_materialize_grads(False)


def _backward(ctx, grad_out, grad_h, grad_c):
    saved, h_in, c_in = ctx.saved_tensors
    if grad_out is None:
        grad_out = torch.zeros_like(saved)
    if grad_h is None:
        grad_h = torch.zeros_like(h_in)
    if grad_c is None:
        grad_c = torch.zeros_like(c_in)
    grad_x, grad_h0, grad_c0 = backward_op(grad_out, grad_h, grad_c, saved)
    return grad_x, grad_h0, grad_c0


forward_op.register_autograd(_backward, setup_context=_setup_context)


def kernel_function(x, h, c):
    out, h_out, c_out = forward_op(x, h, c)
    return (out, h_out, c_out)


def opcheck_arguments(*original_args):
    flat = []

    def _flatten(obj):
        if isinstance(obj, torch.Tensor):
            flat.append(obj)
        elif isinstance(obj, (tuple, list)):
            for e in obj:
                _flatten(e)
        elif isinstance(obj, dict):
            for k in sorted(obj.keys(), key=str):
                _flatten(obj[k])
        else:
            flat.append(obj)

    for a in original_args:
        _flatten(a)
    return (forward_op, tuple(flat))
