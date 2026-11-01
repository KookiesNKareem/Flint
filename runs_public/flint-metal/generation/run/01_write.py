from __future__ import annotations
import torch

FLINT_BACKEND = 'mps'

_MSL = r"""
#include <metal_stdlib>
using namespace metal;

kernel void recurrence_fwd_f32(
    device float* x      [[buffer(0)]],
    device float* h_in   [[buffer(1)]],
    device float* c_in   [[buffer(2)]],
    device float* out    [[buffer(3)]],
    device float* h_out  [[buffer(4)]],
    device float* c_out  [[buffer(5)]],
    constant long& B     [[buffer(6)]],
    constant long& T     [[buffer(7)]],
    constant long& H     [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    long total = B * H;
    if ((long)gid >= total) return;
    long b = (long)gid / H;
    long hk = (long)gid % H;
    long h_base = b * H + hk;
    float h = h_in[h_base];
    float c = c_in[h_base];
    for (long t = 0; t < T; ++t) {
        long x_idx = b * T * H + t * H + hk;
        float x_val = x[x_idx];
        float h_new = tanh(x_val + 0.8f * h + 0.2f * c);
        float c_new = 0.9f * c + 0.1f * h_new;
        h = h_new;
        c = c_new;
        long o_idx = b * T * H + t * H + hk;
        out[o_idx] = h;
    }
    h_out[h_base] = h;
    c_out[h_base] = c;
}

kernel void recurrence_fwd_f16(
    device half* x       [[buffer(0)]],
    device half* h_in    [[buffer(1)]],
    device half* c_in    [[buffer(2)]],
    device half* out     [[buffer(3)]],
    device half* h_out   [[buffer(4)]],
    device half* c_out   [[buffer(5)]],
    constant long& B     [[buffer(6)]],
    constant long& T     [[buffer(7)]],
    constant long& H     [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    long total = B * H;
    if ((long)gid >= total) return;
    long b = (long)gid / H;
    long hk = (long)gid % H;
    long h_base = b * H + hk;
    float h = float(h_in[h_base]);
    float c = float(c_in[h_base]);
    for (long t = 0; t < T; ++t) {
        long idx = b * T * H + t * H + hk;
        float x_val = float(x[idx]);
        float h_new = tanh(x_val + 0.8f * h + 0.2f * c);
        float c_new = 0.9f * c + 0.1f * h_new;
        h = h_new;
        c = c_new;
        out[idx] = half(h);
    }
    h_out[h_base] = half(h);
    c_out[h_base] = half(c);
}
"""

_lib = torch.mps.compile_shader(_MSL)

def kernel_function(x, h, c):
    # shapes: x [B,T,H], h [B,H], c [B,H]
    # respect strides/offsets via contiguous() - allowed and preserves semantics
    x_c = x.contiguous()
    h_c = h.contiguous()
    c_c = c.contiguous()

    B = x_c.shape[0]
    T = x_c.shape[1]
    H = x_c.shape[2]

    out = torch.empty_like(x_c)
    h_out = torch.empty_like(h_c)
    c_out = torch.empty_like(c_c)

    total = B * H
    if total == 0:
        return (out, h_out, c_out)

    dt = x_c.dtype
    if dt == torch.float32:
        _lib.recurrence_fwd_f32([x_c, h_c, c_c, out, h_out, c_out], [B, T, H], threads=total)
        return (out, h_out, c_out)
    elif dt == torch.float16:
        _lib.recurrence_fwd_f16([x_c, h_c, c_c, out, h_out, c_out], [B, T, H], threads=total)
        return (out, h_out, c_out)
    else:
        # generic fallback: use float32 Metal kernel with float accumulation, preserve dtype
        # cheap elementwise prep allowed
        x_f = x_c.float()
        h_f = h_c.float()
        c_f = c_c.float()
        out_f = torch.empty_like(x_f)
        h_out_f = torch.empty_like(h_f)
        c_out_f = torch.empty_like(c_f)
        _lib.recurrence_fwd_f32([x_f, h_f, c_f, out_f, h_out_f, c_out_f], [B, T, H], threads=total)
        return (out_f.to(dt), h_out_f.to(dt), c_out_f.to(dt))
