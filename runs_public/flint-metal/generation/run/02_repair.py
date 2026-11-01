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
    constant long& x_s0  [[buffer(9)]],
    constant long& x_s1  [[buffer(10)]],
    constant long& x_s2  [[buffer(11)]],
    constant long& h_s0  [[buffer(12)]],
    constant long& h_s1  [[buffer(13)]],
    constant long& c_s0  [[buffer(14)]],
    constant long& c_s1  [[buffer(15)]],
    uint gid [[thread_position_in_grid]]
) {
    long total = B * H;
    if ((long)gid >= total) return;
    long b = (long)gid / H;
    long hk = (long)gid % H;
    // strided read of initial states (pointer already at storage offset)
    float h = h_in[b * h_s0 + hk * h_s1];
    float c = c_in[b * c_s0 + hk * c_s1];
    for (long t = 0; t < T; ++t) {
        long x_idx = b * x_s0 + t * x_s1 + hk * x_s2;
        float x_val = x[x_idx];
        float h_new = tanh(x_val + 0.8f * h + 0.2f * c);
        float c_new = 0.9f * c + 0.1f * h_new;
        h = h_new;
        c = c_new;
        long o_idx = b * (T * H) + t * H + hk;
        out[o_idx] = h;
    }
    long ho = b * H + hk;
    h_out[ho] = h;
    c_out[ho] = c;
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
    constant long& x_s0  [[buffer(9)]],
    constant long& x_s1  [[buffer(10)]],
    constant long& x_s2  [[buffer(11)]],
    constant long& h_s0  [[buffer(12)]],
    constant long& h_s1  [[buffer(13)]],
    constant long& c_s0  [[buffer(14)]],
    constant long& c_s1  [[buffer(15)]],
    uint gid [[thread_position_in_grid]]
) {
    long total = B * H;
    if ((long)gid >= total) return;
    long b = (long)gid / H;
    long hk = (long)gid % H;
    float h = float(h_in[b * h_s0 + hk * h_s1]);
    float c = float(c_in[b * c_s0 + hk * c_s1]);
    for (long t = 0; t < T; ++t) {
        long x_idx = b * x_s0 + t * x_s1 + hk * x_s2;
        float x_val = float(x[x_idx]);
        float h_new = tanh(x_val + 0.8f * h + 0.2f * c);
        float c_new = 0.9f * c + 0.1f * h_new;
        h = h_new;
        c = c_new;
        long o_idx = b * (T * H) + t * H + hk;
        out[o_idx] = half(h);
    }
    long ho = b * H + hk;
    h_out[ho] = half(h);
    c_out[ho] = half(c);
}
"""

_lib = torch.mps.compile_shader(_MSL)

def kernel_function(x, h, c):
    # preserve dtypes, do not mutate inputs
    B = x.shape[0]
    T = x.shape[1]
    H = x.shape[2]

    # output allocation - contiguous logical layout B,T,H and B,H
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    h_out = torch.empty_like(h)
    c_out = torch.empty_like(c)

    total = B * H
    if total == 0 or T == 0:
        # handle empty edge case without launch
        if total != 0 and T != 0:
            pass
        else:
            # if T==0 outputs empty, h_out/c_out are copies of inputs
            if total != 0:
                h_out.copy_(h)
                c_out.copy_(c)
            return (out, h_out, c_out)

    # strides are in elements; pointer already includes storage_offset
    x_s0, x_s1, x_s2 = x.stride()
    h_s0, h_s1 = h.stride()
    c_s0, c_s1 = c.stride()

    dt = x.dtype
    if dt == torch.float32:
        _lib.recurrence_fwd_f32(
            [x, h, c, out, h_out, c_out],
            [B, T, H, x_s0, x_s1, x_s2, h_s0, h_s1, c_s0, c_s1],
            threads=total,
        )
        return (out, h_out, c_out)
    elif dt == torch.float16:
        _lib.recurrence_fwd_f16(
            [x, h, c, out, h_out, c_out],
            [B, T, H, x_s0, x_s1, x_s2, h_s0, h_s1, c_s0, c_s1],
            threads=total,
        )
        return (out, h_out, c_out)
    else:
        # bfloat16 or other: float32 accumulation then cast back
        # cheap elementwise prep allowed
        x_f = x.float()
        h_f = h.float()
        c_f = c.float()
        out_f = torch.empty(x_f.shape, dtype=torch.float32, device=x.device)
        h_out_f = torch.empty_like(h_f)
        c_out_f = torch.empty_like(c_f)
        xf_s0, xf_s1, xf_s2 = x_f.stride()
        hf_s0, hf_s1 = h_f.stride()
        cf_s0, cf_s1 = c_f.stride()
        _lib.recurrence_fwd_f32(
            [x_f, h_f, c_f, out_f, h_out_f, c_out_f],
            [B, T, H, xf_s0, xf_s1, xf_s2, hf_s0, hf_s1, cf_s0, cf_s1],
            threads=total,
        )
        return (out_f.to(dt), h_out_f.to(dt), c_out_f.to(dt))
