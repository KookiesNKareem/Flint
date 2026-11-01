from __future__ import annotations
import torch

FLINT_BACKEND = 'mps'

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
    float h = h_in[b * h_s0 + hk * h_s1];
    float c = c_in[b * c_s0 + hk * c_s1];
    for (long t = 0; t < T; ++t) {
        float xv = x[b * x_s0 + t * x_s1 + hk * x_s2];
        float hn = tanh(xv + 0.8f * h + 0.2f * c);
        float cn = 0.9f * c + 0.1f * hn;
        h = hn;
        c = cn;
        out[b * T * H + t * H + hk] = h;
    }
    long ho = b * H + hk;
    h_out[ho] = h;
    c_out[ho] = c;
}
"""

_lib = torch.mps.compile_shader(_MSL)

def kernel_function(x, h, c):
    B = int(x.shape[0])
    T = int(x.shape[1])
    H = int(x.shape[2])

    out = torch.empty((B, T, H), dtype=x.dtype, device=x.device)
    h_out = torch.empty((B, H), dtype=h.dtype, device=h.device)
    c_out = torch.empty((B, H), dtype=c.dtype, device=c.device)

    if B == 0 or H == 0:
        return (out, h_out, c_out)
    if T == 0:
        h_out.copy_(h)
        c_out.copy_(c)
        return (out, h_out, c_out)

    total = B * H

    if x.dtype == torch.float32 and h.dtype == torch.float32 and c.dtype == torch.float32:
        x_s0, x_s1, x_s2 = map(int, x.stride())
        h_s0, h_s1 = map(int, h.stride())
        c_s0, c_s1 = map(int, c.stride())
        _lib.recurrence_fwd(x, h, c, out, h_out, c_out, B, T, H, x_s0, x_s1, x_s2, h_s0, h_s1, c_s0, c_s1, threads=total)
        return (out, h_out, c_out)
    else:
        x_f = x.float()
        h_f = h.float()
        c_f = c.float()
        out_f = torch.empty((B, T, H), dtype=torch.float32, device=x.device)
        h_out_f = torch.empty((B, H), dtype=torch.float32, device=x.device)
        c_out_f = torch.empty((B, H), dtype=torch.float32, device=x.device)
        x_s0, x_s1, x_s2 = map(int, x_f.stride())
        h_s0, h_s1 = map(int, h_f.stride())
        c_s0, c_s1 = map(int, c_f.stride())
        _lib.recurrence_fwd(x_f, h_f, c_f, out_f, h_out_f, c_out_f, B, T, H, x_s0, x_s1, x_s2, h_s0, h_s1, c_s0, c_s1, threads=total)
        return (out_f.to(x.dtype), h_out_f.to(h.dtype), c_out_f.to(c.dtype))
