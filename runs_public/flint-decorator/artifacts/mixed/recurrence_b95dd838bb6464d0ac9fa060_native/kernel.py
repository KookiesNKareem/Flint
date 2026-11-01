from __future__ import annotations
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

FLINT_NATIVE = True
namespace = __name__.replace('.', '_')

def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()

# Forward/backward operate directly on the *raw* inputs.  All cheap
# elementwise prep of the reference (promote to fp32, normalize k,
# logsigmoid(decay)->exp == sigmoid(decay), sigmoid(beta)) is fused
# into the kernels, so autograd saves only references to the original
# input storages (no retained fp32/normalized/sigmoid copies) and the
# backward recomputes states from the initial state instead of
# retaining any per-timestep activations.

@triton.jit
def _fwd_kernel(
    q_ptr, k_ptr, v_ptr, a_ptr, beta_ptr, transfer_ptr,
    fast_ptr, slow_ptr, out_ptr, fast_out_ptr, slow_out_ptr,
    T, D, SCALE,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask_d = offs < D
    mask_2d = (offs[:, None] < D) & (offs[None, :] < D)
    rows = offs[:, None]
    cols = offs[None, :]
    seq_base = pid * T * D
    mat_base = pid * D * D
    fast_mat = tl.load(fast_ptr + mat_base + rows * D + cols, mask=mask_2d, other=0.0).to(tl.float32)
    slow_mat = tl.load(slow_ptr + mat_base + rows * D + cols, mask=mask_2d, other=0.0).to(tl.float32)
    for t in range(T):
        qv = tl.load(q_ptr + seq_base + t * D + offs, mask=mask_d, other=0.0).to(tl.float32)
        kraw = tl.load(k_ptr + seq_base + t * D + offs, mask=mask_d, other=0.0).to(tl.float32)
        vv = tl.load(v_ptr + seq_base + t * D + offs, mask=mask_d, other=0.0).to(tl.float32)
        araw = tl.load(a_ptr + seq_base + t * D + offs, mask=mask_d, other=0.0).to(tl.float32)
        bv = tl.load(beta_ptr + pid * T + t).to(tl.float32)
        tv = tl.load(transfer_ptr + pid * T + t).to(tl.float32)
        knorm = tl.sqrt(tl.sum(kraw * kraw))
        kden = tl.maximum(knorm, 1e-12)
        kv = kraw / kden
        mult = 1.0 / (1.0 + tl.exp(-araw))
        bval = 1.0 / (1.0 + tl.exp(-bv))
        fast_mat = fast_mat * mult[:, None]
        pred = tl.sum(fast_mat * kv[:, None], axis=0)
        delta = bval * (vv - pred)
        fast_mat = fast_mat + kv[:, None] * delta[None, :]
        agreement = tl.sum(fast_mat * slow_mat) / (D * D)
        gate = 1.0 / (1.0 + tl.exp(-(tv + 0.05 * agreement)))
        slow_mat = (1.0 - gate) * slow_mat + gate * fast_mat
        out_vec = tl.sum((fast_mat + slow_mat) * (qv * (0.5 * SCALE))[:, None], axis=0)
        tl.store(out_ptr + seq_base + t * D + offs, out_vec, mask=mask_d)
    tl.store(fast_out_ptr + mat_base + rows * D + cols, fast_mat, mask=mask_2d)
    tl.store(slow_out_ptr + mat_base + rows * D + cols, slow_mat, mask=mask_2d)

@triton.jit
def _bwd_kernel(
    q_ptr, k_ptr, v_ptr, a_ptr, beta_ptr, transfer_ptr,
    fast0_ptr, slow0_ptr,
    grad_out_ptr, grad_fast_ptr, grad_slow_ptr,
    gq_ptr, gk_ptr, gv_ptr, ga_ptr, gbeta_ptr, gtransfer_ptr,
    gfast_ptr, gslow_ptr,
    T, D, SCALE,
    BLOCK_D: tl.constexpr,
    HAS_GO: tl.constexpr, HAS_GF: tl.constexpr, HAS_GS: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask_d = offs < D
    mask_2d = (offs[:, None] < D) & (offs[None, :] < D)
    rows = offs[:, None]
    cols = offs[None, :]
    seq_base = pid * T * D
    mat_base = pid * D * D
    c = 0.5 * SCALE
    if HAS_GF:
        dF = tl.load(grad_fast_ptr + mat_base + rows * D + cols, mask=mask_2d, other=0.0).to(tl.float32)
    else:
        dF = tl.zeros((BLOCK_D, BLOCK_D), tl.float32)
    if HAS_GS:
        dS = tl.load(grad_slow_ptr + mat_base + rows * D + cols, mask=mask_2d, other=0.0).to(tl.float32)
    else:
        dS = tl.zeros((BLOCK_D, BLOCK_D), tl.float32)

    for idx in range(T):
        t = T - 1 - idx
        qv = tl.load(q_ptr + seq_base + t * D + offs, mask=mask_d, other=0.0).to(tl.float32)
        kraw = tl.load(k_ptr + seq_base + t * D + offs, mask=mask_d, other=0.0).to(tl.float32)
        vv = tl.load(v_ptr + seq_base + t * D + offs, mask=mask_d, other=0.0).to(tl.float32)
        araw = tl.load(a_ptr + seq_base + t * D + offs, mask=mask_d, other=0.0).to(tl.float32)
        bv = tl.load(beta_ptr + pid * T + t).to(tl.float32)
        tv = tl.load(transfer_ptr + pid * T + t).to(tl.float32)
        if HAS_GO:
            go = tl.load(grad_out_ptr + seq_base + t * D + offs, mask=mask_d, other=0.0).to(tl.float32)
        else:
            go = tl.zeros((BLOCK_D,), tl.float32)

        knorm = tl.sqrt(tl.sum(kraw * kraw))
        kden = tl.maximum(knorm, 1e-12)
        kv = kraw / kden
        mult = 1.0 / (1.0 + tl.exp(-araw))
        bval = 1.0 / (1.0 + tl.exp(-bv))

        # Single recomputation pass yielding both (F_{t-1}, S_{t-1}) and (F_t, S_t).
        f_cur = tl.load(fast0_ptr + mat_base + rows * D + cols, mask=mask_2d, other=0.0).to(tl.float32)
        s_cur = tl.load(slow0_ptr + mat_base + rows * D + cols, mask=mask_2d, other=0.0).to(tl.float32)
        f_prev = f_cur
        s_prev = s_cur
        for tt in range(T):
            f_prev = tl.where(tt == t, f_cur, f_prev)
            s_prev = tl.where(tt == t, s_cur, s_prev)
            kraw_tt = tl.load(k_ptr + seq_base + tt * D + offs, mask=mask_d, other=0.0).to(tl.float32)
            vv_tt = tl.load(v_ptr + seq_base + tt * D + offs, mask=mask_d, other=0.0).to(tl.float32)
            araw_tt = tl.load(a_ptr + seq_base + tt * D + offs, mask=mask_d, other=0.0).to(tl.float32)
            bv_tt = tl.load(beta_ptr + pid * T + tt).to(tl.float32)
            tv_tt = tl.load(transfer_ptr + pid * T + tt).to(tl.float32)
            knorm_tt = tl.sqrt(tl.sum(kraw_tt * kraw_tt))
            kv_tt = kraw_tt / tl.maximum(knorm_tt, 1e-12)
            mult_tt = 1.0 / (1.0 + tl.exp(-araw_tt))
            bval_tt = 1.0 / (1.0 + tl.exp(-bv_tt))
            f_scaled = f_cur * mult_tt[:, None]
            pred_tt = tl.sum(f_scaled * kv_tt[:, None], axis=0)
            delta_tt = bval_tt * (vv_tt - pred_tt)
            f_next = f_scaled + kv_tt[:, None] * delta_tt[None, :]
            agr_tt = tl.sum(f_next * s_cur) / (D * D)
            gate_tt = 1.0 / (1.0 + tl.exp(-(tv_tt + 0.05 * agr_tt)))
            s_next = (1.0 - gate_tt) * s_cur + gate_tt * f_next
            f_cur = tl.where(tt <= t, f_next, f_cur)
            s_cur = tl.where(tt <= t, s_next, s_cur)
        f_mat = f_cur
        s_mat = s_cur

        gq = c * tl.sum((f_mat + s_mat) * go[None, :], axis=1)
        dF = dF + (c * qv)[:, None] * go[None, :]
        dS = dS + (c * qv)[:, None] * go[None, :]

        agreement = tl.sum(f_mat * s_prev) / (D * D)
        gate = 1.0 / (1.0 + tl.exp(-(tv + 0.05 * agreement)))
        dg = tl.sum(dS * (f_mat - s_prev))
        gu = dg * gate * (1.0 - gate)
        coef = (gu * 0.05) / (D * D)
        dF_total = dF + gate * dS + coef * s_prev
        dS_prev = (1.0 - gate) * dS + coef * f_mat

        f_prime = f_prev * mult[:, None]
        pred = tl.sum(f_prime * kv[:, None], axis=0)
        delta = bval * (vv - pred)

        ddelta = tl.sum(dF_total * kv[:, None], axis=0)
        dk_outer = tl.sum(dF_total * delta[None, :], axis=1)
        gv = bval * ddelta
        dp = -bval * ddelta
        gbeta_t = tl.sum(ddelta * (vv - pred)) * bval * (1.0 - bval)
        dF_prime = dF_total + kv[:, None] * dp[None, :]
        gk_t = dk_outer + tl.sum(f_prime * dp[None, :], axis=1)
        # chain through k-normalization
        gk_dot = tl.sum(gk_t * kv)
        gk_raw = tl.where(knorm > 1e-12, (gk_t - kv * gk_dot) / kden, gk_t / kden)
        ga_t = tl.sum(dF_prime * f_prev, axis=1) * mult * (1.0 - mult)

        tl.store(gq_ptr + seq_base + t * D + offs, gq, mask=mask_d)
        tl.store(gk_ptr + seq_base + t * D + offs, gk_raw, mask=mask_d)
        tl.store(gv_ptr + seq_base + t * D + offs, gv, mask=mask_d)
        tl.store(ga_ptr + seq_base + t * D + offs, ga_t, mask=mask_d)
        tl.store(gbeta_ptr + pid * T + t, gbeta_t)
        tl.store(gtransfer_ptr + pid * T + t, gu)

        dF = dF_prime * mult[:, None]
        dS = dS_prev

    tl.store(gfast_ptr + mat_base + rows * D + cols, dF, mask=mask_2d)
    tl.store(gslow_ptr + mat_base + rows * D + cols, dS, mask=mask_2d)


@torch.library.triton_op(f"{namespace}::recurrence_fwd", mutates_args={})
def recurrence_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay: torch.Tensor, beta: torch.Tensor, transfer: torch.Tensor, fast: torch.Tensor, slow: torch.Tensor, scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    a_c = decay.contiguous()
    beta_c = beta.contiguous()
    transfer_c = transfer.contiguous()
    fast_c = fast.contiguous()
    slow_c = slow.contiguous()
    B, H, T, D = q_c.shape
    out_dtype = torch.promote_types(q_c.dtype, torch.float32)
    state_dtype = torch.promote_types(fast_c.dtype, torch.float32)
    out = torch.empty(q_c.shape, dtype=out_dtype, device=q_c.device)
    fast_out = torch.empty(fast_c.shape, dtype=state_dtype, device=fast_c.device)
    slow_out = torch.empty(slow_c.shape, dtype=state_dtype, device=slow_c.device)
    torch.library.wrap_triton(_fwd_kernel)[(B * H,)](
        q_c, k_c, v_c, a_c, beta_c, transfer_c,
        fast_c, slow_c, out, fast_out, slow_out,
        T, D, float(scale),
        BLOCK_D=_next_pow2(int(D)),
    )
    return (out, fast_out, slow_out)


@torch.library.triton_op(f"{namespace}::recurrence_bwd", mutates_args={})
def recurrence_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay: torch.Tensor, beta: torch.Tensor, transfer: torch.Tensor, fast: torch.Tensor, slow: torch.Tensor, grad_out: torch.Tensor, grad_fast: torch.Tensor, grad_slow: torch.Tensor, scale: float, has_go: int, has_gf: int, has_gs: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    a_c = decay.contiguous()
    beta_c = beta.contiguous()
    transfer_c = transfer.contiguous()
    fast_c = fast.contiguous()
    slow_c = slow.contiguous()
    go_c = grad_out.contiguous()
    gf_c = grad_fast.contiguous()
    gs_c = grad_slow.contiguous()
    B, H, T, D = q_c.shape
    gq = torch.empty_like(q_c)
    gk = torch.empty_like(k_c)
    gv = torch.empty_like(v_c)
    ga = torch.empty_like(a_c)
    gbeta = torch.empty_like(beta_c)
    gtransfer = torch.empty_like(transfer_c)
    gfast = torch.empty_like(fast_c)
    gslow = torch.empty_like(slow_c)
    torch.library.wrap_triton(_bwd_kernel)[(B * H,)](
        q_c, k_c, v_c, a_c, beta_c, transfer_c,
        fast_c, slow_c,
        go_c, gf_c, gs_c,
        gq, gk, gv, ga, gbeta, gtransfer, gfast, gslow,
        T, D, float(scale),
        BLOCK_D=_next_pow2(int(D)),
        HAS_GO=bool(has_go), HAS_GF=bool(has_gf), HAS_GS=bool(has_gs),
    )
    return (gq, gk, gv, ga, gbeta, gtransfer, gfast, gslow)


def _setup_context(ctx, inputs, output):
    q, k, v, decay, beta, transfer, fast, slow, scale = inputs
    # References to operator inputs only: no new storages are retained.
    ctx.save_for_backward(q, k, v, decay, beta, transfer, fast, slow)
    ctx.scale = scale
    ctx.set_materialize_grads(False)

def _backward(ctx, grad_out, grad_fast, grad_slow):
    q, k, v, decay, beta, transfer, fast, slow = ctx.saved_tensors
    has_go = 0 if grad_out is None else 1
    has_gf = 0 if grad_fast is None else 1
    has_gs = 0 if grad_slow is None else 1
    # Placeholders are never read in the kernel when the flag is 0.
    go = q if grad_out is None else grad_out
    gf = fast if grad_fast is None else grad_fast
    gs = slow if grad_slow is None else grad_slow
    gq, gk, gv, gdecay, gbeta, gtransfer, gfast, gslow = recurrence_bwd(
        q, k, v, decay, beta, transfer, fast, slow, go, gf, gs, ctx.scale, has_go, has_gf, has_gs,
    )
    return (gq, gk, gv, gdecay, gbeta, gtransfer, gfast, gslow, None)

recurrence_fwd.register_autograd(_backward, setup_context=_setup_context)

def kernel_function(q, k, v, decay, beta, transfer, state, scale):
    out, fast_final, slow_final = recurrence_fwd(
        q, k, v, decay, beta, transfer, state['fast'], state['slow'], float(scale),
    )
    return (out, {'fast': fast_final, 'slow': slow_final})

def opcheck_arguments(*original_args):
    q, k, v, decay, beta, transfer, state, scale = original_args
    flat = tuple(
        x.detach().to(torch.float32)
        for x in (q, k, v, decay, beta, transfer, state['fast'], state['slow'])
    )
    return (recurrence_fwd, flat + (float(scale),))
