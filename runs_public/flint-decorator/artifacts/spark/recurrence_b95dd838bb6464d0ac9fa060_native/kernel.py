from __future__ import annotations
import torch
import triton
import triton.language as tl

FLINT_NATIVE = True
_NS = __name__.replace(".", "_")


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


@triton.jit
def _fwd_kernel(
    q_ptr, k_ptr, v_ptr, decay_ptr, beta_ptr, transfer_ptr, out_ptr, fast_ptr, slow_ptr,
    stride_q_b, stride_q_h, stride_q_l, stride_q_d,
    stride_k_b, stride_k_h, stride_k_l, stride_k_d,
    stride_v_b, stride_v_h, stride_v_l, stride_v_d,
    stride_decay_b, stride_decay_h, stride_decay_l, stride_decay_d,
    stride_beta_b, stride_beta_h, stride_beta_l,
    stride_tr_b, stride_tr_h, stride_tr_l,
    stride_o_b, stride_o_h, stride_o_l, stride_o_d,
    stride_fast_b, stride_fast_h, stride_fast_m, stride_fast_n,
    stride_slow_b, stride_slow_h, stride_slow_m, stride_slow_n,
    B, H, L, D,
    SCALE,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    q_base = q_ptr + b * stride_q_b + h * stride_q_h
    k_base = k_ptr + b * stride_k_b + h * stride_k_h
    v_base = v_ptr + b * stride_v_b + h * stride_v_h
    decay_base = decay_ptr + b * stride_decay_b + h * stride_decay_h
    beta_base = beta_ptr + b * stride_beta_b + h * stride_beta_h
    tr_base = transfer_ptr + b * stride_tr_b + h * stride_tr_h
    out_base = out_ptr + b * stride_o_b + h * stride_o_h
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_m = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_D)
    mask_2d = (offs_m < D)[:, None] & (offs_n < D)[None, :]
    fast_base = fast_ptr + b * stride_fast_b + h * stride_fast_h
    slow_base = slow_ptr + b * stride_slow_b + h * stride_slow_h
    fast_ptrs = fast_base + offs_m[:, None] * stride_fast_m + offs_n[None, :] * stride_fast_n
    slow_ptrs = slow_base + offs_m[:, None] * stride_slow_m + offs_n[None, :] * stride_slow_n
    fast = tl.load(fast_ptrs, mask=mask_2d, other=0.0)
    slow = tl.load(slow_ptrs, mask=mask_2d, other=0.0)
    scale_half = SCALE * 0.5
    Df = D * D * 1.0
    eps = 1e-12
    for t in range(128):
        if t < L:
            qv = tl.load(q_base + t * stride_q_l + offs_d * stride_q_d, mask=mask_d, other=0.0).to(tl.float32)
            k_raw = tl.load(k_base + t * stride_k_l + offs_d * stride_k_d, mask=mask_d, other=0.0).to(tl.float32)
            vv = tl.load(v_base + t * stride_v_l + offs_d * stride_v_d, mask=mask_d, other=0.0).to(tl.float32)
            dv = tl.load(decay_base + t * stride_decay_l + offs_d * stride_decay_d, mask=mask_d, other=0.0).to(tl.float32)
            bv = tl.load(beta_base + t * stride_beta_l).to(tl.float32)
            trv = tl.load(tr_base + t * stride_tr_l).to(tl.float32)
            k_sq = tl.where(mask_d, k_raw * k_raw, 0.0)
            sum_sq = tl.sum(k_sq, axis=0)
            norm = tl.sqrt(sum_sq)
            norm_clamped = tl.where(norm < eps, eps, norm)
            inv = 1.0 / norm_clamped
            k_norm = tl.where(mask_d, k_raw * inv, 0.0)
            decay_sig = tl.sigmoid(dv)
            beta_sig = tl.sigmoid(bv)
            fast = fast * decay_sig[:, None]
            fast = tl.where(mask_2d, fast, 0.0)
            pred = tl.sum(fast * k_norm[:, None], axis=0)
            pred = tl.where(mask_d, pred, 0.0)
            delta = tl.where(mask_d, beta_sig * (vv - pred), 0.0)
            fast = fast + k_norm[:, None] * delta[None, :]
            fast = tl.where(mask_2d, fast, 0.0)
            agree = tl.sum(tl.sum(fast * slow, axis=0), axis=0) / Df
            gate = tl.sigmoid(trv + 0.05 * agree)
            slow = (1.0 - gate) * slow + gate * fast
            slow = tl.where(mask_2d, slow, 0.0)
            q_scaled = tl.where(mask_d, qv * scale_half, 0.0)
            out_vec = tl.sum((fast + slow) * q_scaled[:, None], axis=0)
            tl.store(out_base + t * stride_o_l + offs_d * stride_o_d, out_vec, mask=mask_d)
    tl.store(fast_ptrs, fast, mask=mask_2d)
    tl.store(slow_ptrs, slow, mask=mask_2d)


@triton.jit
def _bwd_kernel(
    q_ptr, k_ptr, v_ptr, decay_ptr, beta_ptr, transfer_ptr,
    fast_init_ptr, slow_init_ptr,
    grad_out_ptr, grad_fast_final_ptr, grad_slow_final_ptr,
    grad_q_ptr, grad_k_ptr, grad_v_ptr, grad_decay_ptr, grad_beta_ptr, grad_transfer_ptr,
    grad_fast_init_ptr, grad_slow_init_ptr,
    stride_q_b, stride_q_h, stride_q_l, stride_q_d,
    stride_k_b, stride_k_h, stride_k_l, stride_k_d,
    stride_v_b, stride_v_h, stride_v_l, stride_v_d,
    stride_decay_b, stride_decay_h, stride_decay_l, stride_decay_d,
    stride_beta_b, stride_beta_h, stride_beta_l,
    stride_tr_b, stride_tr_h, stride_tr_l,
    stride_fi_b, stride_fi_h, stride_fi_m, stride_fi_n,
    stride_si_b, stride_si_h, stride_si_m, stride_si_n,
    stride_go_b, stride_go_h, stride_go_l, stride_go_d,
    stride_gff_b, stride_gff_h, stride_gff_m, stride_gff_n,
    stride_gsf_b, stride_gsf_h, stride_gsf_m, stride_gsf_n,
    stride_gq_b, stride_gq_h, stride_gq_l, stride_gq_d,
    stride_gk_b, stride_gk_h, stride_gk_l, stride_gk_d,
    stride_gv_b, stride_gv_h, stride_gv_l, stride_gv_d,
    stride_gd_b, stride_gd_h, stride_gd_l, stride_gd_d,
    stride_gb_b, stride_gb_h, stride_gb_l,
    stride_gt_b, stride_gt_h, stride_gt_l,
    stride_gfi_b, stride_gfi_h, stride_gfi_m, stride_gfi_n,
    stride_gsi_b, stride_gsi_h, stride_gsi_m, stride_gsi_n,
    B, H, L, D, SCALE,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_m = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_D)
    mask_2d = (offs_m < D)[:, None] & (offs_n < D)[None, :]

    q_base = q_ptr + b * stride_q_b + h * stride_q_h
    k_base = k_ptr + b * stride_k_b + h * stride_k_h
    v_base = v_ptr + b * stride_v_b + h * stride_v_h
    decay_base = decay_ptr + b * stride_decay_b + h * stride_decay_h
    beta_base = beta_ptr + b * stride_beta_b + h * stride_beta_h
    tr_base = transfer_ptr + b * stride_tr_b + h * stride_tr_h
    go_base = grad_out_ptr + b * stride_go_b + h * stride_go_h
    gq_base = grad_q_ptr + b * stride_gq_b + h * stride_gq_h
    gk_base = grad_k_ptr + b * stride_gk_b + h * stride_gk_h
    gv_base = grad_v_ptr + b * stride_gv_b + h * stride_gv_h
    gd_base = grad_decay_ptr + b * stride_gd_b + h * stride_gd_h
    gb_base = grad_beta_ptr + b * stride_gb_b + h * stride_gb_h
    gt_base = grad_transfer_ptr + b * stride_gt_b + h * stride_gt_h

    fi_base = fast_init_ptr + b * stride_fi_b + h * stride_fi_h
    si_base = slow_init_ptr + b * stride_si_b + h * stride_si_h
    fi_ptrs = fi_base + offs_m[:, None] * stride_fi_m + offs_n[None, :] * stride_fi_n
    si_ptrs = si_base + offs_m[:, None] * stride_si_m + offs_n[None, :] * stride_si_n
    finit = tl.load(fi_ptrs, mask=mask_2d, other=0.0)
    sinit = tl.load(si_ptrs, mask=mask_2d, other=0.0)

    Df = D * D * 1.0
    scale_half = SCALE * 0.5
    eps = 1e-12

    gff_base = grad_fast_final_ptr + b * stride_gff_b + h * stride_gff_h
    gsf_base = grad_slow_final_ptr + b * stride_gsf_b + h * stride_gsf_h
    gff_ptrs = gff_base + offs_m[:, None] * stride_gff_m + offs_n[None, :] * stride_gff_n
    gsf_ptrs = gsf_base + offs_m[:, None] * stride_gsf_m + offs_n[None, :] * stride_gsf_n
    g_fast = tl.load(gff_ptrs, mask=mask_2d, other=0.0)
    g_slow = tl.load(gsf_ptrs, mask=mask_2d, other=0.0)

    for t in range(127, -1, -1):
        if t < L:
            # recompute fast_prev, slow_prev = state before step t
            fast_cur = finit
            slow_cur = sinit
            for j in range(128):
                if j < L and j < t:
                    k_raw_j = tl.load(k_base + j * stride_k_l + offs_d * stride_k_d, mask=mask_d, other=0.0).to(tl.float32)
                    dv_j = tl.load(decay_base + j * stride_decay_l + offs_d * stride_decay_d, mask=mask_d, other=0.0).to(tl.float32)
                    bv_j = tl.load(beta_base + j * stride_beta_l).to(tl.float32)
                    vv_j = tl.load(v_base + j * stride_v_l + offs_d * stride_v_d, mask=mask_d, other=0.0).to(tl.float32)
                    trv_j = tl.load(tr_base + j * stride_tr_l).to(tl.float32)
                    k_sq_j = tl.where(mask_d, k_raw_j * k_raw_j, 0.0)
                    sum_sq_j = tl.sum(k_sq_j, axis=0)
                    norm_j = tl.sqrt(sum_sq_j)
                    norm_clamped_j = tl.where(norm_j < eps, eps, norm_j)
                    k_norm_j = tl.where(mask_d, k_raw_j * (1.0 / norm_clamped_j), 0.0)
                    decay_sig_j = tl.sigmoid(dv_j)
                    beta_sig_j = tl.sigmoid(bv_j)
                    fast_temp_j = tl.where(mask_2d, fast_cur * decay_sig_j[:, None], 0.0)
                    pred_j = tl.sum(fast_temp_j * k_norm_j[:, None], axis=0)
                    pred_j = tl.where(mask_d, pred_j, 0.0)
                    delta_j = tl.where(mask_d, beta_sig_j * (vv_j - pred_j), 0.0)
                    fast_cur = tl.where(mask_2d, fast_temp_j + k_norm_j[:, None] * delta_j[None, :], 0.0)
                    agree_j = tl.sum(tl.sum(fast_cur * slow_cur, axis=0), axis=0) / Df
                    gate_j = tl.sigmoid(trv_j + 0.05 * agree_j)
                    slow_cur = tl.where(mask_2d, (1.0 - gate_j) * slow_cur + gate_j * fast_cur, 0.0)
            fast_prev = fast_cur
            slow_prev = slow_cur

            k_raw = tl.load(k_base + t * stride_k_l + offs_d * stride_k_d, mask=mask_d, other=0.0).to(tl.float32)
            vv = tl.load(v_base + t * stride_v_l + offs_d * stride_v_d, mask=mask_d, other=0.0).to(tl.float32)
            dv = tl.load(decay_base + t * stride_decay_l + offs_d * stride_decay_d, mask=mask_d, other=0.0).to(tl.float32)
            bv = tl.load(beta_base + t * stride_beta_l).to(tl.float32)
            trv = tl.load(tr_base + t * stride_tr_l).to(tl.float32)
            qv = tl.load(q_base + t * stride_q_l + offs_d * stride_q_d, mask=mask_d, other=0.0).to(tl.float32)
            go_vec = tl.load(go_base + t * stride_go_l + offs_d * stride_go_d, mask=mask_d, other=0.0).to(tl.float32)

            k_sq = tl.where(mask_d, k_raw * k_raw, 0.0)
            sum_sq = tl.sum(k_sq, axis=0)
            norm = tl.sqrt(sum_sq)
            norm_clamped = tl.where(norm < eps, eps, norm)
            inv = 1.0 / norm_clamped
            k_norm = tl.where(mask_d, k_raw * inv, 0.0)
            decay_sig = tl.sigmoid(dv)
            beta_sig = tl.sigmoid(bv)

            fast_temp = tl.where(mask_2d, fast_prev * decay_sig[:, None], 0.0)
            pred = tl.sum(fast_temp * k_norm[:, None], axis=0)
            pred = tl.where(mask_d, pred, 0.0)
            delta_vec = tl.where(mask_d, beta_sig * (vv - pred), 0.0)
            fast_t = tl.where(mask_2d, fast_temp + k_norm[:, None] * delta_vec[None, :], 0.0)
            agree = tl.sum(tl.sum(fast_t * slow_prev, axis=0), axis=0) / Df
            gate = tl.sigmoid(trv + 0.05 * agree)
            slow_t = tl.where(mask_2d, (1.0 - gate) * slow_prev + gate * fast_t, 0.0)

            q_scaled = tl.where(mask_d, qv * scale_half, 0.0)
            add_mat = tl.where(mask_2d, q_scaled[:, None] * go_vec[None, :], 0.0)
            g_fast = g_fast + add_mat
            g_slow = g_slow + add_mat
            gq = tl.sum((fast_t + slow_t) * go_vec[None, :], axis=1) * scale_half
            gq = tl.where(mask_d, gq, 0.0)
            tl.store(gq_base + t * stride_gq_l + offs_d * stride_gq_d, gq, mask=mask_d)

            diff = tl.where(mask_2d, fast_t - slow_prev, 0.0)
            d_gate_sum = tl.sum(tl.sum(diff * g_slow, axis=0), axis=0)
            d_z = d_gate_sum * gate * (1.0 - gate)
            tl.store(gt_base + t * stride_gt_l, d_z)
            d_agree = d_z * 0.05
            g_fast = g_fast + tl.where(mask_2d, slow_prev * (d_agree / Df), 0.0)
            g_slow_prev_agree = tl.where(mask_2d, fast_t * (d_agree / Df), 0.0)
            g_fast_tmp = g_fast + tl.where(mask_2d, g_slow * gate, 0.0)
            g_slow_prev = tl.where(mask_2d, g_slow * (1.0 - gate) + g_slow_prev_agree, 0.0)

            G = g_fast_tmp
            d_delta = tl.sum(G * k_norm[:, None], axis=0)
            d_delta = tl.where(mask_d, d_delta, 0.0)
            d_k_outer = tl.sum(G * delta_vec[None, :], axis=1)
            d_k_outer = tl.where(mask_d, d_k_outer, 0.0)
            d_pred = tl.where(mask_d, -d_delta * beta_sig, 0.0)
            d_k_pred = tl.sum(fast_temp * d_pred[None, :], axis=1)
            d_k_pred = tl.where(mask_d, d_k_pred, 0.0)
            d_k_norm = d_k_outer + d_k_pred
            d_fast_temp = tl.where(mask_2d, G + d_pred[None, :] * k_norm[:, None], 0.0)
            d_decay_sig = tl.sum(d_fast_temp * fast_prev, axis=1)
            d_decay_sig = tl.where(mask_d, d_decay_sig, 0.0)
            d_decay_raw = d_decay_sig * decay_sig * (1.0 - decay_sig)
            tl.store(gd_base + t * stride_gd_l + offs_d * stride_gd_d, d_decay_raw, mask=mask_d)
            g_fast_prev = tl.where(mask_2d, d_fast_temp * decay_sig[:, None], 0.0)

            gv = tl.where(mask_d, d_delta * beta_sig, 0.0)
            tl.store(gv_base + t * stride_gv_l + offs_d * stride_gv_d, gv, mask=mask_d)
            d_beta_sig = tl.sum(d_delta * (vv - pred), axis=0)
            d_beta_raw = d_beta_sig * beta_sig * (1.0 - beta_sig)
            tl.store(gb_base + t * stride_gb_l, d_beta_raw)

            dot = tl.sum(d_k_norm * k_raw, axis=0)
            is_small = norm < eps
            gk_small = tl.where(mask_d, d_k_norm / eps, 0.0)
            inv_n = 1.0 / norm_clamped
            inv_n3 = inv_n * inv_n * inv_n
            gk_large = tl.where(mask_d, d_k_norm * inv_n - k_raw * dot * inv_n3, 0.0)
            gk = tl.where(is_small, gk_small, gk_large)
            tl.store(gk_base + t * stride_gk_l + offs_d * stride_gk_d, gk, mask=mask_d)

            g_fast = g_fast_prev
            g_slow = g_slow_prev

    gfi_base = grad_fast_init_ptr + b * stride_gfi_b + h * stride_gfi_h
    gsi_base = grad_slow_init_ptr + b * stride_gsi_b + h * stride_gsi_h
    gfi_ptrs = gfi_base + offs_m[:, None] * stride_gfi_m + offs_n[None, :] * stride_gfi_n
    gsi_ptrs = gsi_base + offs_m[:, None] * stride_gsi_m + offs_n[None, :] * stride_gsi_n
    tl.store(gfi_ptrs, g_fast, mask=mask_2d)
    tl.store(gsi_ptrs, g_slow, mask=mask_2d)


@torch.library.triton_op(f"{_NS}::flint_forward", mutates_args={})
def _forward_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay: torch.Tensor, beta: torch.Tensor, transfer: torch.Tensor, fast_init: torch.Tensor, slow_init: torch.Tensor, scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, L, D = q.shape
    out = torch.empty((B, H, L, D), device=q.device, dtype=torch.float32)
    fast_buf = torch.empty_like(fast_init)
    slow_buf = torch.empty_like(slow_init)
    fast_buf.copy_(fast_init)
    slow_buf.copy_(slow_init)
    BLOCK_D = _next_pow2(D)
    grid = (B * H,)
    torch.library.wrap_triton(_fwd_kernel)[grid](
        q, k, v, decay, beta, transfer, out, fast_buf, slow_buf,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        decay.stride(0), decay.stride(1), decay.stride(2), decay.stride(3),
        beta.stride(0), beta.stride(1), beta.stride(2),
        transfer.stride(0), transfer.stride(1), transfer.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        fast_buf.stride(0), fast_buf.stride(1), fast_buf.stride(2), fast_buf.stride(3),
        slow_buf.stride(0), slow_buf.stride(1), slow_buf.stride(2), slow_buf.stride(3),
        B, H, L, D, float(scale), BLOCK_D=BLOCK_D,
    )
    return out, fast_buf, slow_buf


@torch.library.triton_op(f"{_NS}::flint_backward", mutates_args={})
def _backward_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, decay: torch.Tensor, beta: torch.Tensor, transfer: torch.Tensor, fast_init: torch.Tensor, slow_init: torch.Tensor, grad_out: torch.Tensor, grad_fast_final: torch.Tensor, grad_slow_final: torch.Tensor, scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, L, D = q.shape
    grad_q = torch.empty((B, H, L, D), device=q.device, dtype=torch.float32)
    grad_k = torch.empty((B, H, L, D), device=q.device, dtype=torch.float32)
    grad_v = torch.empty((B, H, L, D), device=q.device, dtype=torch.float32)
    grad_decay = torch.empty((B, H, L, D), device=q.device, dtype=torch.float32)
    grad_beta = torch.empty((B, H, L), device=q.device, dtype=torch.float32)
    grad_transfer = torch.empty((B, H, L), device=q.device, dtype=torch.float32)
    grad_fast_init = torch.empty_like(fast_init, dtype=torch.float32)
    grad_slow_init = torch.empty_like(slow_init, dtype=torch.float32)
    BLOCK_D = _next_pow2(D)
    grid = (B * H,)
    torch.library.wrap_triton(_bwd_kernel)[grid](
        q, k, v, decay, beta, transfer,
        fast_init, slow_init,
        grad_out, grad_fast_final, grad_slow_final,
        grad_q, grad_k, grad_v, grad_decay, grad_beta, grad_transfer,
        grad_fast_init, grad_slow_init,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        decay.stride(0), decay.stride(1), decay.stride(2), decay.stride(3),
        beta.stride(0), beta.stride(1), beta.stride(2),
        transfer.stride(0), transfer.stride(1), transfer.stride(2),
        fast_init.stride(0), fast_init.stride(1), fast_init.stride(2), fast_init.stride(3),
        slow_init.stride(0), slow_init.stride(1), slow_init.stride(2), slow_init.stride(3),
        grad_out.stride(0), grad_out.stride(1), grad_out.stride(2), grad_out.stride(3),
        grad_fast_final.stride(0), grad_fast_final.stride(1), grad_fast_final.stride(2), grad_fast_final.stride(3),
        grad_slow_final.stride(0), grad_slow_final.stride(1), grad_slow_final.stride(2), grad_slow_final.stride(3),
        grad_q.stride(0), grad_q.stride(1), grad_q.stride(2), grad_q.stride(3),
        grad_k.stride(0), grad_k.stride(1), grad_k.stride(2), grad_k.stride(3),
        grad_v.stride(0), grad_v.stride(1), grad_v.stride(2), grad_v.stride(3),
        grad_decay.stride(0), grad_decay.stride(1), grad_decay.stride(2), grad_decay.stride(3),
        grad_beta.stride(0), grad_beta.stride(1), grad_beta.stride(2),
        grad_transfer.stride(0), grad_transfer.stride(1), grad_transfer.stride(2),
        grad_fast_init.stride(0), grad_fast_init.stride(1), grad_fast_init.stride(2), grad_fast_init.stride(3),
        grad_slow_init.stride(0), grad_slow_init.stride(1), grad_slow_init.stride(2), grad_slow_init.stride(3),
        B, H, L, D, float(scale), BLOCK_D=BLOCK_D,
    )
    return grad_q, grad_k, grad_v, grad_decay, grad_beta, grad_transfer, grad_fast_init, grad_slow_init


def _setup_context(ctx, inputs, output):
    ctx.save_for_backward(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7])
    ctx.scale = inputs[8]
    ctx.dtypes = (inputs[0].dtype, inputs[1].dtype, inputs[2].dtype, inputs[3].dtype, inputs[4].dtype, inputs[5].dtype, inputs[6].dtype, inputs[7].dtype)

def _backward_impl(ctx, grad_out, grad_fast, grad_slow):
    q, k, v, decay, beta, transfer, fast_init, slow_init = ctx.saved_tensors
    scale = ctx.scale
    if grad_out is None:
        grad_out = torch.zeros((q.shape[0], q.shape[1], q.shape[2], q.shape[3]), device=q.device, dtype=torch.float32)
    else:
        grad_out = grad_out.to(torch.float32)
    if grad_fast is None:
        grad_fast = torch.zeros_like(fast_init, dtype=torch.float32)
    else:
        grad_fast = grad_fast.to(torch.float32)
    if grad_slow is None:
        grad_slow = torch.zeros_like(slow_init, dtype=torch.float32)
    else:
        grad_slow = grad_slow.to(torch.float32)
    gq, gk, gv, gd, gb, gt, gf_init, gs_init = _backward_op(q, k, v, decay, beta, transfer, fast_init, slow_init, grad_out, grad_fast, grad_slow, float(scale))
    dtypes = ctx.dtypes
    return gq.to(dtypes[0]), gk.to(dtypes[1]), gv.to(dtypes[2]), gd.to(dtypes[3]), gb.to(dtypes[4]), gt.to(dtypes[5]), gf_init.to(dtypes[6]), gs_init.to(dtypes[7]), None

_forward_op.register_autograd(_backward_impl, setup_context=_setup_context)

def kernel_function(q, k, v, decay, beta, transfer, state, scale):
    fast_init = state["fast"]
    slow_init = state["slow"]
    scale_f = float(scale) if not isinstance(scale, torch.Tensor) else float(scale.detach().to(torch.float32).item()) if scale.numel() == 1 else float(scale)
    out, fast_out, slow_out = _forward_op(q, k, v, decay, beta, transfer, fast_init, slow_init, scale_f)
    return (out, {"fast": fast_out, "slow": slow_out})

def opcheck_arguments(*original_args):
    q, k, v, decay, beta, transfer, state, scale = original_args
    fast_init = state["fast"]
    slow_init = state["slow"]
    scale_f = float(scale) if not isinstance(scale, torch.Tensor) else float(scale.detach().to(torch.float32).item()) if scale.numel() == 1 else float(scale)
    return (_forward_op, (q, k, v, decay, beta, transfer, fast_init, slow_init, scale_f))

__all__ = ["kernel_function", "opcheck_arguments", "FLINT_NATIVE"]
