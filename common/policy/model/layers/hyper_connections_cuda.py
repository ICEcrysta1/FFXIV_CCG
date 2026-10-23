"""训练专用系数核；由 CUDA 路径延迟导入，CPU 与导出不依赖 Triton。"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import register_autograd, triton_op, wrap_triton


@triton.jit
def _coefficients_kernel(
    mixes, scale, base, pre, post, comb, history, count,
    MULT: tl.constexpr, ITERATIONS: tl.constexpr, EPS: tl.constexpr,
    BLOCK: tl.constexpr, TOKENS: tl.constexpr, SAVE_HISTORY: tl.constexpr,
):
    token = tl.program_id(0) * TOKENS + tl.arange(0, TOKENS)
    token_valid = token < count
    axis = tl.arange(0, BLOCK)
    valid = axis < MULT
    gate_valid = token_valid[:, None] & valid[None, :]
    offset = token * ((MULT + 2) * MULT)
    if not SAVE_HISTORY:
        pre_logits = tl.load(mixes + offset[:, None] + axis[None, :], gate_valid, 0) * tl.load(scale)
        pre_logits += tl.load(base + axis, valid, 0)[None, :]
        post_logits = tl.load(mixes + offset[:, None] + MULT + axis[None, :], gate_valid, 0) * tl.load(scale + 1)
        post_logits += tl.load(base + MULT + axis, valid, 0)[None, :]
        tl.store(pre + token[:, None] * MULT + axis[None, :], tl.sigmoid(pre_logits) + EPS, gate_valid)
        tl.store(post + token[:, None] * MULT + axis[None, :], 2.0 * tl.sigmoid(post_logits), gate_valid)
    indices = axis[:, None] * MULT + axis[None, :]
    spatial_valid = valid[:, None] & valid[None, :]
    matrix_valid = token_valid[:, None, None] & spatial_valid[None, :, :]
    logits = tl.load(mixes + offset[:, None, None] + 2 * MULT + indices[None, :, :], matrix_valid, 0) * tl.load(scale + 2)
    logits += tl.load(base + 2 * MULT + indices, spatial_valid, 0)[None, :, :]
    logits = tl.where(matrix_valid, logits, float("-inf"))
    values = tl.exp(logits - tl.max(logits, 2)[:, :, None])
    values = tl.div_rn(values, tl.sum(values, 2)[:, :, None]) + EPS
    values = tl.where(matrix_valid, values, 0.0)
    stages: tl.constexpr = 2 * ITERATIONS - 1
    for iteration in tl.range(ITERATIONS, loop_unroll_factor=1):
        if iteration > 0:
            if SAVE_HISTORY:
                address = (token[:, None, None] * stages + 2 * iteration - 1) * MULT * MULT + indices[None, :, :]
                tl.store(history + address, values, matrix_valid)
            values = tl.div_rn(values, tl.sum(values, 2)[:, :, None] + EPS)
        if SAVE_HISTORY:
            address = (token[:, None, None] * stages + 2 * iteration) * MULT * MULT + indices[None, :, :]
            tl.store(history + address, values, matrix_valid)
        values = tl.div_rn(values, tl.sum(values, 1)[:, None, :] + EPS)
    if not SAVE_HISTORY:
        tl.store(comb + token[:, None, None] * MULT * MULT + indices[None, :, :], values, matrix_valid)


@triton.jit
def _coefficients_backward_kernel(
    mixes, scale, base, grad_pre, grad_post, grad_comb, history,
    grad_mixes, grad_scale, grad_base, count,
    MULT: tl.constexpr, ITERATIONS: tl.constexpr, EPS: tl.constexpr,
    BLOCK: tl.constexpr, TOKENS: tl.constexpr,
):
    token = tl.program_id(0) * TOKENS + tl.arange(0, TOKENS)
    token_valid = token < count
    axis = tl.arange(0, BLOCK)
    valid = axis < MULT
    gate_valid = token_valid[:, None] & valid[None, :]
    offset = token * ((MULT + 2) * MULT)
    pre_values = tl.load(mixes + offset[:, None] + axis[None, :], gate_valid, 0)
    post_values = tl.load(mixes + offset[:, None] + MULT + axis[None, :], gate_valid, 0)
    pre_gate = tl.sigmoid(pre_values * tl.load(scale) + tl.load(base + axis, valid, 0)[None, :])
    post_gate = tl.sigmoid(post_values * tl.load(scale + 1) + tl.load(base + MULT + axis, valid, 0)[None, :])
    gate_offset = token[:, None] * MULT + axis[None, :]
    dpre = tl.load(grad_pre + gate_offset, gate_valid, 0) * pre_gate * (1.0 - pre_gate)
    dpost = tl.load(grad_post + gate_offset, gate_valid, 0) * 2.0 * post_gate * (1.0 - post_gate)
    tl.store(grad_mixes + offset[:, None] + axis[None, :], dpre * tl.load(scale), gate_valid)
    tl.store(grad_mixes + offset[:, None] + MULT + axis[None, :], dpost * tl.load(scale + 1), gate_valid)
    tl.store(grad_base + offset[:, None] + axis[None, :], dpre, gate_valid)
    tl.store(grad_base + offset[:, None] + MULT + axis[None, :], dpost, gate_valid)
    tl.store(grad_scale + token * 3, tl.sum(dpre * pre_values, 1), token_valid)
    tl.store(grad_scale + token * 3 + 1, tl.sum(dpost * post_values, 1), token_valid)
    indices = axis[:, None] * MULT + axis[None, :]
    spatial_valid = valid[:, None] & valid[None, :]
    matrix_valid = token_valid[:, None, None] & spatial_valid[None, :, :]
    gradient = tl.load(grad_comb + token[:, None, None] * MULT * MULT + indices[None, :, :], matrix_valid, 0)
    stages: tl.constexpr = 2 * ITERATIONS - 1
    # 仅反向临时重算每步输入；前向上下文不保留逐轮矩阵。
    for reverse in tl.range(stages, loop_unroll_factor=1):
        stage = stages - 1 - reverse
        address = (token[:, None, None] * stages + stage) * MULT * MULT + indices[None, :, :]
        values = tl.load(history + address, matrix_valid, 0)
        if stage % 2 == 0:
            column_total = tl.sum(values, 1)[:, None, :] + EPS
            column_normalized = tl.div_rn(values, column_total)
            gradient = tl.div_rn(gradient - tl.sum(gradient * column_normalized, 1)[:, None, :], column_total)
        else:
            row_total = tl.sum(values, 2)[:, :, None] + EPS
            row_normalized = tl.div_rn(values, row_total)
            gradient = tl.div_rn(gradient - tl.sum(gradient * row_normalized, 2)[:, :, None], row_total)
        gradient = tl.where(matrix_valid, gradient, 0.0)
    raw = tl.load(mixes + offset[:, None, None] + 2 * MULT + indices[None, :, :], matrix_valid, 0)
    logits = raw * tl.load(scale + 2) + tl.load(base + 2 * MULT + indices, spatial_valid, 0)[None, :, :]
    logits = tl.where(matrix_valid, logits, float("-inf"))
    probability = tl.exp(logits - tl.max(logits, 2)[:, :, None])
    probability = tl.div_rn(probability, tl.sum(probability, 2)[:, :, None])
    probability = tl.where(matrix_valid, probability, 0.0)
    dlogits = probability * (gradient - tl.sum(gradient * probability, 2)[:, :, None])
    tl.store(grad_mixes + offset[:, None, None] + 2 * MULT + indices[None, :, :], dlogits * tl.load(scale + 2), matrix_valid)
    tl.store(grad_base + offset[:, None, None] + 2 * MULT + indices[None, :, :], dlogits, matrix_valid)
    tl.store(grad_scale + token * 3 + 2, tl.sum(tl.sum(dlogits * raw, 2), 1), token_valid)


@triton_op("ffxiv_ccg::hc_coefficients", mutates_args={})
def hc_coefficients(
    mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor,
    mult: int, iterations: int, eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = mixes.numel() // ((mult + 2) * mult)
    shape = mixes.shape[:-1]
    pre = mixes.new_empty((*shape, mult))
    post = torch.empty_like(pre)
    comb = mixes.new_empty((*shape, mult, mult))
    wrap_triton(_coefficients_kernel)[(triton.cdiv(count, 4),)](
        mixes, scale, base, pre, post, comb, comb, count,
        mult, iterations, eps, triton.next_power_of_2(mult), 4, False,
        num_warps=2, enable_fp_fusion=False,
    )
    return pre, post, comb


@triton_op("ffxiv_ccg::hc_coefficients_backward", mutates_args={})
def _hc_coefficients_backward(
    mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor,
    grad_pre: torch.Tensor, grad_post: torch.Tensor, grad_comb: torch.Tensor,
    mult: int, iterations: int, eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = mixes.numel() // ((mult + 2) * mult)
    history = mixes.new_empty((count, 2 * iterations - 1, mult, mult))
    grad_mixes = torch.empty_like(mixes)
    partial_base = torch.empty_like(mixes)
    partial_scale = mixes.new_empty((count, 3))
    block = triton.next_power_of_2(mult)
    wrap_triton(_coefficients_kernel)[(triton.cdiv(count, 4),)](
        mixes, scale, base, history, history, history, history, count,
        mult, iterations, eps, block, 4, True, num_warps=2, enable_fp_fusion=False,
    )
    wrap_triton(_coefficients_backward_kernel)[(triton.cdiv(count, 4),)](
        mixes, scale, base, grad_pre.contiguous(), grad_post.contiguous(), grad_comb.contiguous(),
        history, grad_mixes, partial_scale, partial_base, count, mult, iterations, eps, block, 4,
        num_warps=2, enable_fp_fusion=False,
    )
    return grad_mixes, partial_scale.sum(0), partial_base.flatten(0, -2).sum(0)


def _setup_context(ctx, inputs, output):
    mixes, scale, base, ctx.mult, ctx.iterations, ctx.eps = inputs
    ctx.save_for_backward(mixes, scale, base)


def _backward(ctx, grad_pre, grad_post, grad_comb):
    mixes, scale, base = ctx.saved_tensors
    gradients = _hc_coefficients_backward(
        mixes, scale, base, grad_pre, grad_post, grad_comb, ctx.mult, ctx.iterations, ctx.eps,
    )
    return *gradients, None, None, None


register_autograd("ffxiv_ccg::hc_coefficients", _backward, setup_context=_setup_context)
