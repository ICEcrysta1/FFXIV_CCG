"""训练专用旋转核；延迟导入使 CPU、分析和导出路径不依赖 Triton。"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import register_autograd, triton_op, wrap_triton


# 序列长度按运行时取值计算，避免训练中因长度对齐变化再次编译。
@triton.jit(do_not_specialize=["tokens"])
def _rotary_kernel(
    values, cosine, sine, output, count, tokens,
    stride_batch, stride_head, stride_token, stride_channel,
    cosine_batch, cosine_token, cosine_channel,
    sine_batch, sine_token, sine_channel,
    HEADS: tl.constexpr, DIM: tl.constexpr, ROTARY: tl.constexpr,
    INVERSE: tl.constexpr, MERGED: tl.constexpr, BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < count
    channel = offset % DIM
    row = offset // DIM
    if MERGED:
        head = row % HEADS
        token = row // HEADS % tokens
    else:
        token = row % tokens
        head = row // tokens % HEADS
    batch = row // (HEADS * tokens)
    address = batch * stride_batch + head * stride_head + token * stride_token
    original = tl.load(values + address + channel * stride_channel, valid, 0)
    rotary_channel = channel - (DIM - ROTARY)
    tail = rotary_channel >= 0
    pair = rotary_channel // 2
    partner_channel = (DIM - ROTARY) + (rotary_channel ^ 1)
    partner = tl.load(
        values + address + partner_channel * stride_channel, valid & tail, 0,
    ).to(tl.float32)
    cos = tl.load(
        cosine + batch * cosine_batch + token * cosine_token + pair * cosine_channel,
        valid & tail, 1,
    ).to(tl.float32)
    sin = tl.load(
        sine + batch * sine_batch + token * sine_token + pair * sine_channel,
        valid & tail, 0,
    ).to(tl.float32)
    if INVERSE:
        sin = -sin
    signed_sin = tl.where(rotary_channel % 2 == 0, -sin, sin)
    rotated = original.to(tl.float32) * cos + partner * signed_sin
    tl.store(output + offset, tl.where(tail, rotated, original), valid)


@triton_op("ffxiv_ccg::rotary_transform", mutates_args={})
def rotary_transform(
    values: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor,
    inverse: bool, merge_heads: bool,
) -> torch.Tensor:
    batch, heads, tokens, dim = values.shape
    shape = (batch, tokens, heads * dim) if merge_heads else values.shape
    output = values.new_empty(shape)
    wrap_triton(_rotary_kernel)[(triton.cdiv(values.numel(), 256),)](
        values, cosine, sine, output, values.numel(), tokens,
        *values.stride(), cosine.stride(0) if cosine.shape[0] != 1 else 0,
        cosine.stride(2) if cosine.shape[2] != 1 else 0, cosine.stride(3),
        sine.stride(0) if sine.shape[0] != 1 else 0,
        sine.stride(2) if sine.shape[2] != 1 else 0, sine.stride(3),
        heads, dim, cosine.shape[-1] * 2, inverse, merge_heads, 256,
        num_warps=4, enable_fp_fusion=False,
    )
    return output


def _setup_context(ctx, inputs, output):
    values, cosine, sine, ctx.inverse, ctx.merge_heads = inputs
    ctx.heads, ctx.dim = values.shape[1], values.shape[3]
    ctx.save_for_backward(cosine, sine)


def _backward(ctx, grad_output):
    cosine, sine = ctx.saved_tensors
    if ctx.merge_heads:
        grad_output = grad_output.unflatten(-1, (ctx.heads, ctx.dim)).transpose(1, 2)
    # 旋转的转置就是逆旋转；不保存原 activation，梯度直接写入逐头布局。
    gradient = rotary_transform(grad_output, cosine, sine, not ctx.inverse, False)
    return gradient, None, None, None, None


register_autograd("ffxiv_ccg::rotary_transform", _backward, setup_context=_setup_context)
