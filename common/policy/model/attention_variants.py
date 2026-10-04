"""因果编码器的注意力变体与单段 attention 执行。"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from .attention_masks import (
    SegmentAttentionMask,
    build_segment_mask,
)
from .attention_utils import (
    merge_heads,
)
from .grouped_attention import expand_kv_heads, scaled_dot_product_attention


def run_head_attention(
    layer,
    query_heads: torch.Tensor,
    key_heads: torch.Tensor,
    value_heads: torch.Tensor,
    *,
    key_valid: torch.Tensor | None = None,
    causal: bool,
    causal_offset: int = 0,
    collect_attention: bool,
    force_explicit_mask: bool = False,
    segment_mask: SegmentAttentionMask | None = None,
):
    """执行单段 attention 并投影，供 prefix/KV-cache 路径复用。"""
    attended, weights = _attend_heads(
        layer,
        query_heads,
        key_heads,
        value_heads,
        key_valid=key_valid,
        causal=causal,
        causal_offset=causal_offset,
        collect_attention=collect_attention,
        force_explicit_mask=force_explicit_mask,
        segment_mask=segment_mask,
    )
    return layer.self_attn.out_proj(merge_heads(attended)), weights


def _attend_heads(
    layer,
    query_heads: torch.Tensor,
    key_heads: torch.Tensor,
    value_heads: torch.Tensor,
    *,
    key_valid: torch.Tensor | None = None,
    causal: bool,
    causal_offset: int = 0,
    collect_attention: bool,
    force_explicit_mask: bool = False,
    segment_mask: SegmentAttentionMask | None = None,
):
    """执行分头 attention；输出投影由整段或 KV-cache 调用方统一执行。"""
    if key_heads.shape[1] != value_heads.shape[1]:
        raise ValueError(
            "key and value head counts must match: "
            f"{key_heads.shape[1]} != {value_heads.shape[1]}"
        )
    if query_heads.shape[1] % key_heads.shape[1] != 0:
        raise ValueError(
            "query head count must be divisible by key/value head count: "
            f"{query_heads.shape[1]} % {key_heads.shape[1]} != 0"
        )
    query_count = query_heads.shape[2]
    key_count = key_heads.shape[2]
    if segment_mask is None:
        if key_valid is None:
            raise ValueError("attention requires either key_valid or a precomputed mask")
        segment_mask = build_segment_mask(
            key_valid,
            query_count=query_count,
            key_count=key_count,
            causal=causal,
            causal_offset=causal_offset,
            force_explicit_mask=force_explicit_mask,
        )
    allowed = segment_mask.allowed
    if collect_attention:
        if allowed is None and causal:
            query_positions = torch.arange(
                query_count,
                device=key_heads.device,
            ).view(1, 1, query_count, 1) + causal_offset
            key_positions = torch.arange(
                key_count,
                device=key_heads.device,
            ).view(1, 1, 1, key_count)
            allowed = key_positions <= query_positions
        attended, weights = manual_attention(query_heads, key_heads, value_heads, allowed)
    else:
        fully_blocked = segment_mask.fully_blocked
        safe_allowed = segment_mask.attn_mask

        def sdpa(query_value, key_value, value_value):
            with layer._debug_stage("attention"):
                return scaled_dot_product_attention(
                    query_value,
                    key_value,
                    value_value,
                    attn_mask=safe_allowed,
                    dropout_p=(
                        float(layer.self_attn.dropout) if layer.training else 0.0
                    ),
                    is_causal=segment_mask.is_causal,
                )

        if (
            layer.activation_checkpoint_attention
            and not layer._skip_activation_checkpoint
            and layer.training
            and torch.is_grad_enabled()
        ):
            attended = checkpoint(
                sdpa,
                query_heads,
                key_heads,
                value_heads,
                use_reentrant=False,
            )
        else:
            attended = sdpa(query_heads, key_heads, value_heads)
        if fully_blocked is not None:
            attended = attended.masked_fill(fully_blocked.unsqueeze(-1), 0.0)
        weights = None

    return attended, weights


def manual_attention(
    query_heads: torch.Tensor,
    key_heads: torch.Tensor,
    value_heads: torch.Tensor,
    allowed: torch.Tensor | None,
):
    """trace 专用的权重计算，不参与正式训练/推理路径。"""
    key_heads = expand_kv_heads(key_heads, query_heads.shape[1])
    value_heads = expand_kv_heads(value_heads, query_heads.shape[1])
    scale = query_heads.shape[-1] ** -0.5
    scores = torch.matmul(query_heads, key_heads.transpose(-2, -1)) * scale
    if allowed is None:
        weights = torch.softmax(scores, dim=-1)
    else:
        fully_blocked = ~allowed.any(dim=-1)
        safe_allowed = allowed | fully_blocked.unsqueeze(-1)
        weights = torch.softmax(
            scores.masked_fill(~safe_allowed, torch.finfo(scores.dtype).min),
            dim=-1,
        )
        weights = weights.masked_fill(~allowed, 0.0)
        if bool(fully_blocked.any()):
            weights = weights.masked_fill(fully_blocked.unsqueeze(-1), 0.0)
    return torch.matmul(weights, value_heads), weights
