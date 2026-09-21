"""split encoder 的注意力变体与单段 attention 执行。"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from .attention_masks import assemble_attention, build_allowed_mask
from .attention_utils import (
    kv_head_count,
    merge_heads,
    project_qkv,
    rotate_qk,
    split_heads,
)
from .grouped_attention import expand_kv_heads, scaled_dot_product_attention


def _run_split_attention(
    layer,
    hidden: torch.Tensor,
    *,
    prefix_length: int,
    candidate_count: int,
    prefix_valid: torch.Tensor,
    candidate_valid: torch.Tensor,
    cls_valid: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_position_encoding,
    collect_attention: bool = False,
    force_explicit_mask: bool = False,
):
    """只运行 self-attention，供标准残差和 Full AttnRes 共同使用。"""
    attention_input = layer.norm1(hidden) if layer.norm_first else hidden
    query, key, value = project_qkv(layer.self_attn, attention_input)
    query_heads, key_heads = rotate_qk(
        rotary_position_encoding,
        split_heads(query, layer.self_attn.num_heads),
        split_heads(key, kv_head_count(layer.self_attn)),
        position_ids,
        position_ids,
    )
    value_heads = split_heads(value, kv_head_count(layer.self_attn))
    candidate_end = prefix_length + candidate_count

    # 三段各自保留原有可见范围；K/V 直接切片，不再重复拼接。
    prefix_attended, prefix_attention = _attend_heads(
        layer,
        query_heads[:, :, :prefix_length],
        key_heads[:, :, :prefix_length],
        value_heads[:, :, :prefix_length],
        key_valid=prefix_valid,
        causal=True,
        collect_attention=collect_attention,
        force_explicit_mask=force_explicit_mask,
    )
    candidate_attended, candidate_attention = _attend_heads(
        layer,
        query_heads[:, :, prefix_length:candidate_end],
        key_heads[:, :, :candidate_end],
        value_heads[:, :, :candidate_end],
        key_valid=torch.cat((prefix_valid, candidate_valid), dim=1),
        causal=False,
        collect_attention=collect_attention,
        force_explicit_mask=force_explicit_mask,
    )
    cls_attended, cls_attention = _attend_heads(
        layer,
        query_heads[:, :, candidate_end:],
        key_heads,
        value_heads,
        key_valid=torch.cat((prefix_valid, candidate_valid, cls_valid), dim=1),
        causal=False,
        collect_attention=collect_attention,
        force_explicit_mask=force_explicit_mask,
    )

    if not collect_attention:
        attention = None
    else:
        attention = assemble_attention(
            prefix_attention,
            candidate_attention,
            cls_attention,
            prefix_length=prefix_length,
            candidate_count=candidate_count,
        )
    attended = merge_heads(
        torch.cat((prefix_attended, candidate_attended, cls_attended), dim=2)
    )
    return layer.self_attn.out_proj(attended), attention


def run_head_attention(
    layer,
    query_heads: torch.Tensor,
    key_heads: torch.Tensor,
    value_heads: torch.Tensor,
    *,
    key_valid: torch.Tensor,
    causal: bool,
    causal_offset: int = 0,
    collect_attention: bool,
    force_explicit_mask: bool = False,
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
    )
    return layer.self_attn.out_proj(merge_heads(attended)), weights


def _attend_heads(
    layer,
    query_heads: torch.Tensor,
    key_heads: torch.Tensor,
    value_heads: torch.Tensor,
    *,
    key_valid: torch.Tensor,
    causal: bool,
    causal_offset: int = 0,
    collect_attention: bool,
    force_explicit_mask: bool = False,
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
    allowed = build_allowed_mask(
        key_valid,
        query_count=query_count,
        key_count=key_count,
        causal=causal,
        causal_offset=causal_offset,
        force_explicit_mask=force_explicit_mask,
    )
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
        fully_blocked = None
        if allowed is not None:
            fully_blocked = ~allowed.any(dim=-1)
            safe_allowed = allowed | fully_blocked.unsqueeze(-1)
        else:
            safe_allowed = None

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
                    is_causal=causal and safe_allowed is None,
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


# 暴露无下划线别名，便于新模块内部测试按职责调用。
run_split_attention = _run_split_attention
