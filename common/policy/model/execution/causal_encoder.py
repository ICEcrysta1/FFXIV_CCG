"""单一因果序列的共享 Transformer 执行路径。"""

from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from ..attention.attention_masks import build_segment_mask
from ..attention.attention_utils import (
    finish_layer,
    kv_head_count,
    normalize_qk,
    project_qkv,
    rotate_qk,
    split_heads,
)
from ..attention.attention_variants import run_head_attention


def run_causal_layer(
    layer,
    hidden: torch.Tensor,
    *,
    key_valid: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_position_encoding,
    existing_key: torch.Tensor | None = None,
    existing_value: torch.Tensor | None = None,
    existing_length: int = 0,
    attention_residual=None,
    residual_mix=None,
    pre_mix: torch.Tensor | None = None,
    layer_index: int = 0,
    sources: list[torch.Tensor] | None = None,
    collect_attention: bool = False,
    force_explicit_mask: bool = False,
    segment_mask=None,
):
    """编码完整序列或新增因果后缀，返回残差流、新 K/V、注意力和下个 pre。

    Full AttnRes 的 hidden 已由调用方或上一层聚合，本层追加两个子层源。
    """
    if residual_mix is not None:
        if attention_residual is not None:
            raise ValueError("Hyper-Connections and Full AttnRes are mutually exclusive")
        if not layer.norm_first or hidden.ndim != 4 or pre_mix is None:
            raise ValueError("Hyper-Connections requires Pre-Norm residual streams and pre_mix")
        attn_mix, ffn_mix = residual_mix[2 * layer_index:2 * layer_index + 2]
        attn_pre, attn_post, attn_comb = attn_mix.mixes(hidden)
        attention_hidden = attn_mix.collapse(hidden, pre_mix)
    else:
        attention_hidden = hidden
    if attention_residual is not None and (not layer.norm_first or sources is None):
        raise ValueError("Full AttnRes requires Pre-LN layers and a source list")

    def attend(attention_hidden):
        attention_input = layer.norm1(attention_hidden) if layer.norm_first else attention_hidden
        query, key, value = project_qkv(layer.self_attn, attention_input)
        query_heads, key_heads = rotate_qk(
            rotary_position_encoding,
            split_heads(query, layer.self_attn.num_heads),
            split_heads(key, kv_head_count(layer.self_attn)),
            position_ids,
            position_ids,
        )
        # 只归一化本次投影的新 K；缓存已保存归一化后的 K，拼接时不再处理。
        query_heads, key_heads = normalize_qk(
            query_heads, key_heads, scale=layer.qk_norm_scale,
        )
        value_heads = split_heads(value, kv_head_count(layer.self_attn))
        if existing_key is None:
            all_key, all_value = key_heads, value_heads
        else:
            all_key = torch.cat((existing_key, key_heads), dim=2)
            all_value = torch.cat((existing_value, value_heads), dim=2)
        attended, weights = run_head_attention(
            layer,
            query_heads,
            all_key,
            all_value,
            key_valid=key_valid,
            causal=True,
            causal_offset=existing_length,
            collect_attention=collect_attention,
            force_explicit_mask=force_explicit_mask,
            segment_mask=segment_mask,
        )
        return attended, key_heads, value_heads, weights

    if (
        layer.activation_checkpoint_attention
        and layer.activation_checkpoint_attention_block
        and not layer._skip_activation_checkpoint
        and layer.training
        and torch.is_grad_enabled()
        and not collect_attention
    ):
        def attention_block(block_hidden):
            # 整块重算已覆盖 SDPA，不再嵌套它自己的 checkpoint。
            was_skipped = layer._skip_activation_checkpoint
            layer._skip_activation_checkpoint = True
            try:
                return attend(block_hidden)[:3]
            finally:
                layer._skip_activation_checkpoint = was_skipped

        attended, key, value = checkpoint(attention_block, attention_hidden, use_reentrant=False)
        weights = None
    else:
        attended, key, value, weights = attend(attention_hidden)

    if residual_mix is not None:
        hidden = attn_mix.combine(layer.dropout1(attended), hidden, attn_post, attn_comb)
        pre_mix, ffn_post, ffn_comb = ffn_mix.mixes(hidden)
        ffn_input = layer.norm2(ffn_mix.collapse(hidden, attn_pre))
        hidden = ffn_mix.combine(layer._ff_block(ffn_input), hidden, ffn_post, ffn_comb)
    elif attention_residual is not None:
        sources.append(layer.dropout1(attended))
        sources.append(layer._ff_block(layer.norm2(attention_residual(sources, 2 * layer_index + 1))))
        # 此聚合同时是下一层 attention 的输入，层边界不再重复计算同一个 query。
        hidden = attention_residual(sources, 2 * layer_index + 2)
    else:
        hidden = finish_layer(layer, hidden, attended)
    return hidden, key, value, weights, pre_mix


def initialize_encoder_state(encoder, tokens: torch.Tensor):
    """训练、增量缓存与分析共用初始残差布局。"""
    residual_mix = getattr(encoder, "residual_mix", None)
    if residual_mix is not None:
        hidden, pre_mix = residual_mix[0].expand(tokens)
        return hidden, pre_mix, None
    residual = getattr(encoder, "attention_residual", None)
    sources = [tokens] if residual is not None else None
    return (residual(sources, 0) if residual is not None else tokens), None, sources


def collapse_encoder_state(encoder, hidden: torch.Tensor, pre_mix: torch.Tensor | None):
    """按最后一个子层的 pre 读取流，保持分析和最终读出为三维 token。"""
    residual_mix = getattr(encoder, "residual_mix", None)
    return residual_mix[0].collapse(hidden, pre_mix) if residual_mix is not None else hidden


def run_causal_encoder(
    encoder,
    encoded: dict[str, torch.Tensor],
    *,
    collect_attention: bool = False,
    force_explicit_mask: bool = False,
):
    """所有有效 token 使用相同下三角可见性，训练与 trace 共用执行语义。"""
    tokens = encoded["tokens"]
    valid = encoded["valid"]
    position_ids = encoded.get("position_ids")
    if position_ids is None:
        position_ids = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0).expand(
            tokens.shape[0], -1,
        )
    if position_ids.shape != tokens.shape[:2] or valid.shape != tokens.shape[:2]:
        raise ValueError("positions and validity must match the encoded token layout")
    rotary = getattr(encoder, "rotary_position_encoding", None)
    if rotary is None:
        raise ValueError("encoder is missing the required RotaryPositionEncoding module")
    # 同一 batch 的 mask 只构造一次，所有层及重算路径共用。
    mask = build_segment_mask(
        valid,
        query_count=tokens.shape[1],
        key_count=tokens.shape[1],
        causal=True,
        force_explicit_mask=force_explicit_mask,
    )
    residual = getattr(encoder, "attention_residual", None)
    residual_mix = getattr(encoder, "residual_mix", None)

    hidden, pre_mix, sources = initialize_encoder_state(encoder, tokens)
    layer_hidden, attentions = [], []
    for index, layer in enumerate(encoder.layers):
        hidden, _, _, attention, pre_mix = run_causal_layer(
            layer,
            hidden,
            key_valid=valid,
            position_ids=position_ids,
            rotary_position_encoding=rotary,
            attention_residual=residual,
            residual_mix=residual_mix,
            pre_mix=pre_mix,
            layer_index=index,
            sources=sources,
            collect_attention=collect_attention,
            force_explicit_mask=force_explicit_mask,
            segment_mask=mask,
        )
        if collect_attention:
            layer_hidden.append(collapse_encoder_state(encoder, hidden, pre_mix))
            attentions.append(attention)
    hidden = collapse_encoder_state(encoder, hidden, pre_mix)
    if encoder.norm is not None:
        hidden = encoder.norm(hidden)
    return hidden, tuple(layer_hidden), tuple(attentions)
