"""Prefix 因果与候选双向的共享 Transformer 执行路径。"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch.utils.checkpoint import checkpoint

from .attention_masks import build_split_segment_masks
from .attention_utils import (
    finish_layer,
    kv_head_count,
    project_qkv,
    rotate_qk,
    split_heads,
)
from .attention_variants import (
    _run_split_attention,
    manual_attention,
    run_head_attention,
)


def _resolve_position_ids(
    encoded: dict[str, torch.Tensor],
    tokens: torch.Tensor,
) -> torch.Tensor:
    """读取输入编码器生成的逻辑位置；独立底层测试缺省时使用顺序位置。"""
    position_ids = encoded.get("position_ids")
    if position_ids is None:
        return torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0).expand(
            tokens.shape[0], -1
        )
    if tuple(position_ids.shape) != tuple(tokens.shape[:2]):
        raise ValueError(
            "position_ids shape must match encoded tokens: "
            f"{tuple(position_ids.shape)} != {tuple(tokens.shape[:2])}"
        )
    return position_ids


def run_split_encoder(
    encoder,
    encoded: dict[str, torch.Tensor],
    *,
    collect_attention: bool = False,
    force_explicit_mask: bool = False,
):
    """按层执行 prefix causal + candidate bidirectional 编码。"""
    tokens = encoded["tokens"]
    prefix_length = int(encoded["prefix_length"])
    candidate_count = int(encoded["candidate_count"])
    position_ids = _resolve_position_ids(encoded, tokens)
    rotary_position_encoding = getattr(encoder, "rotary_position_encoding", None)
    if rotary_position_encoding is None:
        raise ValueError("encoder is missing the required RotaryPositionEncoding module")
    prefix_hidden = tokens[:, :prefix_length]
    candidate_hidden = tokens[:, prefix_length : prefix_length + candidate_count]
    cls_hidden = tokens[:, prefix_length + candidate_count :]
    prefix_valid = encoded["prefix_valid"]
    candidate_valid = encoded["candidate_valid"]
    cls_valid = encoded["cls_valid"]
    # mask 只取决于 batch 的有效性布局，与层无关；这里一次算好给所有层复用，
    # 避免每层重复构造并触发 device 到 host 的同步。
    segment_masks = build_split_segment_masks(
        prefix_valid,
        candidate_valid,
        cls_valid,
        prefix_length=prefix_length,
        candidate_count=candidate_count,
        cls_count=cls_hidden.shape[1],
        force_explicit_mask=force_explicit_mask,
    )

    attention_residual = getattr(encoder, "attention_residual", None)
    layer_hidden: list[torch.Tensor] = []
    attentions: list[torch.Tensor] = []
    if attention_residual is None:
        for layer in encoder.layers:
            prefix_hidden, candidate_hidden, cls_hidden, attention = run_split_layer(
                layer,
                prefix_hidden,
                candidate_hidden,
                cls_hidden,
                prefix_valid=prefix_valid,
                candidate_valid=candidate_valid,
                cls_valid=cls_valid,
                position_ids=position_ids,
                rotary_position_encoding=rotary_position_encoding,
                collect_attention=collect_attention,
                force_explicit_mask=force_explicit_mask,
                segment_masks=segment_masks,
            )
            if collect_attention:
                layer_hidden.append(
                    torch.cat((prefix_hidden, candidate_hidden, cls_hidden), dim=1)
                )
                attentions.append(attention)
    else:
        # 深度聚合逐 token 独立进行，三个 token 分区可以共享完整 source 列表。
        if _should_checkpoint_full_attention_residual(encoder, collect_attention):
            def recompute_residual_path(
                checkpoint_tokens,
                checkpoint_prefix_valid,
                checkpoint_candidate_valid,
                checkpoint_cls_valid,
            ):
                outputs = _run_full_attention_residual_path(
                    encoder,
                    checkpoint_tokens,
                    prefix_length=prefix_length,
                    candidate_count=candidate_count,
                    prefix_valid=checkpoint_prefix_valid,
                    candidate_valid=checkpoint_candidate_valid,
                    cls_valid=checkpoint_cls_valid,
                    position_ids=position_ids,
                    rotary_position_encoding=rotary_position_encoding,
                    collect_attention=False,
                    force_explicit_mask=force_explicit_mask,
                    segment_masks=segment_masks,
                )
                return outputs[:3]

            prefix_hidden, candidate_hidden, cls_hidden = checkpoint(
                recompute_residual_path,
                tokens,
                prefix_valid,
                candidate_valid,
                cls_valid,
                use_reentrant=False,
                context_fn=lambda: (
                    _skip_layer_activation_checkpoints(encoder.layers),
                    _skip_layer_activation_checkpoints(encoder.layers),
                ),
            )
            if encoder.norm is not None:
                prefix_hidden = encoder.norm(prefix_hidden)
                candidate_hidden = encoder.norm(candidate_hidden)
                cls_hidden = encoder.norm(cls_hidden)
            return prefix_hidden, candidate_hidden, cls_hidden, (), ()

        (
            prefix_hidden,
            candidate_hidden,
            cls_hidden,
            layer_hidden,
            attentions,
        ) = _run_full_attention_residual_path(
            encoder,
            tokens,
            prefix_length=prefix_length,
            candidate_count=candidate_count,
            prefix_valid=prefix_valid,
            candidate_valid=candidate_valid,
            cls_valid=cls_valid,
            position_ids=position_ids,
            rotary_position_encoding=rotary_position_encoding,
            collect_attention=collect_attention,
            force_explicit_mask=force_explicit_mask,
            segment_masks=segment_masks,
        )

    if encoder.norm is not None:
        prefix_hidden = encoder.norm(prefix_hidden)
        candidate_hidden = encoder.norm(candidate_hidden)
        cls_hidden = encoder.norm(cls_hidden)

    return (
        prefix_hidden,
        candidate_hidden,
        cls_hidden,
        tuple(layer_hidden),
        tuple(attentions),
    )


def _should_checkpoint_full_attention_residual(encoder, collect_attention: bool) -> bool:
    """仅在所有层配置一致时用整段 checkpoint 释放 Full AttnRes source 激活。"""
    if collect_attention or not encoder.training or not torch.is_grad_enabled():
        return False
    layer_flags = tuple(
        (layer.activation_checkpoint_ffn, layer.activation_checkpoint_attention)
        for layer in encoder.layers
    )
    if not layer_flags or not any(layer_flags[0]):
        return False
    return all(flags == layer_flags[0] for flags in layer_flags)


@contextmanager
def _skip_layer_activation_checkpoints(layers):
    """整段或整块 checkpoint 时临时跳过层内子 checkpoint。"""
    layers = tuple(layers)
    previous = tuple(layer._skip_activation_checkpoint for layer in layers)
    for layer in layers:
        layer._skip_activation_checkpoint = True
    try:
        yield
    finally:
        for layer, was_skipped in zip(layers, previous):
            layer._skip_activation_checkpoint = was_skipped


def _run_full_attention_residual_path(
    encoder,
    tokens: torch.Tensor,
    *,
    prefix_length: int,
    candidate_count: int,
    prefix_valid: torch.Tensor,
    candidate_valid: torch.Tensor,
    cls_valid: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_position_encoding,
    collect_attention: bool,
    force_explicit_mask: bool,
    segment_masks,
):
    """执行 Full AttnRes 主路径，供普通和 checkpoint 路径共用。"""
    attention_residual = encoder.attention_residual
    sources = [tokens]
    hidden = tokens
    layer_hidden: list[torch.Tensor] = []
    attentions: list[torch.Tensor] = []
    for layer_index, layer in enumerate(encoder.layers):
        hidden, attention = run_attention_residual_layer(
            layer,
            sources,
            attention_residual,
            query_index=2 * layer_index,
            prefix_length=prefix_length,
            candidate_count=candidate_count,
            prefix_valid=prefix_valid,
            candidate_valid=candidate_valid,
            cls_valid=cls_valid,
            position_ids=position_ids,
            rotary_position_encoding=rotary_position_encoding,
            collect_attention=collect_attention,
            force_explicit_mask=force_explicit_mask,
            segment_masks=segment_masks,
        )
        if collect_attention:
            layer_hidden.append(hidden)
            attentions.append(attention)
    prefix_hidden = hidden[:, :prefix_length]
    candidate_hidden = hidden[:, prefix_length : prefix_length + candidate_count]
    cls_hidden = hidden[:, prefix_length + candidate_count :]
    return (
        prefix_hidden,
        candidate_hidden,
        cls_hidden,
        tuple(layer_hidden),
        tuple(attentions),
    )


def run_split_layer(
    layer,
    prefix_hidden: torch.Tensor,
    candidate_hidden: torch.Tensor,
    cls_hidden: torch.Tensor,
    *,
    prefix_valid: torch.Tensor,
    candidate_valid: torch.Tensor,
    cls_valid: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_position_encoding,
    collect_attention: bool = False,
    force_explicit_mask: bool = False,
    segment_masks=None,
):
    """执行一层共享权重的 prefix、candidate 和 CLS 路径。"""
    lengths = (prefix_hidden.shape[1], candidate_hidden.shape[1], cls_hidden.shape[1])
    hidden = torch.cat((prefix_hidden, candidate_hidden, cls_hidden), dim=1)
    if (
        layer.activation_checkpoint_attention
        and layer.activation_checkpoint_attention_block
        and not layer._skip_activation_checkpoint
        and layer.training
        and torch.is_grad_enabled()
        and not collect_attention
    ):
        # 整块 attention（norm + Q/K/V 投影 + 三段 SDPA + merge + out_proj）作为一次
        # checkpoint：反向只重算这一块，块内 SDPA 不再单独 checkpoint，省下投影与
        # attention 输出的全部中间激活。
        def attention_block(block_input: torch.Tensor) -> torch.Tensor:
            with _skip_layer_activation_checkpoints((layer,)):
                blocked, _ = _run_split_attention(
                    layer,
                    block_input,
                    prefix_length=lengths[0],
                    candidate_count=lengths[1],
                    prefix_valid=prefix_valid,
                    candidate_valid=candidate_valid,
                    cls_valid=cls_valid,
                    position_ids=position_ids,
                    rotary_position_encoding=rotary_position_encoding,
                    collect_attention=False,
                    force_explicit_mask=force_explicit_mask,
                    segment_masks=segment_masks,
                )
            return blocked

        attended = checkpoint(attention_block, hidden, use_reentrant=False)
        attention = None
    else:
        attended, attention = _run_split_attention(
            layer,
            hidden,
            prefix_length=lengths[0],
            candidate_count=lengths[1],
            prefix_valid=prefix_valid,
            candidate_valid=candidate_valid,
            cls_valid=cls_valid,
            position_ids=position_ids,
            rotary_position_encoding=rotary_position_encoding,
            collect_attention=collect_attention,
            force_explicit_mask=force_explicit_mask,
            segment_masks=segment_masks,
        )

    # 残差、LayerNorm 和 FFN 均逐 token 独立，整段执行可共用一次投影。
    hidden = finish_layer(layer, hidden, attended)
    prefix_hidden, candidate_hidden, cls_hidden = hidden.split(lengths, dim=1)
    return prefix_hidden, candidate_hidden, cls_hidden, attention


def run_attention_residual_layer(
    layer,
    sources: list[torch.Tensor],
    attention_residual,
    *,
    query_index: int,
    prefix_length: int,
    candidate_count: int,
    prefix_valid: torch.Tensor,
    candidate_valid: torch.Tensor,
    cls_valid: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_position_encoding,
    collect_attention: bool = False,
    force_explicit_mask: bool = False,
    segment_masks=None,
):
    """执行一个 Full AttnRes block，并追加 attention/FFN 两个深度源。"""
    if not layer.norm_first:
        raise ValueError("full attention residuals require Pre-LN transformer layers")

    hidden = attention_residual(sources, query_index)
    attended, attention = _run_split_attention(
        layer,
        hidden,
        prefix_length=prefix_length,
        candidate_count=candidate_count,
        prefix_valid=prefix_valid,
        candidate_valid=candidate_valid,
        cls_valid=cls_valid,
        position_ids=position_ids,
        rotary_position_encoding=rotary_position_encoding,
        collect_attention=collect_attention,
        force_explicit_mask=force_explicit_mask,
        segment_masks=segment_masks,
    )

    sources.append(layer.dropout1(attended))
    mlp_hidden = attention_residual(sources, query_index + 1)
    sources.append(layer._ff_block(layer.norm2(mlp_hidden)))
    return attention_residual(sources, query_index + 2), attention


def run_cached_candidate_layer(
    layer,
    candidate_hidden: torch.Tensor,
    cls_hidden: torch.Tensor,
    *,
    prefix_key: torch.Tensor,
    prefix_value: torch.Tensor,
    prefix_valid: torch.Tensor,
    prefix_position_ids: torch.Tensor,
    candidate_position_ids: torch.Tensor,
    cls_position_ids: torch.Tensor,
    candidate_valid: torch.Tensor,
    cls_valid: torch.Tensor,
    rotary_position_encoding,
    attention_residual=None,
    candidate_sources: list[torch.Tensor] | None = None,
    cls_sources: list[torch.Tensor] | None = None,
    query_index: int = 0,
    segment_masks=None,
):
    """使用已缓存 prefix K/V 计算当前候选和 CLS。"""
    if attention_residual is not None:
        if not layer.norm_first:
            raise ValueError("full attention residuals require Pre-LN transformer layers")
        if candidate_sources is None or cls_sources is None:
            raise ValueError("Full AttnRes cached execution requires source lists")
        candidate_input = layer.norm1(attention_residual(candidate_sources, query_index))
        cls_input = layer.norm1(attention_residual(cls_sources, query_index))
    elif layer.norm_first:
        candidate_input = layer.norm1(candidate_hidden)
        cls_input = layer.norm1(cls_hidden)
    else:
        candidate_input = candidate_hidden
        cls_input = cls_hidden

    candidate_query, candidate_key, candidate_value = project_qkv(
        layer.self_attn,
        candidate_input,
    )
    cls_query, cls_key, cls_value = project_qkv(layer.self_attn, cls_input)
    candidate_query_heads, candidate_key_heads = rotate_qk(
        rotary_position_encoding,
        split_heads(candidate_query, layer.self_attn.num_heads),
        split_heads(candidate_key, kv_head_count(layer.self_attn)),
        candidate_position_ids,
        candidate_position_ids,
    )
    cls_query_heads, cls_key_heads = rotate_qk(
        rotary_position_encoding,
        split_heads(cls_query, layer.self_attn.num_heads),
        split_heads(cls_key, kv_head_count(layer.self_attn)),
        cls_position_ids,
        cls_position_ids,
    )
    candidate_value_heads = split_heads(candidate_value, kv_head_count(layer.self_attn))
    cls_value_heads = split_heads(cls_value, kv_head_count(layer.self_attn))

    if segment_masks is None:
        candidate_mask = cls_mask = None
    else:
        candidate_mask, cls_mask = segment_masks
    candidate_attended, _ = run_head_attention(
        layer,
        candidate_query_heads,
        torch.cat((prefix_key, candidate_key_heads), dim=2),
        torch.cat((prefix_value, candidate_value_heads), dim=2),
        key_valid=(
            None
            if candidate_mask is not None
            else torch.cat((prefix_valid, candidate_valid), dim=1)
        ),
        causal=False,
        collect_attention=False,
        segment_mask=candidate_mask,
    )
    cls_attended, _ = run_head_attention(
        layer,
        cls_query_heads,
        torch.cat((prefix_key, candidate_key_heads, cls_key_heads), dim=2),
        torch.cat((prefix_value, candidate_value_heads, cls_value_heads), dim=2),
        key_valid=(
            None
            if cls_mask is not None
            else torch.cat((prefix_valid, candidate_valid, cls_valid), dim=1)
        ),
        causal=False,
        collect_attention=False,
        segment_mask=cls_mask,
    )
    if attention_residual is not None:
        candidate_sources.append(layer.dropout1(candidate_attended))
        cls_sources.append(layer.dropout1(cls_attended))
        candidate_sources.append(
            layer._ff_block(
                layer.norm2(attention_residual(candidate_sources, query_index + 1))
            )
        )
        cls_sources.append(
            layer._ff_block(
                layer.norm2(attention_residual(cls_sources, query_index + 1))
            )
        )
        return (
            attention_residual(candidate_sources, query_index + 2),
            attention_residual(cls_sources, query_index + 2),
        )
    return (
        finish_layer(layer, candidate_hidden, candidate_attended),
        finish_layer(layer, cls_hidden, cls_attended),
    )


def run_prefix_layer(
    layer,
    prefix_hidden: torch.Tensor,
    *,
    prefix_valid: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_position_encoding,
    existing_key: torch.Tensor | None = None,
    existing_value: torch.Tensor | None = None,
    existing_length: int = 0,
    attention_residual=None,
    sources: list[torch.Tensor] | None = None,
    query_index: int = 0,
):
    """执行 prefix 因果层，并返回本层新增/完整 K/V。"""
    if attention_residual is not None:
        if not layer.norm_first:
            raise ValueError("full attention residuals require Pre-LN transformer layers")
        if sources is None:
            raise ValueError("Full AttnRes prefix execution requires a source list")
        prefix_hidden = attention_residual(sources, query_index)
    prefix_input = layer.norm1(prefix_hidden) if layer.norm_first else prefix_hidden
    query, key, value = project_qkv(layer.self_attn, prefix_input)
    query_heads, key_heads = rotate_qk(
        rotary_position_encoding,
        split_heads(query, layer.self_attn.num_heads),
        split_heads(key, kv_head_count(layer.self_attn)),
        position_ids,
        position_ids,
    )
    value_heads = split_heads(value, kv_head_count(layer.self_attn))
    if existing_key is None:
        all_key = key_heads
        all_value = value_heads
        causal_offset = 0
    else:
        all_key = torch.cat((existing_key, key_heads), dim=2)
        all_value = torch.cat((existing_value, value_heads), dim=2)
        causal_offset = existing_length
    attended, _ = run_head_attention(
        layer,
        query_heads,
        all_key,
        all_value,
        key_valid=prefix_valid,
        causal=True,
        causal_offset=causal_offset,
        collect_attention=False,
    )
    if attention_residual is not None:
        sources.append(layer.dropout1(attended))
        sources.append(
            layer._ff_block(
                layer.norm2(attention_residual(sources, query_index + 1))
            )
        )
        return attention_residual(sources, query_index + 2), key_heads, value_heads
    return finish_layer(layer, prefix_hidden, attended), key_heads, value_heads
