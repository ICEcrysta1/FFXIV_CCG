"""单一因果序列的共享 Transformer 执行路径。"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch.utils.checkpoint import checkpoint

from .attention_masks import build_segment_mask
from .attention_utils import (
    finish_layer,
    kv_head_count,
    normalize_qk,
    project_qkv,
    rotate_qk,
    split_heads,
)
from .attention_variants import run_head_attention


@contextmanager
def _skip_layer_activation_checkpoints(layers):
    """整段或整块重算时，避免重复嵌套层内 checkpoint。"""
    layers = tuple(layers)
    previous = tuple(layer._skip_activation_checkpoint for layer in layers)
    for layer in layers:
        layer._skip_activation_checkpoint = True
    try:
        yield
    finally:
        for layer, was_skipped in zip(layers, previous):
            layer._skip_activation_checkpoint = was_skipped


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
    initial_tokens: torch.Tensor | None = None,
    layer_index: int = 0,
    sources: list[torch.Tensor] | None = None,
    query_index: int = 0,
    collect_attention: bool = False,
    force_explicit_mask: bool = False,
    segment_mask=None,
):
    """编码完整序列或新增因果后缀，返回 hidden、本层新增 K/V 与注意力。"""
    if residual_mix is not None:
        if attention_residual is not None:
            raise ValueError("learned residual mixing and Full AttnRes are mutually exclusive")
        if initial_tokens is None or initial_tokens.shape != hidden.shape:
            raise ValueError("residual mixing requires initial tokens aligned with the encoded block")
        # 混合后的 hidden 同时作为 attention 输入和该层 skip；FFN 不再次混合。
        hidden = residual_mix(hidden, initial_tokens, layer_index)
    if attention_residual is not None:
        if not layer.norm_first or sources is None:
            raise ValueError("Full AttnRes requires Pre-LN layers and a source list")
        hidden = attention_residual(sources, query_index)

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
            with _skip_layer_activation_checkpoints((layer,)):
                return attend(block_hidden)[:3]

        attended, key, value = checkpoint(attention_block, hidden, use_reentrant=False)
        weights = None
    else:
        attended, key, value, weights = attend(hidden)

    if attention_residual is not None:
        sources.append(layer.dropout1(attended))
        sources.append(layer._ff_block(layer.norm2(attention_residual(sources, query_index + 1))))
        hidden = attention_residual(sources, query_index + 2)
    else:
        hidden = finish_layer(layer, hidden, attended)
    return hidden, key, value, weights


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

    def run_path(path_tokens, path_valid):
        hidden = path_tokens
        sources = [path_tokens] if residual is not None else None
        layer_hidden, attentions = [], []
        for index, layer in enumerate(encoder.layers):
            hidden, _, _, attention = run_causal_layer(
                layer,
                hidden,
                key_valid=path_valid,
                position_ids=position_ids,
                rotary_position_encoding=rotary,
                attention_residual=residual,
                residual_mix=residual_mix,
                initial_tokens=path_tokens,
                layer_index=index,
                sources=sources,
                query_index=2 * index,
                collect_attention=collect_attention,
                force_explicit_mask=force_explicit_mask,
                segment_mask=mask,
            )
            if collect_attention:
                layer_hidden.append(hidden)
                attentions.append(attention)
        return hidden, tuple(layer_hidden), tuple(attentions)

    flags = tuple(
        (layer.activation_checkpoint_ffn, layer.activation_checkpoint_attention)
        for layer in encoder.layers
    )
    checkpoint_residual = (
        residual is not None
        and not collect_attention
        and encoder.training
        and torch.is_grad_enabled()
        and bool(flags)
        and any(flags[0])
        and all(flag == flags[0] for flag in flags)
    )
    if checkpoint_residual:
        hidden = checkpoint(
            lambda path_tokens, path_valid: run_path(path_tokens, path_valid)[0],
            tokens,
            valid,
            use_reentrant=False,
            context_fn=lambda: (
                _skip_layer_activation_checkpoints(encoder.layers),
                _skip_layer_activation_checkpoints(encoder.layers),
            ),
        )
        layer_hidden, attentions = (), ()
    else:
        hidden, layer_hidden, attentions = run_path(tokens, valid)
    if encoder.norm is not None:
        hidden = encoder.norm(hidden)
    return hidden, layer_hidden, attentions
