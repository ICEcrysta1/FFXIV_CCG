"""推理专用的稳定场景/历史因果 KV-cache。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .attention_masks import build_segment_mask
from .causal_encoder import run_causal_layer


@dataclass
class TransformerKVCache:
    """只缓存稳定场景和历史，各步当前状态按真实请求重新编码。"""

    prefix_tokens: torch.Tensor
    prefix_valid: torch.Tensor
    prefix_position_ids: torch.Tensor
    key_cache: tuple[torch.Tensor, ...]
    value_cache: tuple[torch.Tensor, ...]
    scene_length: int
    history_length: int
    history_token_length: int


def encode_with_kv_cache(encoder, encoded, cache=None):
    """复用因果前缀，用同一因果层编码本次最新状态。"""
    tokens = encoded["tokens"]
    prefix_length = int(encoded["prefix_length"])
    scene_length = int(encoded["scene_length"])
    prefix_tokens = tokens[:, :prefix_length]
    prefix_valid = encoded["prefix_valid"]
    position_ids = encoded["position_ids"]
    prefix_positions = position_ids[:, :prefix_length]
    residual = getattr(encoder, "attention_residual", None)
    rotary = getattr(encoder, "rotary_position_encoding", None)
    if rotary is None:
        raise ValueError("encoder is missing the required RotaryPositionEncoding module")
    if not _cache_matches(cache, prefix_tokens, prefix_valid, prefix_positions, scene_length):
        cache = _build_prefix_cache(
            encoder, prefix_tokens, prefix_valid, prefix_positions, scene_length,
            attention_residual=residual, rotary_position_encoding=rotary,
        )
    elif prefix_length > cache.prefix_tokens.shape[1]:
        cache = _append_prefix(
            encoder, cache, prefix_tokens, prefix_valid, prefix_positions,
            attention_residual=residual, rotary_position_encoding=rotary,
        )
    hidden = tokens[:, prefix_length:]
    sources = [hidden] if residual is not None else None
    valid = encoded["valid"]
    mask = build_segment_mask(
        valid, query_count=1, key_count=valid.shape[1],
        causal=True, causal_offset=prefix_length,
    )
    for index, layer in enumerate(encoder.layers):
        hidden, _, _, _ = run_causal_layer(
            layer, hidden, key_valid=valid,
            position_ids=position_ids[:, prefix_length:],
            rotary_position_encoding=rotary,
            existing_key=cache.key_cache[index], existing_value=cache.value_cache[index],
            existing_length=prefix_length,
            attention_residual=residual, sources=sources, query_index=2 * index,
            segment_mask=mask,
        )
    if encoder.norm is not None:
        hidden = encoder.norm(hidden)
    return hidden, cache


def _cache_matches(
    cache: TransformerKVCache | None,
    prefix_tokens: torch.Tensor,
    prefix_valid: torch.Tensor,
    prefix_position_ids: torch.Tensor,
    scene_length: int,
) -> bool:
    if cache is None:
        return False
    cached_length = cache.prefix_tokens.shape[1]
    if prefix_tokens.shape[1] < cached_length:
        return False
    if cache.scene_length != scene_length:
        return False
    if not torch.equal(prefix_tokens[:, :cached_length], cache.prefix_tokens):
        return False
    if not torch.equal(prefix_valid[:, :cached_length], cache.prefix_valid):
        return False
    if not torch.equal(prefix_position_ids[:, :cached_length], cache.prefix_position_ids):
        return False
    return True


def _build_prefix_cache(
    encoder: torch.nn.Module,
    prefix_tokens: torch.Tensor,
    prefix_valid: torch.Tensor,
    prefix_position_ids: torch.Tensor,
    scene_length: int,
    *,
    attention_residual=None,
    rotary_position_encoding,
) -> TransformerKVCache:
    hidden = prefix_tokens
    sources = [prefix_tokens] if attention_residual is not None else None
    key_cache: list[torch.Tensor] = []
    value_cache: list[torch.Tensor] = []

    for layer_index, layer in enumerate(encoder.layers):
        if hidden.shape[1] == 0:
            head_count = int(getattr(layer.self_attn, "num_kv_heads", layer.self_attn.num_heads))
            head_dim = layer.self_attn.head_dim
            empty_shape = (hidden.shape[0], head_count, 0, head_dim)
            key_cache.append(hidden.new_empty(empty_shape))
            value_cache.append(hidden.new_empty(empty_shape))
            continue

        hidden, key, value, _ = run_causal_layer(
            layer,
            hidden,
            key_valid=prefix_valid,
            position_ids=prefix_position_ids,
            rotary_position_encoding=rotary_position_encoding,
            attention_residual=attention_residual,
            sources=sources,
            query_index=2 * layer_index,
        )
        key_cache.append(key)
        value_cache.append(value)

    prefix_length = prefix_tokens.shape[1]
    return TransformerKVCache(
        prefix_tokens=prefix_tokens.detach().clone(),
        prefix_valid=prefix_valid.detach().clone(),
        prefix_position_ids=prefix_position_ids.detach().clone(),
        key_cache=tuple(key_cache),
        value_cache=tuple(value_cache),
        scene_length=scene_length,
        history_length=(prefix_length - scene_length) // 2,
        history_token_length=prefix_length - scene_length,
    )


def _append_prefix(
    encoder: torch.nn.Module,
    cache: TransformerKVCache,
    prefix_tokens: torch.Tensor,
    prefix_valid: torch.Tensor,
    prefix_position_ids: torch.Tensor,
    *,
    attention_residual=None,
    rotary_position_encoding,
) -> TransformerKVCache:
    cached_length = cache.prefix_tokens.shape[1]
    new_tokens = prefix_tokens[:, cached_length:]
    new_valid = prefix_valid[:, cached_length:]
    new_position_ids = prefix_position_ids[:, cached_length:]
    if new_tokens.shape[1] == 0:
        return cache

    working_valid = torch.cat((cache.prefix_valid, new_valid), dim=1)
    hidden = new_tokens
    sources = [new_tokens] if attention_residual is not None else None
    key_cache = list(cache.key_cache)
    value_cache = list(cache.value_cache)

    for layer_index, layer in enumerate(encoder.layers):
        hidden, new_key, new_value, _ = run_causal_layer(
            layer,
            hidden,
            key_valid=working_valid,
            position_ids=new_position_ids,
            rotary_position_encoding=rotary_position_encoding,
            existing_key=key_cache[layer_index],
            existing_value=value_cache[layer_index],
            existing_length=cached_length,
            attention_residual=attention_residual,
            sources=sources,
            query_index=2 * layer_index,
        )
        key_cache[layer_index] = torch.cat(
            (key_cache[layer_index], new_key),
            dim=2,
        )
        value_cache[layer_index] = torch.cat(
            (value_cache[layer_index], new_value),
            dim=2,
        )

    return TransformerKVCache(
        prefix_tokens=prefix_tokens.detach().clone(),
        prefix_valid=prefix_valid.detach().clone(),
        prefix_position_ids=prefix_position_ids.detach().clone(),
        key_cache=tuple(key_cache),
        value_cache=tuple(value_cache),
        scene_length=cache.scene_length,
        history_length=(prefix_tokens.shape[1] - cache.scene_length) // 2,
        history_token_length=prefix_tokens.shape[1] - cache.scene_length,
    )
