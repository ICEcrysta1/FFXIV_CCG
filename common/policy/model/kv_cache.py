"""推理专用的场景/历史 prefix KV-cache。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .attention_masks import build_cached_segment_masks
from .split_encoder import (
    run_cached_candidate_layer,
    run_prefix_layer,
)


@dataclass
class TransformerKVCache:
    """缓存稳定的 scene/history prefix 在每一层的 K/V。"""

    prefix_tokens: torch.Tensor
    prefix_valid: torch.Tensor
    prefix_position_ids: torch.Tensor
    key_cache: tuple[torch.Tensor, ...]
    value_cache: tuple[torch.Tensor, ...]
    scene_length: int
    history_length: int


def encode_with_kv_cache(
    encoder: torch.nn.Module,
    encoded: dict[str, torch.Tensor | int],
    cache: TransformerKVCache | None = None,
) -> tuple[torch.Tensor, TransformerKVCache]:
    """使用 prefix cache 编码当前候选集合。

    scene/history prefix 逐层保持因果并缓存；候选在每次决策中只保留一份，
    以非因果 SDPA 同时读取完整 prefix K/V 与全部候选 K/V。CLS 单独读取
    prefix、候选和自身，候选不会读取 CLS。
    """

    tokens = encoded["tokens"]
    prefix_length = int(encoded["prefix_length"])
    candidate_count = int(encoded["candidate_count"])
    scene_length = int(encoded["scene_length"])

    assert isinstance(tokens, torch.Tensor)
    prefix_tokens = tokens[:, :prefix_length]
    prefix_valid = encoded["prefix_valid"]
    position_ids = encoded["position_ids"]
    candidate_valid = encoded["candidate_valid"]
    cls_valid = encoded["cls_valid"]
    assert isinstance(prefix_valid, torch.Tensor)
    assert isinstance(position_ids, torch.Tensor)
    assert isinstance(candidate_valid, torch.Tensor)
    assert isinstance(cls_valid, torch.Tensor)

    candidate_tokens = tokens[:, prefix_length : prefix_length + candidate_count]
    cls_tokens = tokens[:, prefix_length + candidate_count :]
    attention_residual = getattr(encoder, "attention_residual", None)

    prefix_position_ids = position_ids[:, :prefix_length]
    candidate_position_ids = position_ids[:, prefix_length : prefix_length + candidate_count]
    cls_position_ids = position_ids[:, prefix_length + candidate_count :]
    rotary_position_encoding = getattr(encoder, "rotary_position_encoding", None)
    if rotary_position_encoding is None:
        raise ValueError("encoder is missing the required RotaryPositionEncoding module")

    if not _cache_matches(
        cache,
        prefix_tokens,
        prefix_valid,
        prefix_position_ids,
        scene_length,
    ):
        cache = _build_prefix_cache(
            encoder,
            prefix_tokens,
            prefix_valid,
            prefix_position_ids,
            scene_length,
            attention_residual=attention_residual,
            rotary_position_encoding=rotary_position_encoding,
        )
    elif prefix_length > cache.prefix_tokens.shape[1]:
        cache = _append_prefix(
            encoder,
            cache,
            prefix_tokens,
            prefix_valid,
            prefix_position_ids,
            attention_residual=attention_residual,
            rotary_position_encoding=rotary_position_encoding,
        )

    candidate_hidden = candidate_tokens
    cls_hidden = cls_tokens
    candidate_sources = [candidate_tokens] if attention_residual is not None else None
    cls_sources = [cls_tokens] if attention_residual is not None else None
    # 候选 / CLS 两段 mask 只取决于本步的有效性布局，与层无关：一次算好复用，
    # 避免每层重复构造并触发 device 到 host 的同步。
    segment_masks = build_cached_segment_masks(
        cache.prefix_valid,
        candidate_valid,
        cls_valid,
        candidate_count=candidate_count,
        cls_count=cls_tokens.shape[1],
    )
    for layer_index, layer in enumerate(encoder.layers):
        candidate_hidden, cls_hidden = run_cached_candidate_layer(
            layer,
            candidate_hidden,
            cls_hidden,
            prefix_key=cache.key_cache[layer_index],
            prefix_value=cache.value_cache[layer_index],
            prefix_valid=cache.prefix_valid,
            prefix_position_ids=cache.prefix_position_ids,
            candidate_position_ids=candidate_position_ids,
            cls_position_ids=cls_position_ids,
            candidate_valid=candidate_valid,
            cls_valid=cls_valid,
            rotary_position_encoding=rotary_position_encoding,
            attention_residual=attention_residual,
            candidate_sources=candidate_sources,
            cls_sources=cls_sources,
            query_index=2 * layer_index,
            segment_masks=segment_masks,
        )

    if encoder.norm is not None:
        candidate_hidden = encoder.norm(candidate_hidden)
        cls_hidden = encoder.norm(cls_hidden)

    return torch.cat((candidate_hidden, cls_hidden), dim=1), cache


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

        hidden, key, value = run_prefix_layer(
            layer,
            hidden,
            prefix_valid=prefix_valid,
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
        history_length=prefix_length - scene_length,
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
        hidden, new_key, new_value = run_prefix_layer(
            layer,
            hidden,
            prefix_valid=working_valid,
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
        history_length=prefix_tokens.shape[1] - cache.scene_length,
    )
