"""split attention 的可见性 mask 与分析矩阵构建。"""

from __future__ import annotations

import torch


def build_allowed_mask(
    key_valid: torch.Tensor,
    *,
    query_count: int,
    key_count: int,
    causal: bool,
    causal_offset: int = 0,
    force_explicit_mask: bool = False,
) -> torch.Tensor | None:
    """构造 SDPA 的允许访问 mask；全有效标准路径返回 None。"""
    all_valid = not force_explicit_mask and bool(key_valid.all())
    if all_valid and (not causal or causal_offset == 0):
        return None
    allowed = key_valid[:, None, None, :].expand(-1, 1, query_count, -1)
    if causal:
        query_positions = torch.arange(
            query_count,
            device=key_valid.device,
        ).view(1, 1, query_count, 1) + causal_offset
        key_positions = torch.arange(
            key_count,
            device=key_valid.device,
        ).view(1, 1, 1, key_count)
        allowed = allowed & (key_positions <= query_positions)
    return allowed


def assemble_attention(
    prefix_attention: torch.Tensor,
    candidate_attention: torch.Tensor,
    cls_attention: torch.Tensor,
    *,
    prefix_length: int,
    candidate_count: int,
) -> torch.Tensor:
    """把三次区域 attention 组装为分析用完整矩阵。"""
    total_length = prefix_length + candidate_count + 1
    attention = prefix_attention.new_zeros(
        prefix_attention.shape[0],
        prefix_attention.shape[1],
        total_length,
        total_length,
    )
    attention[:, :, :prefix_length, :prefix_length] = prefix_attention
    attention[
        :, :, prefix_length : prefix_length + candidate_count, : prefix_length + candidate_count
    ] = candidate_attention
    attention[:, :, -1:, :] = cls_attention
    return attention


def build_split_attention_mask(
    *,
    prefix_length: int,
    candidate_count: int,
    device: torch.device,
) -> torch.Tensor:
    """生成 trace/可视化使用的 split attention 禁止矩阵。"""
    if prefix_length < 0 or candidate_count < 0:
        raise ValueError("attention token lengths must be non-negative")
    total_length = prefix_length + candidate_count + 1
    allowed = torch.zeros(
        (total_length, total_length),
        dtype=torch.bool,
        device=device,
    )
    allowed[:prefix_length, :prefix_length] = torch.tril(
        torch.ones((prefix_length, prefix_length), dtype=torch.bool, device=device)
    )
    candidate_start = prefix_length
    candidate_end = prefix_length + candidate_count
    allowed[candidate_start:candidate_end, :candidate_end] = True
    allowed[-1, :] = True
    return ~allowed
