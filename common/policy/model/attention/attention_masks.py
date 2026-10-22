"""因果注意力的可见性 mask 与分析矩阵构建。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def build_allowed_mask(
    key_valid: torch.Tensor,
    *,
    query_count: int,
    key_count: int,
    causal: bool,
    causal_offset: int = 0,
    force_explicit_mask: bool = False,
    all_valid: bool | None = None,
) -> torch.Tensor | None:
    """构造 SDPA 的允许访问 mask；全有效标准路径返回 None。

    ``all_valid`` 允许调用方预先算好整段有效性，避免每层重复触发一次
    device 到 host 的同步。
    """
    # 强制显式 mask 时不能读取 device 上的取值：导出路径需要完全静态的控制流。
    if force_explicit_mask:
        all_valid = False
    elif all_valid is None:
        all_valid = bool(key_valid.all())
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


@dataclass(frozen=True)
class SegmentAttentionMask:
    """单段 attention 预先算好的 mask 与全屏蔽行处理。"""

    allowed: torch.Tensor | None
    attn_mask: torch.Tensor | None
    fully_blocked: torch.Tensor | None
    is_causal: bool


def build_segment_mask(
    key_valid: torch.Tensor,
    *,
    query_count: int,
    key_count: int,
    causal: bool,
    causal_offset: int = 0,
    force_explicit_mask: bool = False,
    all_valid: bool | None = None,
) -> SegmentAttentionMask:
    """构造单段 mask，并把全屏蔽行的安全列处理一并完成。

    结果与逐层现算完全一致，只是把构造时机提前到整个 encoder 之前。
    """
    allowed = build_allowed_mask(
        key_valid,
        query_count=query_count,
        key_count=key_count,
        causal=causal,
        causal_offset=causal_offset,
        force_explicit_mask=force_explicit_mask,
        all_valid=all_valid,
    )
    if allowed is None:
        return SegmentAttentionMask(None, None, None, causal)
    fully_blocked = ~allowed.any(dim=-1)
    return SegmentAttentionMask(
        allowed=allowed,
        attn_mask=allowed | fully_blocked.unsqueeze(-1),
        fully_blocked=fully_blocked,
        is_causal=False,
    )


def build_causal_attention_mask(*, token_count: int, device: torch.device) -> torch.Tensor:
    """生成 trace/可视化使用的严格因果禁止矩阵。"""
    if token_count < 0:
        raise ValueError("attention token length must be non-negative")
    return torch.triu(
        torch.ones((token_count, token_count), dtype=torch.bool, device=device),
        diagonal=1,
    )
