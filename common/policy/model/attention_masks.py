"""split attention 的可见性 mask 与分析矩阵构建。"""

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


def build_split_segment_masks(
    prefix_valid: torch.Tensor,
    candidate_valid: torch.Tensor,
    cls_valid: torch.Tensor,
    *,
    prefix_length: int,
    candidate_count: int,
    cls_count: int,
    force_explicit_mask: bool = False,
) -> tuple[SegmentAttentionMask, SegmentAttentionMask, SegmentAttentionMask]:
    """一次算好 prefix / candidate / CLS 三段 mask，供所有层复用。

    三段的有效性只取决于输入 batch，与层无关，因此这里只做一次 host 同步：
    三段 key 的“全有效”标志由同一个 stack 结果读出。
    """
    if force_explicit_mask:
        prefix_all_valid = candidate_block_valid = cls_block_valid = False
    else:
        prefix_all_valid, candidate_block_valid, cls_block_valid = (
            torch.stack(
                (
                    prefix_valid.all(),
                    candidate_valid.all(),
                    cls_valid.all(),
                )
            )
            .tolist()
        )
    prefix_mask = build_segment_mask(
        prefix_valid,
        query_count=prefix_length,
        key_count=prefix_length,
        causal=True,
        force_explicit_mask=force_explicit_mask,
        all_valid=prefix_all_valid,
    )
    candidate_mask = build_segment_mask(
        torch.cat((prefix_valid, candidate_valid), dim=1),
        query_count=candidate_count,
        key_count=prefix_length + candidate_count,
        causal=False,
        force_explicit_mask=force_explicit_mask,
        all_valid=prefix_all_valid and candidate_block_valid,
    )
    cls_mask = build_segment_mask(
        torch.cat((prefix_valid, candidate_valid, cls_valid), dim=1),
        query_count=cls_count,
        key_count=prefix_length + candidate_count + cls_count,
        causal=False,
        force_explicit_mask=force_explicit_mask,
        all_valid=prefix_all_valid and candidate_block_valid and cls_block_valid,
    )
    return prefix_mask, candidate_mask, cls_mask


def build_cached_segment_masks(
    prefix_valid: torch.Tensor,
    candidate_valid: torch.Tensor,
    cls_valid: torch.Tensor,
    *,
    candidate_count: int,
    cls_count: int,
    force_explicit_mask: bool = False,
) -> tuple[SegmentAttentionMask, SegmentAttentionMask]:
    """KV-cache 解码用的候选 / CLS 两段 mask，一次算好供所有层复用。"""
    prefix_length = prefix_valid.shape[1]
    if force_explicit_mask:
        # 导出/显式 mask 路径不允许读取 device 取值，控制流必须保持静态。
        prefix_all_valid = candidate_all_valid = cls_all_valid = False
    else:
        prefix_all_valid, candidate_all_valid, cls_all_valid = torch.stack(
            (
                prefix_valid.all(),
                candidate_valid.all(),
                cls_valid.all(),
            )
        ).tolist()
    candidate_mask = build_segment_mask(
        torch.cat((prefix_valid, candidate_valid), dim=1),
        query_count=candidate_count,
        key_count=prefix_length + candidate_count,
        causal=False,
        force_explicit_mask=force_explicit_mask,
        all_valid=prefix_all_valid and candidate_all_valid,
    )
    cls_mask = build_segment_mask(
        torch.cat((prefix_valid, candidate_valid, cls_valid), dim=1),
        query_count=cls_count,
        key_count=prefix_length + candidate_count + cls_count,
        causal=False,
        force_explicit_mask=force_explicit_mask,
        all_valid=prefix_all_valid and candidate_all_valid and cls_all_valid,
    )
    return candidate_mask, cls_mask


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
