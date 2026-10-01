"""动作质量标签到逐样本正向学习权重的映射。"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from ...config import ActionQualityLossConfig


def action_quality_sample_weights(
    batch: Mapping[str, object],
    config: ActionQualityLossConfig,
) -> torch.Tensor:
    """消费 collator 已校验的监督字段；仅削弱被明确归因的 target。"""
    labels = batch["label_index"]
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
        raise ValueError("label_index must be a [batch] tensor")
    weights = torch.ones(labels.shape[0], device=labels.device, dtype=torch.float32)
    if not config.enabled:
        return weights

    levels = batch.get("quality_label_levels")
    mask = batch.get("quality_label_mask")
    quality = batch.get("source_quality")
    available = batch.get("quality_annotation_available")
    if not all(isinstance(value, torch.Tensor) for value in (levels, mask, quality, available)):
        raise ValueError("enabled action quality loss requires compiled quality supervision")
    if levels.ndim != 2 or levels.shape[0] != labels.shape[0] or mask.shape != levels.shape:
        raise ValueError("quality_label_levels and quality_label_mask must be [batch, labels]")
    if quality.shape != labels.shape or available.shape != labels.shape:
        raise ValueError("source_quality and quality_annotation_available must be [batch]")
    if any(value.device != labels.device for value in (levels, mask, quality, available)):
        raise ValueError("quality supervision must be on the same device as label_index")

    if levels.shape[1] == 0:
        return weights
    tagged = mask.any(dim=1)
    # 无标签样本的排名可以缺失或非有限，先隔离它们，避免零严重度乘出 NaN。
    normalized_quality = quality.float().masked_fill(~tagged, 0.0)
    severity_lookup = torch.tensor(
        (0.0, *config.severity_weights), device=labels.device, dtype=torch.float32
    )
    severity = severity_lookup[levels.masked_fill(~mask, 0).long()].max(dim=1).values
    exponent = torch.pow(
        normalized_quality.clamp_min(0.0) / config.scale, config.exponent
    )
    # (1 - s) + s * (1 - exp(-z)) 避免低排名时 1 - exp(-z) 的相消误差。
    return (weights - severity) + severity * (-torch.expm1(-exponent))
