"""行为克隆的主交叉熵损失。"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F


def primary_loss(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, object],
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """用 FP32 计算逐样本 CE，再按原始 batch 大小求均值。"""
    losses = F.cross_entropy(
        output["logits"].float(), batch["label_index"], reduction="none"
    )
    if sample_weights is not None:
        if sample_weights.shape != losses.shape:
            raise ValueError("primary sample weights must be [batch]")
        losses = losses * sample_weights.float()
    return losses.mean()
