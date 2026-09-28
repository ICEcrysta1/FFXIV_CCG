"""行为克隆的主交叉熵损失。"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F


def primary_loss(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, object],
) -> torch.Tensor:
    """用 FP32 计算主交叉熵，保持原先 autocast 下的数值口径。"""
    return F.cross_entropy(output["logits"].float(), batch["label_index"])
