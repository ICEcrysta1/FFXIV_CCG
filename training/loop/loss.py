"""行为克隆训练损失的统一装配入口。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch

from training.config import ValuePreferenceConfig

from .losses.primary import primary_loss
from .losses.value_preference import compute_value_preference_loss


@dataclass(frozen=True)
class AuxiliaryLoss:
    """一个可插拔的训练辅助项；weight 为零时不执行计算。"""

    metric_name: str
    weight: float
    compute: Callable[[Mapping[str, torch.Tensor], Mapping[str, object]], torch.Tensor]


@dataclass(frozen=True)
class LossBreakdown:
    """保留原始辅助损失，便于指标记录；total 才参与反向传播。"""

    total: torch.Tensor
    primary: torch.Tensor
    auxiliary: dict[str, torch.Tensor]


def configured_auxiliary_losses(
    *, value_preference: ValuePreferenceConfig | None = None,
) -> tuple[AuxiliaryLoss, ...]:
    """集中注册可选辅助项；关闭的项不参与计算。"""
    config = value_preference or ValuePreferenceConfig()
    return (
        AuxiliaryLoss(
            metric_name="value_preference_loss",
            weight=config.loss_weight if config.enabled else 0.0,
            compute=lambda output, batch: compute_value_preference_loss(
                output["logits"], batch, config,
            ),
        ),
    )


def compose_training_loss(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, object],
    auxiliary_losses: Sequence[AuxiliaryLoss] = (),
) -> LossBreakdown:
    """主交叉熵固定存在；辅助项仅由显式配置列表决定。"""
    primary = primary_loss(output, batch)
    total = primary
    auxiliary: dict[str, torch.Tensor] = {}
    for term in auxiliary_losses:
        if term.metric_name in auxiliary:
            raise ValueError(f"duplicate auxiliary loss metric: {term.metric_name}")
        if term.weight < 0:
            raise ValueError(
                f"auxiliary loss weight must be non-negative: {term.metric_name}"
            )
        if term.weight > 0:
            value = term.compute(output, batch)
            total = total + term.weight * value
        else:
            value = primary.new_zeros(())
        auxiliary[term.metric_name] = value
    return LossBreakdown(total=total, primary=primary, auxiliary=auxiliary)
