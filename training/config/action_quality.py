"""动作质量监督的训练配置。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ActionQualityLossConfig:
    """按日志排名衰减错误动作的正向学习权重。"""

    enabled: bool = False
    scale: float = 0.6
    exponent: float = 4.0
    severity_weights: tuple[float, float, float] = (0.25, 0.5, 1.0)


def parse_action_quality_loss_config(
    training: Mapping[str, object],
    quality: object,
) -> ActionQualityLossConfig:
    """从同一份已合并模型配置读取曲线与职业等级映射。"""
    raw = training.get("action_quality_loss")
    if raw is None:
        return ActionQualityLossConfig()
    if not isinstance(raw, Mapping):
        raise TypeError("training.action_quality_loss must be a mapping")
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise TypeError("training.action_quality_loss.enabled must be boolean")
    if not enabled:
        return ActionQualityLossConfig()

    decay = raw.get("percentile_decay")
    if not isinstance(decay, Mapping):
        raise TypeError("training.action_quality_loss.percentile_decay must be a mapping")
    if decay.get("function") != "exp_negative_power":
        raise ValueError("training.action_quality_loss.percentile_decay.function must be exp_negative_power")
    scale = _positive_number(decay.get("scale"), "training.action_quality_loss.percentile_decay.scale")
    exponent = _positive_number(
        decay.get("exponent"), "training.action_quality_loss.percentile_decay.exponent"
    )

    if not isinstance(quality, Mapping):
        raise TypeError("action_quality must be a mapping when action quality loss is enabled")
    weights = quality.get("severity_weights")
    if not isinstance(weights, Mapping):
        raise TypeError("action_quality.severity_weights must be a mapping")
    severity_weights = tuple(
        _severity_weight(weights.get(level), f"action_quality.severity_weights.{level}")
        for level in ("minor", "medium", "major")
    )
    return ActionQualityLossConfig(
        enabled=True,
        scale=scale,
        exponent=exponent,
        severity_weights=severity_weights,
    )


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return number


def _severity_weight(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return number
