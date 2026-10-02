"""BC 与 GRPO 共用的优化器配置校验，两个阶段独立选择算法和超参数。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class OptimizerConfig:
    """选择优化器，并显式记录 Muon 的更新规则。"""

    name: str = "adamw"
    momentum: float = 0.95
    nesterov: bool = True
    ns_steps: int = 5
    adjust_lr_fn: str = "match_rms_adamw"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or self.name not in {"adamw", "muon"}:
            raise ValueError("optimizer.name must be adamw or muon")
        if isinstance(self.momentum, bool) or not isinstance(self.momentum, (int, float)):
            raise ValueError("optimizer.momentum must be numeric")
        # 先检查区间，避免对任意大小的整数做浮点转换时溢出。
        if not 0.0 <= self.momentum < 1.0 or not math.isfinite(self.momentum):
            raise ValueError("optimizer.momentum must be finite and within [0, 1)")
        if not isinstance(self.nesterov, bool):
            raise ValueError("optimizer.nesterov must be boolean")
        if isinstance(self.ns_steps, bool) or not isinstance(self.ns_steps, int):
            raise ValueError("optimizer.ns_steps must be an integer within [1, 99]")
        if not 1 <= self.ns_steps <= 99:
            raise ValueError("optimizer.ns_steps must be an integer within [1, 99]")
        if not isinstance(self.adjust_lr_fn, str) or self.adjust_lr_fn not in {
            "original",
            "match_rms_adamw",
        }:
            raise ValueError(
                "optimizer.adjust_lr_fn must be original or match_rms_adamw"
            )

    @classmethod
    def from_mapping(cls, raw: object) -> OptimizerConfig:
        """严格读取算法配置，避免拼错字段后静默使用默认值。"""
        if not isinstance(raw, Mapping):
            raise ValueError("optimizer must be a mapping")
        allowed = {item.name for item in fields(cls)}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(
                "optimizer contains unknown fields: "
                + ", ".join(sorted(str(key) for key in unknown))
            )
        return cls(**raw)


def parse_optimizer_settings(
    policy_config: Mapping[str, object],
    *,
    stage: str,
    legacy: Mapping[str, object],
    default_learning_rate: float,
    default_weight_decay: float,
    default_warmup_steps: int,
) -> dict[str, object]:
    """读取阶段专属配置，兼容旧单文件并拒绝同阶段的两份配置权威。"""
    if stage not in {"bc", "grpo"}:
        raise ValueError("optimizer stage must be bc or grpo")
    optimizers = policy_config.get("optimizers", {})
    if not isinstance(optimizers, Mapping):
        raise ValueError("optimizers must be a mapping")
    unknown_stages = set(optimizers) - {"bc", "grpo"}
    if unknown_stages:
        raise ValueError(
            "optimizers contains unknown stages: "
            + ", ".join(sorted(str(key) for key in unknown_stages))
        )
    shared_fields = {"learning_rate", "weight_decay", "warmup_steps"}
    if stage in optimizers:
        values = optimizers[stage]
        if not isinstance(values, Mapping):
            raise ValueError(f"optimizers.{stage} must be a mapping")
        duplicate = set(legacy) & (shared_fields | {"optimizer"})
        if duplicate:
            legacy_section = "training" if stage == "bc" else "grpo"
            raise ValueError(
                f"optimizers.{stage} conflicts with {legacy_section} optimizer settings: "
                + ", ".join(sorted(duplicate))
            )
        algorithm_values = {key: value for key, value in values.items() if key not in shared_fields}
        label = f"optimizers.{stage}"
    else:
        values = legacy
        algorithm_values = values.get("optimizer", {})
        label = "training" if stage == "bc" else "grpo"
    optimizer = OptimizerConfig.from_mapping(algorithm_values)
    learning_rate = _finite_number(
        values.get("learning_rate", default_learning_rate), f"{label}.learning_rate"
    )
    weight_decay = _finite_number(
        values.get("weight_decay", default_weight_decay), f"{label}.weight_decay"
    )
    warmup_steps = values.get("warmup_steps", default_warmup_steps)
    if learning_rate <= 0.0:
        raise ValueError(f"{label}.learning_rate must be > 0")
    if weight_decay < 0.0:
        raise ValueError(f"{label}.weight_decay must be >= 0")
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise ValueError(f"{label}.warmup_steps must be an integer >= 0")
    return {
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
    }


def _finite_number(value: object, label: str) -> float:
    """严格校验学习率与权重衰减，拒绝布尔值和非有限数值。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric and finite")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be numeric and finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be numeric and finite")
    return number
