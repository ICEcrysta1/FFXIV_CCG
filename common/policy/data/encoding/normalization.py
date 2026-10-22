"""策略输入归一化配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.configuration.yaml import load_yaml_mapping
from common.pytorch.dependencies import import_torch

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[4] / "config" / "policy" / "normalizer.yaml"
)

# 固定字段的单位由此声明；职业资源和 Buff 层数使用各自注册的单位与上限。
# 名称中碰巧包含 seconds/ready/timer 的其他字段不会自动获得对应语义。
FIELD_UNITS = {
    "available": "binary",
    "active": "binary",
    "is_legal": "binary",
    "boss_targetable": "binary",
    "is_moving": "binary",
    "kind": "identity",
    "mp": "mp",
    "max_mp": "mp",
    "actual_mp_cost": "mp_cost",
    "mp_cost": "mp_cost",
    "time_seconds": "time_anchor",
    "current_gcd_seconds": "seconds",
    "next_untargetable_in_seconds": "seconds",
    "downtime_remaining_seconds": "seconds",
    "remaining_seconds": "seconds",
    "next_cooldown_seconds": "seconds",
    "cast_time.seconds": "seconds",
    "gcd_window.seconds": "seconds",
    "start_offset_seconds": "time_anchor",
    "end_offset_seconds": "time_anchor",
    "duration_seconds": "seconds",
    "potency": "potency",
    "current_potency": "potency",
    "current_gcd_dot_potency": "potency",
    "target_count": "target_count",
    "max_charges": "charges",
}


def _fp32_limits():
    """直接读取计算 dtype 的范围，避免手写 IEEE 浮点边界。"""
    torch = import_torch()
    return torch.finfo(torch.float32)


@dataclass(frozen=True)
class NormalizerConfig:
    """策略输入的公共数值尺度；职业资源和层数上限由职业 YAML 提供。"""

    mp_max: float = 10000.0
    time_scale: float = 270.0
    duration_scale: float = 270.0
    target_count_max: float = 5.0
    current_potency_max: float = 2000.0

    def __post_init__(self) -> None:
        fp32 = _fp32_limits()
        for name in self.__dataclass_fields__:
            raw = getattr(self, name)
            if isinstance(raw, bool):
                raise TypeError(f"normalizer.{name} must be a finite positive number")
            try:
                value = float(raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"normalizer.{name} must be a finite positive number") from exc
            if not fp32.tiny <= value <= fp32.max:
                raise ValueError(f"normalizer.{name} must be within the FP32 normal positive range")
            object.__setattr__(self, name, value)


def load_normalizer_config(path: Path | None = None) -> NormalizerConfig:
    """读取策略输入归一化配置，不依赖预训练配置模块。"""
    config_path = path or DEFAULT_CONFIG_PATH
    raw = load_yaml_mapping(config_path, description="policy normalization config")
    values = raw.get("normalizer", {}) or {}
    if not isinstance(values, dict):
        raise TypeError("normalizer config section must be a mapping")
    unknown = values.keys() - NormalizerConfig.__dataclass_fields__.keys()
    if unknown:
        raise ValueError("unknown or removed normalizer options: " + ", ".join(sorted(unknown)))
    return NormalizerConfig(**values)
