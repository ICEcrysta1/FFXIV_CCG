"""策略输入归一化配置。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from common.configuration.yaml import load_yaml_mapping

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "policy" / "normalizer.yaml"
)


@dataclass(frozen=True)
class NormalizerConfig:
    """策略输入的公共数值尺度；职业资源和层数上限由职业 YAML 提供。"""

    mp_max: float = 10000.0
    time_scale: float = 270.0
    target_count_max: float = 5.0
    current_potency_max: float = 2000.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            raw = getattr(self, name)
            if isinstance(raw, bool):
                raise TypeError(f"normalizer.{name} must be a finite positive number")
            try:
                value = float(raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"normalizer.{name} must be a finite positive number") from exc
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"normalizer.{name} must be a finite positive number")
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
