"""策略输入归一化配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.yaml_config import load_yaml_mapping


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "policy" / "normalizer.yaml"
)


@dataclass(frozen=True)
class NormalizerConfig:
    """策略输入归一化上限与累计威力变换。"""

    mp_max: float = 10000.0
    remaining_seconds_max: float = 120.0
    fight_time_max: float = 1800.0
    target_count_max: float = 3.0
    remaining_gcds_max: float = 48.0
    current_potency_max: float = 2500.0
    cumulative_potency_mode: str = "log1p"


def load_normalizer_config(path: Path | None = None) -> NormalizerConfig:
    """读取策略输入归一化配置，不依赖预训练配置模块。"""
    config_path = path or DEFAULT_CONFIG_PATH
    raw = load_yaml_mapping(config_path, description="policy normalization config")
    values = raw.get("normalizer", {}) or {}
    if not isinstance(values, dict):
        raise ValueError("normalizer config section must be a mapping")
    return NormalizerConfig(
        mp_max=float(values.get("mp_max", NormalizerConfig.mp_max)),
        remaining_seconds_max=float(
            values.get("remaining_seconds_max", NormalizerConfig.remaining_seconds_max)
        ),
        fight_time_max=float(values.get("fight_time_max", NormalizerConfig.fight_time_max)),
        target_count_max=float(values.get("target_count_max", NormalizerConfig.target_count_max)),
        remaining_gcds_max=float(
            values.get("remaining_gcds_max", NormalizerConfig.remaining_gcds_max)
        ),
        current_potency_max=float(
            values.get("current_potency_max", NormalizerConfig.current_potency_max)
        ),
        cumulative_potency_mode=str(
            values.get("cumulative_potency_mode", NormalizerConfig.cumulative_potency_mode)
        ),
    )
