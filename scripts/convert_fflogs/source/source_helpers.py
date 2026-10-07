"""raw 转换阶段的技能特征与真实执行指标读取。"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from common.numeric import flatten_numeric_mapping

if TYPE_CHECKING:
    from .source_reader import TrainingSourceReader


TEXT_SKILL_FIELDS = frozenset({"skill_key", "skill_name", "invalid_reason"})
SKILL_ID_FIELD = "skill_id"
TRAINING_ONLY_SKILL_FIELDS = frozenset({"value"})
SKILL_HISTORY_FIELDS = tuple(sorted((
    "skill_id", "skill_key", "skill_name", "potency", "value", "kind",
    "actual_mp_cost", "cast_time", "gcd_window", "is_legal", "invalid_reason",
    "next_cooldown_seconds", "available_charges", "max_charges",
    "job_resources_consumed",
)))
SKILL_NUMERIC_FEATURES = (
    "actual_mp_cost", "available_charges", "cast_time.seconds", "gcd_window.seconds",
    "is_legal", "kind", "max_charges", "next_cooldown_seconds", "potency",
)


def require_numeric_skill_kind(row: dict[str, object], *, context: str) -> float:
    """读取 token 中的数值 kind；1 表示 GCD，0 表示 oGCD。"""
    value = row.get("kind")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} skill kind must be numeric 0 or 1")
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{context} skill kind must be numeric 0 or 1") from exc
    if not math.isfinite(numeric_value) or numeric_value not in (0.0, 1.0):
        raise ValueError(
            f"{context} skill kind must be numeric 0 or 1, got {value!r}"
        )
    return numeric_value


def extract_execution_metric(
    metrics: dict[str, object],
    *,
    feature_name: str,
    context: str,
) -> float:
    """从真实执行 metadata 读取原始指标，禁止改从模型状态两段推导结果。"""
    value = metrics.get(feature_name) if isinstance(metrics, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{context} execution metric {feature_name!r} must be finite numeric")
    return float(value)


def derive_skill_feature_names(reader: "TrainingSourceReader") -> tuple[str, ...]:
    """由稳定技能字段和职业资源定义推导，空历史也有同一宽度。"""
    from common.config import load_project_config

    config = load_project_config(job_tag=reader.job_tag)
    resource_names = tuple(
        f"job_resources_consumed.{key}" for key in config.job.resource_limits
    )
    return tuple(sorted((*SKILL_NUMERIC_FEATURES, *resource_names)))


def build_skill_feature_matrix(rows: list[dict[str, object]], *, feature_names: tuple[str, ...], torch, dtype):
    if not feature_names:
        return torch.zeros((len(rows), 0), dtype=dtype)
    if not rows:
        return torch.zeros((0, len(feature_names)), dtype=dtype)

    # 一次性创建矩阵，避免每个技能行都单独创建一个临时 tensor 再复制到
    # 预分配矩阵。raw JSON 编译阶段只为已执行历史构建技能特征。
    numeric_rows = []
    for row in rows:
        numeric_features = flatten_skill_numeric_features(row)
        numeric_rows.append(
            [
                float(numeric_features.get(feature_name, 0.0))
                for feature_name in feature_names
            ]
        )
    return torch.tensor(numeric_rows, dtype=dtype)


def flatten_skill_numeric_features(row: dict[str, object]) -> dict[str, float]:
    if "time_seconds" in row:
        raise ValueError("skill token must not include time_seconds; recompile raw source")
    flattened = flatten_numeric_mapping(row, ignored_keys=TEXT_SKILL_FIELDS)
    flattened.pop(SKILL_ID_FIELD, None)
    for field_name in TRAINING_ONLY_SKILL_FIELDS:
        flattened.pop(field_name, None)
    return flattened


def to_optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
