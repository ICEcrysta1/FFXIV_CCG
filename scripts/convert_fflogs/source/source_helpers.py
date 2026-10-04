"""raw 转换阶段的技能特征与列式数据辅助。"""

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


def extract_mapping_row(node: dict[str, object], indices: tuple[int, ...]) -> dict[str, object]:
    row: dict[str, object] = {}
    for key, value in node.items():
        row[key] = _extract_mapping_value(value, indices)
    return row


def _extract_mapping_value(node: object, indices: tuple[int, ...]):
    """从列式节点读取一行，并处理按样本嵌套的映射结构。"""
    if is_nullable_tensor_node(node):
        return extract_nullable_scalar(node, indices)
    if isinstance(node, dict):
        return extract_mapping_row(node, indices)

    # 先取样本映射，再将剩余索引用于其中字段，避免对中间 dict 使用整数下标。
    if isinstance(node, (list, tuple)) and indices:
        sample_index, *remaining_indices = indices
        try:
            selected = node[sample_index]
        except IndexError:
            return None
        if isinstance(selected, dict):
            return _extract_mapping_value(selected, tuple(remaining_indices))

    return extract_scalar(node, indices)


def extract_scalar(node: object, indices: tuple[int, ...]):
    if node is None:
        return None
    result = node
    try:
        for index in indices:
            result = result[index]
    except (IndexError, KeyError, TypeError):
        return None
    if hasattr(result, "item"):
        try:
            return result.item()
        except ValueError:
            return result
    return result


def extract_nullable_scalar(node: dict[str, object], indices: tuple[int, ...]):
    result = extract_scalar(node.get("values"), indices)
    if result is None:
        return None
    if bool(extract_scalar(node.get("is_null"), indices)):
        return None
    return result


def extract_tensor_values(node: object):
    if node is None or (isinstance(node, (list, tuple)) and not node):
        return None
    return node["values"] if is_nullable_tensor_node(node) else node


def slice_nullable_tensor(
    node: object,
    *,
    prefix_indices: tuple[int, ...] | None,
    offset: int,
    length: int,
    width: int,
    torch,
    dtype,
    clone: bool = True,
):
    prefix_indices = prefix_indices or ()
    if node is None or (isinstance(node, (list, tuple)) and not node):
        return (
            torch.zeros((length, width), dtype=dtype),
            torch.ones((length, width), dtype=torch.bool),
        )

    values_node = node["values"] if is_nullable_tensor_node(node) else node
    null_node = node.get("is_null") if is_nullable_tensor_node(node) else None
    values = index_tensor_prefix(values_node, prefix_indices)
    if values is None:
        return (
            torch.zeros((length, width), dtype=dtype),
            torch.ones((length, width), dtype=torch.bool),
        )

    if values.ndim == 1:
        values = values.unsqueeze(0)
    values = values[offset : offset + length].to(dtype=dtype)
    if clone:
        values = values.clone()

    if null_node is None:
        null_mask = torch.zeros_like(values, dtype=torch.bool)
    else:
        null_mask = index_tensor_prefix(null_node, prefix_indices)
        if null_mask is None:
            null_mask = torch.zeros_like(values, dtype=torch.bool)
        else:
            if null_mask.ndim == 1:
                null_mask = null_mask.unsqueeze(0)
            null_mask = null_mask[offset : offset + length].to(dtype=torch.bool)
            if clone:
                null_mask = null_mask.clone()

    if values.shape[-1] != width:
        raise ValueError(f"state vector width mismatch: {values.shape[-1]} != expected {width}")
    return values, null_mask


def index_tensor_prefix(node: object, prefix_indices: tuple[int, ...]):
    if node is None:
        return None
    result = node
    try:
        for index in prefix_indices:
            result = result[index]
    except (IndexError, KeyError, TypeError):
        return None
    return result


def is_nullable_tensor_node(node: object) -> bool:
    return isinstance(node, dict) and set(node.keys()) == {"values", "is_null"}


def to_optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
