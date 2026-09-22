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


def extract_history_after_value(
    state_token: dict[str, object],
    schema,
    *,
    feature_name: str,
    context: str,
) -> float:
    """从状态历史 token 读取一个 after 数值，缺失时显式失败。"""
    group_key = "target_buff_state"
    qualified_name = (
        feature_name
        if feature_name.startswith("after.")
        else f"after.{feature_name}"
    )
    feature_keys = tuple(schema.state_group_feature_keys.get(group_key, ()))
    try:
        feature_index = feature_keys.index(qualified_name)
    except ValueError as exc:
        raise ValueError(
            f"{context} state history is missing feature {qualified_name!r}"
        ) from exc
    values = state_token.get(group_key)
    if not isinstance(values, (list, tuple)) or feature_index >= len(values):
        raise ValueError(
            f"{context} state history target_buff_state has no {qualified_name!r} value"
        )
    value = values[feature_index]
    if value is None:
        raise ValueError(f"{context} state history feature {qualified_name!r} is null")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context} state history feature {qualified_name!r} is not numeric"
        ) from exc


def derive_skill_feature_names(reader: "TrainingSourceReader") -> tuple[str, ...]:
    """从候选和历史技能行推导统一的数值特征名。"""
    seen: set[str] = set()
    sources = []
    candidate_rows = reader.candidate_skill_rows(0) if reader.num_candidates > 0 else []
    if candidate_rows:
        sources.append(candidate_rows[0])

    for sample_idx in range(reader.num_samples):
        history_rows = reader.history_skill_rows(sample_idx, max_history=1)
        if history_rows:
            sources.append(history_rows[0])
            break

    for row in sources:
        for feature_name in flatten_skill_numeric_features(row):
            seen.add(feature_name)
    return tuple(sorted(seen))


def build_skill_feature_matrix(rows: list[dict[str, object]], *, feature_names: tuple[str, ...], torch, dtype):
    if not feature_names:
        return torch.zeros((len(rows), 0), dtype=dtype)
    if not rows:
        return torch.zeros((0, len(feature_names)), dtype=dtype)

    # 一次性创建矩阵，避免每个技能行都单独创建一个临时 tensor 再复制到
    # 预分配矩阵。raw JSON 编译阶段这个函数会被每个样本的历史和候选各调用一次。
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
    """从列式节点读取一行，并处理候选动态字段的按样本嵌套结构。"""
    if is_nullable_tensor_node(node):
        return extract_nullable_scalar(node, indices)
    if isinstance(node, dict):
        return extract_mapping_row(node, indices)

    # candidate_skill_dynamic 的部分字段是
    # ``list[样本] -> {feature: {values: [候选], is_null: [候选]}}``。
    # 先取样本映射后，剩余索引才用于读取候选值；不能把 candidate_idx
    # 直接应用到这个中间 dict，否则会静默退化为 None，再被特征矩阵填成 0。
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
