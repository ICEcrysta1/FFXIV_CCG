"""候选技能 canonical 顺序与样本重排工具。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from common.yaml_config import load_yaml_mapping


def load_candidate_order(
    path: Path,
    *,
    expected_action_keys: Sequence[str],
) -> tuple[str, ...]:
    """读取 1-based 技能顺序文件，并严格校验它覆盖完整候选集合。"""
    path = Path(path).resolve()
    payload = load_yaml_mapping(path, description="candidate order")
    raw_order = payload.get("candidate_order")
    if not isinstance(raw_order, Mapping):
        raise ValueError("candidate order file must define a candidate_order mapping")

    expected = tuple(str(key) for key in expected_action_keys)
    expected_set = set(expected)
    indexed: dict[int, str] = {}
    for raw_index, raw_key in raw_order.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "candidate_order keys must be consecutive 1-based integers"
            ) from exc
        if index in indexed:
            raise ValueError(f"candidate_order contains duplicate index: {index}")
        key = str(raw_key).strip()
        if not key:
            raise ValueError(f"candidate_order[{index}] must not be empty")
        indexed[index] = key

    if set(indexed) != set(range(1, len(expected) + 1)):
        raise ValueError(
            "candidate_order indices must be exactly 1.."
            f"{len(expected)}"
        )
    order = tuple(indexed[index] for index in range(1, len(expected) + 1))
    if len(set(order)) != len(order):
        raise ValueError("candidate_order must not contain duplicate skills")
    if set(order) != expected_set:
        missing = sorted(expected_set - set(order))
        unknown = sorted(set(order) - expected_set)
        details = []
        if missing:
            details.append("missing=" + ", ".join(missing))
        if unknown:
            details.append("unknown=" + ", ".join(unknown))
        raise ValueError(
            "candidate_order must match the compiled candidate set exactly; "
            "update candidate_order.yaml after changing the candidate set "
            "and re-stat the ranking when frequencies change: "
            + "; ".join(details)
        )
    return order


def candidate_permutation(
    source_action_keys: Sequence[str],
    target_action_keys: Sequence[str],
) -> tuple[int, ...]:
    """返回 target 顺序对应的 source 下标。"""
    source = tuple(str(key) for key in source_action_keys)
    target = tuple(str(key) for key in target_action_keys)
    source_set = set(source)
    target_set = set(target)
    if len(source_set) != len(source):
        raise ValueError("source candidate order contains duplicate skills")
    if len(target_set) != len(target):
        raise ValueError("target candidate order contains duplicate skills")
    if source_set != target_set:
        details = []
        missing = sorted(target_set - source_set)
        unknown = sorted(source_set - target_set)
        if missing:
            details.append("missing=" + ", ".join(missing))
        if unknown:
            details.append("unknown=" + ", ".join(unknown))
        raise ValueError(
            "source and target candidate orders differ: " + "; ".join(details)
        )
    source_indices = {key: index for index, key in enumerate(source)}
    return tuple(source_indices[key] for key in target)


def reorder_candidate_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    target_action_keys: Sequence[str],
) -> list[dict[str, object]]:
    """按 target 顺序重排 candidate skill/state rows。"""
    source_action_keys = [str(row.get("skill_key", "")) for row in rows]
    permutation = candidate_permutation(source_action_keys, target_action_keys)
    return [dict(rows[index]) for index in permutation]
