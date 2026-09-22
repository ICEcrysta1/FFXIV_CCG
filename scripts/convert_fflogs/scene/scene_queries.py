"""绝对时间 scene window 的运行时查询。

场景标量（移动、停手）由 `scripts.common.scene_state` 承担；本模块只保留
样本装配需要的窗口查询：完整 scene_context 视图、人工排查锚点与停手总时长。
"""

from __future__ import annotations

from common.scene_context_schema import (
    FORCED_MOVEMENT_CONTEXT_KEY,
    RAID_BUFF_WINDOW_CONTEXT_KEY,
    SCENE_CONTEXT_KEYS,
    TARGETABLE_WINDOW_CONTEXT_KEY,
)
from common.scene_window import (
    feature_index_map,
)

from ..config.constants import SCENE_EPSILON, _round_time
from .scene_builders import normalize_scene_context


def build_scene_context_view(scene_context: dict[str, object]) -> dict[str, object]:
    """返回完整的绝对时间 scene_context。"""
    normalized = normalize_scene_context(scene_context)
    return {context_key: normalized[context_key] for context_key in SCENE_CONTEXT_KEYS}


def resolve_anchor(time_offset: float, scene_context: dict[str, object]) -> str:
    """为动作挑一个便于人工排查的锚点。"""
    targetable_window_context = scene_context[TARGETABLE_WINDOW_CONTEXT_KEY]
    downtime_index = 0
    for token in targetable_window_context["tokens"]:
        is_targetable = _token_flag(targetable_window_context, token, "targetable")
        if not is_targetable:
            downtime_index += 1
        if not _token_contains(targetable_window_context, token, time_offset):
            continue
        if is_targetable:
            break
        return f"downtime_{downtime_index}"

    forced_movement_context = scene_context[FORCED_MOVEMENT_CONTEXT_KEY]
    forced_movement_index = _matching_token_index(forced_movement_context, time_offset)
    if forced_movement_index is not None:
        return f"forced_movement_{forced_movement_index}"

    raid_buff_window_context = scene_context[RAID_BUFF_WINDOW_CONTEXT_KEY]
    raid_buff_index = _matching_token_index(raid_buff_window_context, time_offset)
    if raid_buff_index is not None:
        return f"raid_buff_window_{raid_buff_index}"

    if targetable_window_context["tokens"]:
        last_token = targetable_window_context["tokens"][-1]
        if _token_contains(targetable_window_context, last_token, time_offset):
            if _token_flag(targetable_window_context, last_token, "segment_kind.combat_final"):
                return "combat_final"
    return "combat"


def compute_downtime_total(targetable_window_context: dict[str, object]) -> float:
    """统计所有 downtime token 的总时长。"""
    total = 0.0
    for token in targetable_window_context["tokens"]:
        if not _token_flag(targetable_window_context, token, "targetable"):
            total += _token_value(targetable_window_context, token, "duration_seconds")
    return _round_time(total)


def _matching_token_index(window_context: dict[str, object], time_offset: float) -> int | None:
    for index, token in enumerate(window_context["tokens"], 1):
        if _token_contains(window_context, token, time_offset):
            return index
    return None


def _token_contains(window_context: dict[str, object], token: list[float], time_offset: float) -> bool:
    start, end = _token_bounds(window_context, token)
    return start - SCENE_EPSILON <= time_offset <= end + SCENE_EPSILON


def _token_bounds(window_context: dict[str, object], token: list[float]) -> tuple[float, float]:
    return (
        _token_value(window_context, token, "start_offset_seconds"),
        _token_value(window_context, token, "end_offset_seconds"),
    )


def _token_flag(window_context: dict[str, object], token: list[float], feature_key: str) -> bool:
    return _token_value(window_context, token, feature_key) >= 0.5


def _token_value(window_context: dict[str, object], token: list[float], feature_key: str) -> float:
    feature_index = feature_index_map(tuple(window_context["feature_keys"]))[feature_key]
    return float(token[feature_index])
