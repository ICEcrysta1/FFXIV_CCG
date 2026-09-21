"""测试共用辅助函数。"""

from __future__ import annotations

from common.scene_context_schema import (
    FORCED_MOVEMENT_CONTEXT_KEY,
    RAID_BUFF_WINDOW_CONTEXT_KEY,
    TARGET_COUNT_WINDOW_CONTEXT_KEY,
    TARGETABLE_WINDOW_CONTEXT_KEY,
    build_empty_scene_context,
)
from scripts.convert_fflogs.scene_context import (
    build_forced_movement_window_token as forced_movement_window_token,
    build_raid_buff_window_token as raid_buff_window_token,
    build_targetable_window_token as targetable_window_token,
)

DEFAULT_BASE_GCD = 2.5


def build_test_scene_context(
    *,
    targetable_tokens: list[list[float]] | None = None,
    forced_movement_tokens: list[list[float]] | None = None,
    raid_buff_tokens: list[list[float]] | None = None,
    target_count_tokens: list[list[float]] | None = None,
) -> dict[str, object]:
    scene_context = build_empty_scene_context()
    if targetable_tokens is not None:
        scene_context[TARGETABLE_WINDOW_CONTEXT_KEY]["tokens"] = targetable_tokens
    if forced_movement_tokens is not None:
        scene_context[FORCED_MOVEMENT_CONTEXT_KEY]["tokens"] = forced_movement_tokens
    if raid_buff_tokens is not None:
        scene_context[RAID_BUFF_WINDOW_CONTEXT_KEY]["tokens"] = raid_buff_tokens
    if target_count_tokens is not None:
        scene_context[TARGET_COUNT_WINDOW_CONTEXT_KEY]["tokens"] = target_count_tokens
    return scene_context


def history_vector_value(history_payload, group_key: str, feature_key: str, *, token_index: int = 0) -> float:
    index = history_payload[f"{group_key}_feature_keys"].index(feature_key)
    return history_payload["tokens"][token_index][group_key][index]


def candidate_state_token_by_skill(payload, skill_key: str) -> dict[str, list[float]]:
    index = next(
        index
        for index, token in enumerate(payload["candidate_skill_context"])
        if token["skill_key"] == skill_key
    )
    return payload["candidate_state_context"]["tokens"][index]


def candidate_state_vector_value(payload, skill_key: str, group_key: str, feature_key: str) -> float:
    candidate_state_context = payload["candidate_state_context"]
    index = candidate_state_context[f"{group_key}_feature_keys"].index(feature_key)
    return candidate_state_token_by_skill(payload, skill_key)[group_key][index]
