"""scene window 构造与输入规范化。"""

from __future__ import annotations

from collections import Counter

from common.scene_context_schema import (
    FORCED_MOVEMENT_CONTEXT_KEY,
    FORCED_MOVEMENT_FEATURE_KEYS,
    RAID_BUFF_WINDOW_CONTEXT_KEY,
    RAID_BUFF_WINDOW_FEATURE_KEYS,
    RAID_BUFF_WINDOW_SOURCE_KEYS,
    raid_buff_window_feature_keys,
    TARGET_COUNT_WINDOW_CONTEXT_KEY,
    TARGET_COUNT_WINDOW_FEATURE_KEYS,
    TARGETABLE_WINDOW_CONTEXT_KEY,
    TARGETABLE_WINDOW_FEATURE_KEYS,
    TARGETABLE_SEGMENT_KINDS,
    build_empty_scene_context,
    build_window_context,
)

from ..config.constants import (
    MULTI_TARGET_MARKER_GRACE_SECONDS,
    MULTI_TARGET_OBSERVATION_TIMEOUT_SECONDS,
    _round_time,
)


def build_scene_context(
    *,
    targetable_window_context: dict[str, object] | None = None,
    forced_movement_context: dict[str, object] | None = None,
    raid_buff_window_context: dict[str, object] | None = None,
    target_count_window_context: dict[str, object] | None = None,
) -> dict[str, object]:
    """组装完整 scene_context。"""
    scene_context = build_empty_scene_context()
    if targetable_window_context is not None:
        scene_context[TARGETABLE_WINDOW_CONTEXT_KEY] = targetable_window_context
    if forced_movement_context is not None:
        scene_context[FORCED_MOVEMENT_CONTEXT_KEY] = forced_movement_context
    if raid_buff_window_context is not None:
        scene_context[RAID_BUFF_WINDOW_CONTEXT_KEY] = raid_buff_window_context
    if target_count_window_context is not None:
        scene_context[TARGET_COUNT_WINDOW_CONTEXT_KEY] = target_count_window_context
    return scene_context


def build_targetable_window_context(
    fight_start: float,
    fight_end: float,
    downtime_windows: list[dict[str, float]],
) -> dict[str, object]:
    """把可输出/停手窗口转换成向量 token 上下文。"""
    duration = fight_end - fight_start
    if duration <= 0:
        return build_window_context(TARGETABLE_WINDOW_FEATURE_KEYS, [])

    tokens: list[list[float]] = []
    cursor = 0.0
    if not downtime_windows:
        tokens.append(build_targetable_window_token(0.0, duration, targetable=True, segment_kind="combat"))
        return build_window_context(TARGETABLE_WINDOW_FEATURE_KEYS, tokens)

    for window in downtime_windows:
        relative_start = float(window["start"]) - fight_start
        relative_end = float(window["end"]) - fight_start
        if relative_start > cursor:
            tokens.append(
                build_targetable_window_token(
                    cursor,
                    relative_start,
                    targetable=True,
                    segment_kind="combat",
                )
            )
        tokens.append(
            build_targetable_window_token(
                relative_start,
                relative_end,
                targetable=False,
                segment_kind="downtime",
            )
        )
        cursor = relative_end

    if cursor < duration:
        tokens.append(
            build_targetable_window_token(
                cursor,
                duration,
                targetable=True,
                segment_kind="combat_final",
            )
        )
    return build_window_context(TARGETABLE_WINDOW_FEATURE_KEYS, tokens)


def build_forced_movement_context(
    merged_windows: list[tuple[float, float]],
    *,
    fight_start: float,
) -> dict[str, object]:
    """把强制移动窗口转换成向量 token 上下文。"""
    tokens = [
        build_forced_movement_window_token(start - fight_start, end - fight_start)
        for start, end in merged_windows
    ]
    return build_window_context(FORCED_MOVEMENT_FEATURE_KEYS, tokens)


def build_target_count_window_context(
    events: list[dict[str, object]],
    *,
    source_id: int,
    skill_book,
    fight_start: float,
    fight_end: float,
) -> dict[str, object]:
    """从 FFLogs AoE 命中事件构造多目标辅助标记。

    这里只观察支持多目标的技能实际命中了多少个不同 targetID；只有确认过
    ``target_count >= 2`` 才生成标记。不把实体 ID 写进 scene token，避免模型记忆
    每场战斗的临时对象编号。标记只描述回放中出现过的多目标区间，不直接承担
    CombatState 的目标数量或伤害倍率逻辑。
    """
    duration = max(0.0, float(fight_end) - float(fight_start))
    if duration <= 0.0:
        return build_window_context(TARGET_COUNT_WINDOW_FEATURE_KEYS, [])

    aoe_game_ids = {
        int(skill.game_id)
        for skill in skill_book.enabled_skills()
        if int(skill.max_targets) > 1
    }
    observed_counts = _collect_target_count_observations(
        events,
        source_id=source_id,
        aoe_game_ids=aoe_game_ids,
        fight_start=fight_start,
        fight_end=fight_end,
    )
    tokens = [
        build_target_count_window_token(start, end, count)
        for start, end, count in _build_multi_target_segments(duration, observed_counts)
    ]
    return build_window_context(TARGET_COUNT_WINDOW_FEATURE_KEYS, tokens)


def _collect_target_count_observations(
    events: list[dict[str, object]],
    *,
    source_id: int,
    aoe_game_ids: set[int],
    fight_start: float,
    fight_end: float,
) -> list[tuple[float, int]]:
    grouped_targets: dict[tuple[float, int], set[int]] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("sourceID") != source_id:
            continue
        if event.get("type") not in {"damage", "calculateddamage"}:
            continue
        try:
            ability_id = int(event.get("abilityGameID", 0))
            target_id = int(event.get("targetID", -1))
            timestamp = float(event["timestamp"]) / 1000.0 - float(fight_start)
        except (KeyError, TypeError, ValueError):
            continue
        if ability_id not in aoe_game_ids or target_id < 0:
            continue
        if timestamp < 0.0 or timestamp > float(fight_end) - float(fight_start):
            continue
        grouped_targets.setdefault((round(timestamp, 4), ability_id), set()).add(target_id)

    return sorted(
        (timestamp, len(target_ids))
        for (timestamp, _ability_id), target_ids in grouped_targets.items()
    )


def _build_multi_target_segments(
    duration: float,
    observations: list[tuple[float, int]],
) -> list[tuple[float, float, int]]:
    """按双目标事件和超时窗口构造仅包含多目标的区间。"""
    multi_target_observations = [
        (min(max(float(timestamp), 0.0), duration), max(2, int(count)))
        for timestamp, count in observations
        if int(count) >= 2
    ]
    if not multi_target_observations:
        return [(0.0, duration, 1)]

    clusters: list[list[tuple[float, int]]] = []
    for observation in multi_target_observations:
        if (
            not clusters
            or observation[0] - clusters[-1][-1][0]
            > MULTI_TARGET_OBSERVATION_TIMEOUT_SECONDS
        ):
            clusters.append([observation])
        else:
            clusters[-1].append(observation)

    return [
        (
            cluster[0][0],
            min(duration, cluster[-1][0] + MULTI_TARGET_MARKER_GRACE_SECONDS),
            Counter(count for _timestamp, count in cluster).most_common(1)[0][0],
        )
        for cluster in clusters
        if cluster[-1][0] < duration
    ]


def build_raid_buff_window_context(
    actions: list[dict[str, object]],
    *,
    fight_start: float,
    window_duration: float,
    marker_keys: tuple[str, ...] = RAID_BUFF_WINDOW_SOURCE_KEYS,
) -> dict[str, object]:
    """把团辅窗口转换成向量 token 上下文。"""
    if not marker_keys:
        raise ValueError("raid buff marker keys must not be empty")

    marker_events: list[tuple[float, str]] = []
    for action in actions:
        action_key = str(action["action_key"])
        if action_key not in marker_keys:
            continue
        marker_events.append((float(action["timestamp"]) - fight_start, action_key))

    marker_events.sort(key=lambda item: item[0])
    grouped_events: list[list[tuple[float, str]]] = []
    for event in marker_events:
        if not grouped_events or event[0] - grouped_events[-1][-1][0] > window_duration:
            grouped_events.append([event])
        else:
            grouped_events[-1].append(event)

    tokens: list[list[float]] = []
    feature_keys = raid_buff_window_feature_keys(marker_keys)
    for grouped_event in grouped_events:
        start_offset = sum(timestamp for timestamp, _ in grouped_event) / len(grouped_event)
        source_keys = tuple(sorted({source_key for _, source_key in grouped_event}))
        tokens.append(
            build_raid_buff_window_token(
                start_offset=start_offset,
                end_offset=start_offset + window_duration,
                marker_keys=marker_keys,
                source_keys=source_keys,
            )
        )
    return build_window_context(feature_keys, tokens)


def normalize_scene_context(scene_context_payload: object) -> dict[str, object]:
    """校验并规范化 scene_context 载荷。"""
    normalized = build_empty_scene_context()
    payload = scene_context_payload if isinstance(scene_context_payload, dict) else {}
    normalized[TARGETABLE_WINDOW_CONTEXT_KEY] = _normalize_window_context(
        payload.get(TARGETABLE_WINDOW_CONTEXT_KEY),
        expected_feature_keys=TARGETABLE_WINDOW_FEATURE_KEYS,
    )
    normalized[FORCED_MOVEMENT_CONTEXT_KEY] = _normalize_window_context(
        payload.get(FORCED_MOVEMENT_CONTEXT_KEY),
        expected_feature_keys=FORCED_MOVEMENT_FEATURE_KEYS,
    )
    raid_buff_payload = payload.get(RAID_BUFF_WINDOW_CONTEXT_KEY)
    normalized[RAID_BUFF_WINDOW_CONTEXT_KEY] = _normalize_window_context(
        raid_buff_payload,
        expected_feature_keys=_resolve_raid_buff_feature_keys(raid_buff_payload),
    )
    normalized[TARGET_COUNT_WINDOW_CONTEXT_KEY] = _normalize_window_context(
        payload.get(TARGET_COUNT_WINDOW_CONTEXT_KEY),
        expected_feature_keys=TARGET_COUNT_WINDOW_FEATURE_KEYS,
    )
    return normalized


def _resolve_raid_buff_feature_keys(window_context_payload: object) -> tuple[str, ...]:
    """从已落盘窗口上下文保留职业/系统配置生成的来源维度。"""
    if not isinstance(window_context_payload, dict):
        return RAID_BUFF_WINDOW_FEATURE_KEYS
    raw_feature_keys = window_context_payload.get("feature_keys")
    if not isinstance(raw_feature_keys, list) or len(raw_feature_keys) < 4:
        return RAID_BUFF_WINDOW_FEATURE_KEYS
    feature_keys = tuple(str(feature_key) for feature_key in raw_feature_keys)
    if feature_keys[:3] != (
        "start_offset_seconds",
        "end_offset_seconds",
        "duration_seconds",
    ):
        return RAID_BUFF_WINDOW_FEATURE_KEYS
    source_keys = tuple(
        feature_key.removeprefix("source.")
        for feature_key in feature_keys[3:]
        if feature_key.startswith("source.") and feature_key.removeprefix("source.")
    )
    if len(source_keys) != len(feature_keys) - 3 or len(set(source_keys)) != len(source_keys):
        return RAID_BUFF_WINDOW_FEATURE_KEYS
    return raid_buff_window_feature_keys(source_keys)


def _normalize_window_context(
    window_context_payload: object,
    *,
    expected_feature_keys: tuple[str, ...],
) -> dict[str, object]:
    if not isinstance(window_context_payload, dict):
        return build_window_context(expected_feature_keys, [])

    raw_tokens = window_context_payload.get("tokens", [])
    normalized_tokens: list[list[float]] = []
    if isinstance(raw_tokens, list):
        expected_size = len(expected_feature_keys)
        for token in raw_tokens:
            if isinstance(token, list) and len(token) == expected_size:
                normalized_tokens.append([float(value) for value in token])
    return build_window_context(expected_feature_keys, normalized_tokens)


def build_targetable_window_token(
    start_offset: float,
    end_offset: float,
    *,
    targetable: bool,
    segment_kind: str,
) -> list[float]:
    if segment_kind not in TARGETABLE_SEGMENT_KINDS:
        raise ValueError(f"unsupported targetable segment kind: {segment_kind}")
    return build_window_vector(
        start_offset=start_offset,
        end_offset=end_offset,
        extras=[
            1.0 if targetable else 0.0,
            *(1.0 if candidate_kind == segment_kind else 0.0 for candidate_kind in TARGETABLE_SEGMENT_KINDS),
        ],
    )


def build_raid_buff_window_token(
    start_offset: float,
    end_offset: float,
    *,
    source_key: str | None = None,
    marker_keys: tuple[str, ...] = RAID_BUFF_WINDOW_SOURCE_KEYS,
    source_keys: tuple[str, ...] | None = None,
) -> list[float]:
    selected_sources = source_keys if source_keys is not None else ((source_key or marker_keys[0]),)
    unsupported_sources = set(selected_sources) - set(marker_keys)
    if unsupported_sources:
        raise ValueError(f"unsupported raid buff source keys: {sorted(unsupported_sources)}")
    return build_window_vector(
        start_offset=start_offset,
        end_offset=end_offset,
        extras=[1.0 if candidate_source in selected_sources else 0.0 for candidate_source in marker_keys],
    )


def build_forced_movement_window_token(start_offset: float, end_offset: float) -> list[float]:
    """构造强制移动窗口 token。"""
    return build_window_vector(start_offset=start_offset, end_offset=end_offset)


def build_target_count_window_token(
    start_offset: float,
    end_offset: float,
    target_count: int,
) -> list[float]:
    """构造目标数量 scene token。"""
    if target_count < 0:
        raise ValueError(f"target_count must be >= 0, got {target_count}")
    return build_window_vector(
        start_offset=start_offset,
        end_offset=end_offset,
        extras=[float(target_count)],
    )


def build_window_vector(
    *,
    start_offset: float,
    end_offset: float,
    extras: list[float] | tuple[float, ...] = (),
) -> list[float]:
    start_value = _round_time(start_offset)
    end_value = _round_time(end_offset)
    duration_value = _round_time(max(0.0, end_offset - start_offset))
    return [start_value, end_value, duration_value, *[float(value) for value in extras]]
