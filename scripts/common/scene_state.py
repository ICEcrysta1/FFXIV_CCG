"""scene 事实调度与输出层场景标量合成（转换与回放共用）。

- `SceneStateLookup` 按绝对时刻回答不进入状态机的场景标量（移动、停手 ETA/剩余）。
- `SceneFactScheduler` 把 scene 窗口转换为按时间提交的外部事实。
- `rewrite_scene_player_state` 按这些标量改写 canonical context 的 player 向量。

转换链路的 `SceneFactScheduler` 只有三类事实进入状态机，因为它们参与内部数值结算：Boss 可选中
（DoT tick 是否累计威力、野火命中判定）、目标数（AoE 衰减倍率）、团辅窗口（增伤倍率）；移动状态
继续由输出层合成并应用滑步豁免。自回归、PPG 与 GRPO 使用自己的 `SceneTemplateProvider`，会把经过
滑步豁免的 `movement_changed` 一并提交给状态机，以闭合在线请求时的移动合法性判断。

scene 窗口的时间轴与状态机一致（原点都是 fight 起点），因此窗口时间可直接
作为外部事实的绝对时间戳。
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from functools import lru_cache

from common.contracts import (
    FORCED_MOVEMENT_CONTEXT_KEY,
    RAID_BUFF_WINDOW_CONTEXT_KEY,
    SCENE_EPSILON,
    SLIDECAST_WINDOW_SECONDS,
    TARGETABLE_WINDOW_CONTEXT_KEY,
    TARGET_COUNT_WINDOW_CONTEXT_KEY,
)
from common.scene_window import feature_index_map

BOSS_TARGETABLE_CHANGED = "boss_targetable_changed"
TARGET_COUNT_CHANGED = "target_count_changed"
RAID_BUFF_WINDOW_CHANGED = "raid_buff_window_changed"

# 同刻事实之间的稳定提交顺序；三类事实互不依赖，顺序只用于确定性。
_FACT_ORDER = {
    BOSS_TARGETABLE_CHANGED: 0,
    TARGET_COUNT_CHANGED: 1,
    RAID_BUFF_WINDOW_CHANGED: 2,
}

_PLAYER_SCENE_FIELDS = (
    "is_moving",
    "next_untargetable_in_seconds",
    "downtime_remaining_seconds",
)


@dataclass(frozen=True)
class SceneState:
    """某个绝对时刻的场景标量；数值口径与状态机 player 向量一致。"""

    is_moving: bool
    next_downtime_eta: float
    downtime_remaining: float


@dataclass(frozen=True)
class SceneFact:
    """一条待提交的外部事实。"""

    timestamp: float
    event_kind: str
    value: bool | None = None
    target_count: int | None = None
    remaining_seconds: float | None = None


class SceneStateLookup:
    """按绝对时刻回答固定场景快照的标量，缓存不改变查询或窗口边界语义。"""

    def __init__(self, scene_context: dict[str, object] | None):
        targetable_tokens, targetable_index = _window_tokens(
            scene_context,
            TARGETABLE_WINDOW_CONTEXT_KEY,
        )
        movement_tokens, movement_index = _window_tokens(
            scene_context,
            FORCED_MOVEMENT_CONTEXT_KEY,
        )

        # 历史前缀会反复查询相同时间；缓存属于本场景，最多 4096 项。
        # 闭包只持有窗口快照，不引用 self，不让全局缓存延长已结束队列的生命周期。
        @lru_cache(maxsize=4096)
        def cached_state_at(timestamp: float) -> SceneState:
            targetable = _resolve_targetable_state(
                targetable_tokens,
                targetable_index,
                timestamp=timestamp,
            )
            return SceneState(
                is_moving=_resolve_is_moving(movement_tokens, movement_index, timestamp=timestamp),
                next_downtime_eta=targetable["next_downtime_eta"],
                downtime_remaining=targetable["downtime_remaining"],
            )

        self._cached_state_at = cached_state_at

    def state_at(self, timestamp: float) -> SceneState:
        return self._cached_state_at(timestamp)


class SceneFactScheduler:
    """把 scene 窗口转换为按时间顺序提交的外部事实流。

    调用方必须先取出所有 `timestamp <= 目标时刻` 的事实并提交，再推进时钟：
    `apply_external_event` 自身会把时钟推进到事实时刻，而状态机拒绝过去事件。
    """

    def __init__(self, scene_context: dict[str, object] | None):
        self.scene_context = scene_context
        self._lookup = SceneStateLookup(scene_context)
        self._facts = _build_scene_facts(scene_context)
        self._cursor = 0

    def state_at(self, timestamp: float) -> SceneState:
        """查询场景标量，供输出层改写 player 向量使用。"""
        return self._lookup.state_at(timestamp)

    def pop_facts_through(self, timestamp: float) -> list[SceneFact]:
        """取出所有不晚于 `timestamp` 且尚未提交的事实，并推进游标。

        比较必须严格：窗口端点动作的时刻与停手区间边界只差一个内缩量，
        用容差比较会把"动作同刻才生效的场景变化"提前到动作校验之前。
        """
        due: list[SceneFact] = []
        while self._cursor < len(self._facts) and self._facts[self._cursor].timestamp <= timestamp:
            due.append(self._facts[self._cursor])
            self._cursor += 1
        return due

def resolve_target_count_at(
    scene_context: dict[str, object] | None,
    timestamp: float,
) -> int:
    """读取某时刻的目标数量；没有窗口时按单目标处理。"""
    tokens, index = _window_tokens(scene_context, TARGET_COUNT_WINDOW_CONTEXT_KEY)
    for token in tokens:
        start, end = _token_bounds(index, token)
        if not (start - SCENE_EPSILON <= timestamp < end - SCENE_EPSILON):
            continue
        return max(0, int(round(_token_value(index, token, "target_count"))))
    return 1


def rewrite_scene_player_state(
    canonical: dict[str, object],
    *,
    observation_timestamp: float,
    next_observation_timestamp: float | None,
    scene_state_at,
) -> dict[str, object]:
    """使用两段状态自身的原始秒数改写场景字段；所有动作及最新状态共用同一规则。"""
    del observation_timestamp, next_observation_timestamp
    for context_key in ("state_history_context", "current_state_context"):
        context = canonical.get(context_key)
        if not isinstance(context, dict):
            raise ValueError(f"{context_key} must be a mapping")
        keys = context.get("player_state_feature_keys")
        if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
            raise ValueError(f"{context_key} lacks player_state_feature_keys")
        if len(set(keys)) != len(keys):
            raise ValueError(f"{context_key} has duplicate player state feature keys")
        positions = {key: index for index, key in enumerate(keys)}
        fields = ("time_seconds", "is_moving", "next_untargetable_in_seconds", "downtime_remaining_seconds")
        prefixes = ("previous_action_after", "request_state")
        for prefix in prefixes:
            for field in fields:
                if f"{prefix}.{field}" not in positions:
                    raise ValueError(f"{context_key} lacks {prefix}.{field}")
        tokens = context.get("tokens")
        if not isinstance(tokens, list):
            raise ValueError(f"{context_key}.tokens must be a list")
        for index, token in enumerate(tokens):
            vector = token.get("player_state") if isinstance(token, dict) else None
            label = f"{context_key} token={index}"
            if not isinstance(vector, list) or len(vector) != len(keys):
                raise ValueError(f"{label} player state width mismatch")
            for prefix in prefixes:
                timestamp = vector[positions[f"{prefix}.time_seconds"]]
                if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
                    raise ValueError(f"{label} {prefix}.time_seconds must be finite numeric")
                state = scene_state_at(float(timestamp))
                values = {
                    "is_moving": 1.0 if state.is_moving else 0.0,
                    "next_untargetable_in_seconds": state.next_downtime_eta,
                    "downtime_remaining_seconds": state.downtime_remaining,
                }
                for field, value in values.items():
                    position = positions[f"{prefix}.{field}"]
                    if vector[position] is not None:
                        vector[position] = float(value)
    return canonical


def _build_scene_facts(scene_context: dict[str, object] | None) -> list[SceneFact]:
    facts = [
        *_targetable_facts(scene_context),
        *_target_count_facts(scene_context),
        *_raid_buff_facts(scene_context),
    ]
    facts.sort(key=lambda fact: (fact.timestamp, _FACT_ORDER[fact.event_kind]))
    return facts


def _targetable_facts(scene_context: dict[str, object] | None) -> list[SceneFact]:
    tokens, index = _window_tokens(scene_context, TARGETABLE_WINDOW_CONTEXT_KEY)
    facts: list[SceneFact] = []
    current = True
    for token in tokens:
        start, _end = _token_bounds(index, token)
        targetable = _token_flag(index, token, "targetable")
        if targetable == current:
            continue
        facts.append(
            SceneFact(timestamp=start, event_kind=BOSS_TARGETABLE_CHANGED, value=targetable)
        )
        current = targetable
    return facts


def _target_count_facts(scene_context: dict[str, object] | None) -> list[SceneFact]:
    """按分段常量时间线产出目标数事实。

    目标数量 token 只描述多目标区间，区间之外隐含单目标（对照
    `resolve_target_count_at`）。因此区间端点必须显式归位：状态机的
    `CombatState.TargetCount` 是持久字段，只发"进入多目标"不发"离开多目标"
    会让 AoE 衰减倍率从第一次多目标之后永久生效。

    同一时刻的多个端点按区间起点先后取最终值（后开始的区间胜出），
    跟随 `resolve_target_count_at` 在区间左闭右开的判定，避免区间首尾相接时
    产生无意义的来回翻转。
    """
    tokens, index = _window_tokens(scene_context, TARGET_COUNT_WINDOW_CONTEXT_KEY)
    segments = sorted(
        (
            *_token_bounds(index, token),
            max(0, int(round(_token_value(index, token, "target_count")))),
        )
        for token in tokens
    )

    timeline: dict[float, int] = {}
    for start, end, target_count in segments:
        timeline[start] = target_count
        timeline[end] = 1

    facts: list[SceneFact] = []
    current = 1
    for timestamp in sorted(timeline):
        target_count = timeline[timestamp]
        if target_count == current:
            continue
        facts.append(
            SceneFact(
                timestamp=timestamp,
                event_kind=TARGET_COUNT_CHANGED,
                target_count=target_count,
            )
        )
        current = target_count
    return facts


def _raid_buff_facts(scene_context: dict[str, object] | None) -> list[SceneFact]:
    tokens, index = _window_tokens(scene_context, RAID_BUFF_WINDOW_CONTEXT_KEY)
    facts: list[SceneFact] = []
    for token in tokens:
        start, end = _token_bounds(index, token)
        duration = end - start
        if duration <= SCENE_EPSILON:
            continue
        facts.append(
            SceneFact(
                timestamp=start,
                event_kind=RAID_BUFF_WINDOW_CHANGED,
                value=True,
                remaining_seconds=duration,
            )
        )
        facts.append(
            SceneFact(timestamp=end, event_kind=RAID_BUFF_WINDOW_CHANGED, value=False)
        )
    return facts


def _resolve_targetable_state(
    tokens: list[list[float]],
    index: dict[str, int],
    *,
    timestamp: float,
) -> dict[str, float]:
    """解析某时刻的停手标量；口径对齐状态机 player 向量（无窗口时一律 0.0）。"""
    for token in tokens:
        start, end = _token_bounds(index, token)
        if not (start - SCENE_EPSILON <= timestamp < end - SCENE_EPSILON):
            continue
        if _token_flag(index, token, "targetable"):
            return {
                "next_downtime_eta": _next_downtime_eta(tokens, index, timestamp),
                "downtime_remaining": 0.0,
            }
        return {
            "next_downtime_eta": 0.0,
            "downtime_remaining": _round_time(max(0.0, end - timestamp)),
        }
    return {"next_downtime_eta": 0.0, "downtime_remaining": 0.0}


def _next_downtime_eta(
    tokens: list[list[float]],
    index: dict[str, int],
    timestamp: float,
) -> float:
    for token in tokens:
        if _token_flag(index, token, "targetable"):
            continue
        start, _end = _token_bounds(index, token)
        if start > timestamp + SCENE_EPSILON:
            return _round_time(start - timestamp)
    return 0.0


def _resolve_is_moving(
    tokens: list[list[float]],
    index: dict[str, int],
    *,
    timestamp: float,
) -> bool:
    """判断某时刻是否处于强制移动中，并应用滑步豁免。

    移动窗口末端距当前时刻不超过滑步容差时视为已停止移动：日志里的硬读条
    常常在移动窗口末尾完成，不做豁免会让模型看到"运动中仍硬读条"的样本。
    """
    for token in tokens:
        start, end = _token_bounds(index, token)
        if not (start - SCENE_EPSILON <= timestamp < end - SCENE_EPSILON):
            continue
        return (end - timestamp) > SLIDECAST_WINDOW_SECONDS + SCENE_EPSILON
    return False


def _window_tokens(
    scene_context: dict[str, object] | None,
    context_key: str,
) -> tuple[list[list[float]], dict[str, int]]:
    window = scene_context.get(context_key) if isinstance(scene_context, dict) else None
    if not isinstance(window, dict):
        return [], {}
    raw_tokens = window.get("tokens", [])
    feature_keys = window.get("feature_keys", [])
    if not isinstance(raw_tokens, list) or not isinstance(feature_keys, list):
        return [], {}
    tokens = [
        [float(value) for value in token]
        for token in raw_tokens
        if isinstance(token, list) and len(token) == len(feature_keys)
    ]
    return tokens, feature_index_map(tuple(str(key) for key in feature_keys))


def _token_bounds(index: dict[str, int], token: list[float]) -> tuple[float, float]:
    return (
        _token_value(index, token, "start_offset_seconds"),
        _token_value(index, token, "end_offset_seconds"),
    )


def _token_flag(index: dict[str, int], token: list[float], feature_key: str) -> bool:
    return _token_value(index, token, feature_key) >= 0.5


def _token_value(index: dict[str, int], token: list[float], feature_key: str) -> float:
    if feature_key not in index:
        raise KeyError(f"scene window token is missing feature key: {feature_key}")
    return float(token[index[feature_key]])


def _round_time(value: float) -> float:
    return round(float(value), 4)
