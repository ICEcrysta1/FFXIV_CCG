"""转换和回放共用的场景执行视图、事实批次及状态 ETA 合成。"""

from __future__ import annotations

import math
import struct
from bisect import bisect_right
from dataclasses import asdict, dataclass
from functools import lru_cache

from common.contracts import (
    FORCED_MOVEMENT_CONTEXT_KEY, RAID_BUFF_WINDOW_CONTEXT_KEY, SCENE_EPSILON,
    SLIDECAST_WINDOW_SECONDS, TARGETABLE_WINDOW_CONTEXT_KEY, TARGET_COUNT_WINDOW_CONTEXT_KEY,
)
from common.output_context_schema import STATE_SNAPSHOTS

BOSS_TARGETABLE_CHANGED = "boss_targetable_changed"
MOVEMENT_CHANGED = "movement_changed"
TARGET_COUNT_CHANGED = "target_count_changed"
RAID_BUFF_WINDOW_CHANGED = "raid_buff_window_changed"


@dataclass(frozen=True)
class SceneState:
    """同一执行视图的查询结果；不直接替代状态机冻结的事实字段。"""

    is_moving: bool
    next_downtime_eta: float
    downtime_remaining: float
    boss_targetable: bool = True
    target_count: int = 1
    raid_buff_end: float | None = None


@dataclass(frozen=True)
class SceneFact:
    timestamp: float
    event_kind: str
    value: bool | None = None
    target_count: int | None = None
    remaining_seconds: float | None = None


class SceneStateLookup:
    """只读执行视图：端点按已有 raw FP32 表达投影，绝不回写 scene token。"""

    def __init__(self, scene_context: dict[str, object] | None):
        self.targetable_windows = _windows(scene_context, TARGETABLE_WINDOW_CONTEXT_KEY, "targetable")
        self.target_count_windows = _windows(scene_context, TARGET_COUNT_WINDOW_CONTEXT_KEY, "target_count")
        movement = _windows(scene_context, FORCED_MOVEMENT_CONTEXT_KEY)
        self.movement_windows = _merge_intervals(
            (start, end - SLIDECAST_WINDOW_SECONDS)
            for start, end, _ in movement
            if end - start > SLIDECAST_WINDOW_SECONDS + SCENE_EPSILON
        )
        self.raid_buff_windows = _merge_intervals(
            (start, end) for start, end, _ in _windows(scene_context, RAID_BUFF_WINDOW_CONTEXT_KEY)
        )
        targetable, counts = self.targetable_windows, self.target_count_windows
        movements, raid_buffs = self.movement_windows, self.raid_buff_windows

        # 缓存只捕获不可变窗口，不持有场景/轨迹实例。
        @lru_cache(maxsize=4096)
        def cached_state_at(timestamp: float) -> SceneState:
            active = _active_window(targetable, timestamp)
            boss_targetable = active is None or bool(active[2])
            downtime_remaining = 0.0 if boss_targetable else max(0.0, active[1] - timestamp)
            future = [start for start, _end, value in targetable if not value and start > timestamp]
            eta = max(0.0, min(future) - timestamp) if boss_targetable and future else 0.0
            count = _active_window(counts, timestamp)
            return SceneState(
                is_moving=_active_end(movements, timestamp) is not None,
                next_downtime_eta=round(eta, 4),
                downtime_remaining=round(downtime_remaining, 4),
                boss_targetable=boss_targetable,
                target_count=1 if count is None else int(count[2]),
                raid_buff_end=_active_end(raid_buffs, timestamp),
            )

        self._cached_state_at = cached_state_at

    def state_at(self, timestamp: float) -> SceneState:
        return self._cached_state_at(_finite_number(timestamp, "scene query timestamp"))

    def last_targetable_end(self) -> float:
        ends = [end for _start, end, value in self.targetable_windows if value]
        if not ends:
            raise ValueError("scene cache has no targetable Boss window")
        return max(ends)

    def facts(self) -> tuple[SceneFact, ...]:
        boundaries = sorted({
            bound
            for windows in (self.targetable_windows, self.target_count_windows,
                            self.movement_windows, self.raid_buff_windows)
            for window in windows for bound in window[:2]
        })
        previous = SceneState(False, 0.0, 0.0)
        facts = []
        for timestamp in boundaries:
            state = self.state_at(timestamp)
            facts.extend(_state_facts(timestamp, state, previous))
            previous = state
        return tuple(facts)


class SceneFactScheduler:
    """独立轨迹游标，同刻全部事实在一次 session 调用中提交。"""

    def __init__(self, scene_context: dict[str, object] | None):
        self.scene_context = scene_context
        self.lookup = SceneStateLookup(scene_context)
        self._facts = self.lookup.facts()
        self._fact_times = tuple(fact.timestamp for fact in self._facts)
        self.reset()

    def reset(self) -> None:
        self._cursor = 0
        self._synced_time: float | None = None

    def state_at(self, timestamp: float) -> SceneState:
        return self.lookup.state_at(timestamp)

    def next_event_after(self, timestamp: float, *, event_kind: str | None = None) -> float | None:
        return next((fact.timestamp for fact in self._facts[bisect_right(self._fact_times, timestamp):]
                     if event_kind is None or fact.event_kind == event_kind), None)

    def pop_facts_through(self, timestamp: float) -> list[SceneFact]:
        """取出到期事实；比较严格，不能提前把未来边界合并进当前时刻。"""
        end = bisect_right(self._fact_times, timestamp)
        due = list(self._facts[self._cursor:end])
        self._cursor = max(self._cursor, end)
        return due

    def sync_through(self, backend, timestamp: float) -> None:
        """首次提交当前有效状态，之后按原始边界分批推进，不重放过去事实。"""
        timestamp = _finite_number(timestamp, "scene sync timestamp")
        if self._synced_time is None:
            _submit_batch(backend, _state_facts(timestamp, self.state_at(timestamp), None))
            self._cursor = bisect_right(self._fact_times, timestamp)
        else:
            if timestamp < self._synced_time:
                raise ValueError("scene fact cursor cannot move backwards; reset the trajectory first")
            end = bisect_right(self._fact_times, timestamp)
            while self._cursor < end:
                batch_end = bisect_right(self._fact_times, self._facts[self._cursor].timestamp)
                _submit_batch(backend, self._facts[self._cursor:batch_end])
                # 只在成功提交后移动游标，失败时可见原批次。
                self._cursor = batch_end
        self._synced_time = timestamp


def _submit_batch(backend, facts) -> None:
    result = backend.apply_external_events([asdict(fact) for fact in facts])
    if not result.accepted:
        raise ValueError(f"scene facts rejected at {facts[0].timestamp}: {result.reason}")


def resolve_target_count_at(scene_context: dict[str, object] | None, timestamp: float) -> int:
    """一次性场景查询；连续查询的调用方复用 SceneStateLookup。"""
    return SceneStateLookup(scene_context).state_at(timestamp).target_count


def rewrite_scene_player_state(canonical: dict[str, object], *, scene_state_at) -> dict[str, object]:
    """纯合成两段各自时刻的 ETA/剩余；共享原对象及所有非状态 token 保持不变。"""
    result = canonical
    for context_key in ("state_history_context", "current_state_context"):
        context = canonical.get(context_key)
        if not isinstance(context, dict):
            raise ValueError(f"{context_key} must be a mapping")
        keys = context.get("player_state_feature_keys")
        if not isinstance(keys, (list, tuple)) or not all(isinstance(key, str) for key in keys):
            raise ValueError(f"{context_key} lacks player_state_feature_keys")
        if len(set(keys)) != len(keys):
            raise ValueError(f"{context_key} has duplicate player state feature keys")
        positions = {key: index for index, key in enumerate(keys)}
        prefixes = STATE_SNAPSHOTS
        fields = ("next_untargetable_in_seconds", "downtime_remaining_seconds")
        for prefix in prefixes:
            for field in ("time_seconds", *fields):
                if f"{prefix}.{field}" not in positions:
                    raise ValueError(f"{context_key} lacks {prefix}.{field}")
        tokens = context.get("tokens")
        if not isinstance(tokens, list):
            raise ValueError(f"{context_key}.tokens must be a list")
        changed_tokens = None
        for index, token in enumerate(tokens):
            vector = token.get("player_state") if isinstance(token, dict) else None
            label = f"{context_key} token={index}"
            if not isinstance(vector, (list, tuple)) or len(vector) != len(keys):
                raise ValueError(f"{label} player state width mismatch")
            changed_vector = None
            for prefix in prefixes:
                timestamp = _finite_number(vector[positions[f"{prefix}.time_seconds"]],
                                           f"{label} {prefix}.time_seconds")
                state = scene_state_at(timestamp)
                for field, value in zip(fields, (state.next_downtime_eta, state.downtime_remaining), strict=True):
                    position = positions[f"{prefix}.{field}"]
                    if vector[position] is not None and vector[position] != value:
                        if changed_vector is None:
                            changed_vector = list(vector)
                        changed_vector[position] = float(value)
            if changed_vector is not None:
                if changed_tokens is None:
                    changed_tokens = list(tokens)
                changed_tokens[index] = {**token, "player_state": changed_vector}
        if changed_tokens is not None:
            if result is canonical:
                result = dict(canonical)
            result[context_key] = {**context, "tokens": changed_tokens}
    return result


def _state_facts(timestamp: float, state: SceneState, previous: SceneState | None) -> list[SceneFact]:
    facts = []
    if previous is None or state.boss_targetable != previous.boss_targetable:
        facts.append(SceneFact(timestamp, BOSS_TARGETABLE_CHANGED, value=state.boss_targetable))
    if previous is None or state.is_moving != previous.is_moving:
        facts.append(SceneFact(timestamp, MOVEMENT_CHANGED, value=state.is_moving))
    if previous is None or state.target_count != previous.target_count:
        facts.append(SceneFact(timestamp, TARGET_COUNT_CHANGED, target_count=state.target_count))
    if previous is None or state.raid_buff_end != previous.raid_buff_end:
        active = state.raid_buff_end is not None
        facts.append(SceneFact(timestamp, RAID_BUFF_WINDOW_CHANGED, value=active,
                               remaining_seconds=state.raid_buff_end - timestamp if active else None))
    return facts


def _finite_number(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite numeric")
    return float(value)


def _fp32(value, label: str) -> float:
    number = _finite_number(value, label)
    try:
        projected = struct.unpack("f", struct.pack("f", number))[0]
    except OverflowError as exc:
        raise ValueError(f"{label} must be finite FP32") from exc
    if not math.isfinite(projected):
        raise ValueError(f"{label} must be finite FP32")
    return projected


def _windows(scene_context, context_key: str, value_key: str | None = None) -> tuple:
    window = scene_context.get(context_key) if isinstance(scene_context, dict) else None
    if window is None:
        return ()
    if not isinstance(window, dict):
        raise ValueError(f"{context_key} must be a mapping")
    keys, tokens = window.get("feature_keys"), window.get("tokens")
    if not isinstance(keys, (list, tuple)) or not isinstance(tokens, (list, tuple)):
        raise ValueError(f"{context_key} requires feature_keys and tokens")
    if len(set(keys)) != len(keys):
        raise ValueError(f"{context_key} has duplicate feature keys")
    positions = {key: index for index, key in enumerate(keys)}
    for field in ("start_offset_seconds", "end_offset_seconds", *((value_key,) if value_key else ())):
        if field not in positions:
            raise ValueError(f"{context_key} lacks {field}")
    windows = []
    for token in tokens:
        if not isinstance(token, (list, tuple)) or len(token) != len(keys):
            raise ValueError(f"{context_key} token width mismatch")
        values = [_fp32(value, f"{context_key}.{key}") for key, value in zip(keys, token, strict=True)]
        start, end = (values[positions[key]] for key in ("start_offset_seconds", "end_offset_seconds"))
        if float(token[positions["start_offset_seconds"]]) > float(token[positions["end_offset_seconds"]]):
            raise ValueError(f"{context_key} requires start <= end")
        value = None if value_key is None else values[positions[value_key]]
        if value_key == "targetable" and value not in (0.0, 1.0):
            raise ValueError("raw scene targetable must be binary")
        if value_key == "target_count" and (value < 0 or not value.is_integer()):
            raise ValueError("raw scene target_count must be a non-negative integer")
        if end - start > SCENE_EPSILON:
            windows.append((start, end, value))
    return tuple(sorted(windows, key=lambda item: (item[0], item[1])))


def _merge_intervals(intervals) -> tuple[tuple[float, float], ...]:
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _active_window(windows, timestamp: float):
    # 同刻冲突按后开始窗口优先，查询与事实生成共享唯一规则。
    return next((window for window in reversed(windows) if window[0] <= timestamp < window[1]), None)


def _active_end(windows, timestamp: float) -> float | None:
    active = _active_window(windows, timestamp)
    return None if active is None else active[1]
