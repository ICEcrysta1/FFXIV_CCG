"""自回放与 PPG 共用的绝对时间决策调度。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from types import SimpleNamespace

from common.yaml_config import load_yaml_mapping


DECISION_TIME_EPSILON = 1e-6


@dataclass(frozen=True)
class DecisionTiming:
    post_gcd_delay_seconds: float
    gcd_request_lead_seconds: float
    ogcd_interval_seconds: float


@lru_cache(maxsize=1)
def decision_timing() -> DecisionTiming:
    """调用方时序只读取 YAML；状态机不模拟动画锁。"""
    config = load_yaml_mapping(Path(__file__).resolve().parents[2] / "config/default.yaml")
    values = config["decision_timing"]
    timing = DecisionTiming(**{key: float(values[key]) for key in DecisionTiming.__dataclass_fields__})
    if any(not math.isfinite(value) or value < 0 for value in vars(timing).values()):
        raise ValueError("decision_timing values must be finite and nonnegative")
    if timing.ogcd_interval_seconds <= 0:
        raise ValueError("ogcd_interval_seconds must be positive")
    if timing.gcd_request_lead_seconds > float(config["engine_timing"]["action_queue_window_seconds"]):
        raise ValueError("gcd_request_lead_seconds exceeds action queue window")
    return timing


def gcd_request_delay(state) -> float:
    return max(0.0, float(state.gcd_remaining) - decision_timing().gcd_request_lead_seconds)


def is_gcd_decision(gcd_remaining: float) -> bool:
    return float(gcd_remaining) <= decision_timing().gcd_request_lead_seconds + DECISION_TIME_EPSILON


class DecisionScheduler:
    """把动作占用和 scene 边界推进统一委托给状态机的绝对时间接口。"""

    def __init__(self, backend, observer: Callable[[float], object], scene_provider=None):
        self._backend = backend
        self._observer = observer
        self._scene_provider = scene_provider

    def _sync_scene(self, state):
        sync_state = getattr(self._scene_provider, "sync_state", None)
        if callable(sync_state):
            sync_state(state)
        return state

    def _next_scene_event_after(self, timestamp: float) -> float | None:
        resolver = getattr(self._scene_provider, "next_state_event_after", None)
        if callable(resolver):
            return resolver(timestamp)
        resolver = getattr(self._scene_provider, "next_targetable_event_after", None)
        return None if not callable(resolver) else resolver(timestamp)

    def advance_by(
        self,
        state,
        seconds: float,
        *,
        interrupt_on_scene_event: bool = False,
        end_time: float | None = None,
    ):
        """按绝对时间推进，必要时在 scene 边界重新同步外部事实。"""
        remaining = max(0.0, float(seconds))
        while remaining > DECISION_TIME_EPSILON:
            next_scene_event = self._next_scene_event_after(float(state.time))
            step = remaining
            if next_scene_event is not None:
                step = min(step, max(0.0, float(next_scene_event) - float(state.time)))
            if end_time is not None:
                step = min(step, max(0.0, float(end_time) - float(state.time)))
            if step <= DECISION_TIME_EPSILON:
                break
            target_time = float(state.time) + step
            # 外部事实与同戳内部事件共用时间线排序；先挂载 scene 事实，
            # 再推进到边界，才能让外部事实按 ExternalScene 优先级生效。
            if (
                next_scene_event is not None
                and abs(target_time - float(next_scene_event)) <= DECISION_TIME_EPSILON
            ):
                self._sync_scene(SimpleNamespace(time=target_time))
            point = self._backend.advance_to(target_time)
            state = self._observer(point.timestamp)
            remaining -= step
            state = self._sync_scene(state)
            if (
                interrupt_on_scene_event
                and next_scene_event is not None
                and abs(float(state.time) - float(next_scene_event)) <= DECISION_TIME_EPSILON
                and remaining > DECISION_TIME_EPSILON
            ):
                break
        return state

    def advance_submitted_action(
        self,
        state,
        submission,
        *,
        action_kind: str,
        end_time: float | None = None,
    ):
        """从提交结果计算动作占用并推进到下一个决策时刻。"""
        accepted_timestamp = submission.accepted_timestamp
        if accepted_timestamp is None:
            return state
        if accepted_timestamp > float(state.time) + DECISION_TIME_EPSILON:
            state = self.advance_by(
                state,
                accepted_timestamp - float(state.time),
                end_time=end_time,
            )
        if end_time is not None and float(state.time) >= end_time - DECISION_TIME_EPSILON:
            return state
        # 排队动作在接受时刻才建立读条锁，必须重新观测实际读条剩余时间。
        state = self._observer(float(state.time))
        state = self._sync_scene(state)
        timing = decision_timing()
        if action_kind == "gcd":
            # 使用完整读条剩余时间，不由提前生效时间反推读条。
            occupancy = float(state.cast_remaining) + timing.post_gcd_delay_seconds
        else:
            occupancy = timing.ogcd_interval_seconds
        state = self.advance_by(state, occupancy, end_time=end_time)
        # 首次插入预留两段动画间隔；后续插入只需留足下一段间隔。
        reserve = timing.ogcd_interval_seconds * (2 if action_kind == "gcd" else 1)
        if float(state.gcd_remaining) > reserve + DECISION_TIME_EPSILON:
            return state
        return self.advance_by(state, gcd_request_delay(state), end_time=end_time)

    def advance_to_next_decision(self, state, *, end_time=None):
        """无候选时跳到下一真实事件；插入阶段直接让出剩余 GCD 窗口。"""
        delay = gcd_request_delay(state)
        if delay <= DECISION_TIME_EPSILON:
            times = [getattr(state, "next_scheduled_event_time", None),
                     self._next_scene_event_after(float(state.time))]
            if float(state.gcd_remaining) > DECISION_TIME_EPSILON:
                times.append(float(state.time) + float(state.gcd_remaining))
            future = [float(value) - float(state.time) for value in times
                      if value is not None and float(value) > float(state.time) + DECISION_TIME_EPSILON]
            if not future:
                return None
            delay = min(future)
        result = self.advance_by(state, delay, end_time=end_time)
        return result if float(result.time) > float(state.time) + DECISION_TIME_EPSILON else None
