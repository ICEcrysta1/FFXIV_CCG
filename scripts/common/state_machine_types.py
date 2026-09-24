"""进程内 C# 状态机接口使用的不可变结果类型。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimelinePoint:
    """推进后的时间线位置；`next_scheduled_event_time` 为状态机声明的下一事件时刻。"""

    timestamp: float
    next_scheduled_event_time: float | None


@dataclass(frozen=True)
class ActionSubmissionResult:
    """一次动作提交的轻量结果。

    不含技能类型与实际占用时长：调用方按需从自己的 skill_book 判定 gcd/ogcd。
    `queued=True` 时 `accepted_timestamp > request_timestamp`，动作在 `effect_timestamp`
    才生效，调用方推进时不能用 `accepted + 读条` 自行推算。
    """

    accepted: bool
    queued: bool
    reason: str
    action_instance_id: str | None
    request_timestamp: float
    accepted_timestamp: float | None
    effect_timestamp: float | None
    next_scheduled_event_time: float | None


@dataclass(frozen=True)
class ValidationResult:
    """只读合法性探测结果；探测会把时钟推进到请求时刻，但不提交动作。"""

    legal: bool
    reason: str
    timestamp: float
    next_scheduled_event_time: float | None


@dataclass(frozen=True)
class ExternalEventResult:
    accepted: bool
    reason: str
    event_kind: str
    timestamp: float
    next_scheduled_event_time: float | None


@dataclass(frozen=True)
class PolicyDecisionResult:
    action: str
    timestamp: float
    next_observation_timestamp: float
    gcd_index: int


@dataclass(frozen=True)
class ObservationResult:
    timestamp: float
    format: str
    next_scheduled_event_time: float | None
    context: object
