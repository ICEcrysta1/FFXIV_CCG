"""通过原生扩展在当前 Python 进程中直接调用 Rust 战斗状态机。"""

from __future__ import annotations

import json
import pickle
import threading
from pathlib import Path
from typing import Any, Self

from common.context.contracts import SIDECAR_CONTRACT_VERSION
from scripts.common.state_machine_types import (
    ActionSubmissionResult,
    ExternalEventResult,
    ObservationResult,
    PolicyDecisionResult,
    TimelinePoint,
    ValidationResult,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_NATIVE_LOCK = threading.Lock()
_NATIVE_MODULE: Any | None = None
# 与 Rust 引擎的 MAX_QUEUE_CAPACITY 一致；实际并发由所属工作流的 .env 参数决定。
MAX_QUEUE_CAPACITY = 100


def _load_native_module() -> Any:
    """首次使用时加载原生扩展，并核对构建时嵌入的契约版本。"""
    global _NATIVE_MODULE
    if _NATIVE_MODULE is not None:
        return _NATIVE_MODULE

    with _NATIVE_LOCK:
        if _NATIVE_MODULE is not None:
            return _NATIVE_MODULE

        try:
            import fight_engine_native
        except ImportError as exc:
            raise RuntimeError(
                "无法加载 Rust FightEngine 原生扩展；请执行 ./setup.sh 安装当前工作树的项目依赖。"
            ) from exc

        native_version = int(fight_engine_native.CONTRACT_VERSION)
        if native_version != SIDECAR_CONTRACT_VERSION:
            raise RuntimeError(
                "实际加载的 FightEngine 原生扩展契约版本与当前 Python schema 不匹配："
                f"expected={SIDECAR_CONTRACT_VERSION}, native={native_version}。"
                "请使用当前工作树重新构建 Rust Python 扩展。"
            )
        _NATIVE_MODULE = fight_engine_native
        return _NATIVE_MODULE


def _build_death_token(result: ExternalEventResult) -> dict[str, object]:
    """复用 Rust 死亡 token 装配；首次可见步号由转换回放层补充。"""
    encoded = _load_native_module().death_token_packet(result.timestamp, result.history_cursor)
    return pickle.loads(encoded)


class InProcessEngine:
    """一个 Rust 引擎管理多个独立队列，供转换、验证和回放共同使用。

    同一进程的多个 Python 线程可共享引擎；每个任务持有自己的 backend。
    capacity 限制同时存活的队列数，满载立即报错，调度和任务等待由调用方负责。
    """

    def __init__(self, job_tag: str, *, capacity: int = MAX_QUEUE_CAPACITY):
        if (
            isinstance(capacity, bool) or not isinstance(capacity, int)
            or not 1 <= capacity <= MAX_QUEUE_CAPACITY
        ):
            raise ValueError(f"capacity must be an integer between 1 and {MAX_QUEUE_CAPACITY}")
        native = _load_native_module()
        self.job_tag = job_tag
        self.capacity = capacity
        self._engine = native.SimulationEngine(str(_PROJECT_ROOT), job_tag, capacity)

    def _require_engine(self) -> Any:
        if self._engine is None:
            raise RuntimeError("in-process engine is closed")
        return self._engine

    @property
    def active_count(self) -> int:
        return int(self._require_engine().active_count)

    def create_backend(
        self,
        *,
        max_history: int | None,
        actual_base_gcd: float | None = None,
        fight_remaining: float | None = None,
        initial_timestamp: float | None = None,
    ) -> InProcessBackend:
        """创建队列；显式传 None 保留完整历史，整数只限制记录量，不限制回放时长。"""
        return InProcessBackend(
            self.job_tag, engine=self, max_history=max_history,
            actual_base_gcd=actual_base_gcd, fight_remaining=fight_remaining,
            initial_timestamp=initial_timestamp,
        )

    def close(self) -> None:
        engine, self._engine = self._engine, None
        if engine is not None:
            engine.close()

    def __enter__(self) -> Self:
        self._require_engine()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class InProcessBackend:
    """共享 Rust 引擎中的一个队列句柄；只释放队列，引擎由调用方管理。"""

    def __init__(
        self,
        job_tag: str,
        *,
        actual_base_gcd: float | None = None,
        fight_remaining: float | None = None,
        max_history: int | None = None,
        initial_timestamp: float | None = None,
        engine: InProcessEngine,
    ):
        if engine is None:
            raise ValueError("backend requires a shared engine")
        if engine.job_tag != job_tag:
            raise ValueError("backend job_tag must match the shared engine")
        self.job_tag = job_tag
        self._max_history = max_history
        self._initial_timestamp = float(initial_timestamp or 0.0)
        self._session: Any | None = None
        self._lifecycle_lock = threading.RLock()
        self._history_markers: list[dict[str, object]] = []
        self._engine = engine
        self._closed = False
        try:
            self.init(
                actual_base_gcd=actual_base_gcd,
                fight_remaining=fight_remaining,
                max_history=max_history,
                initial_timestamp=initial_timestamp,
            )
        except BaseException:
            self.close()
            raise

    def _require_session(self) -> Any:
        if self._closed or self._engine._engine is None:
            raise RuntimeError("in-process backend is closed")
        if self._session is None:
            raise RuntimeError("in-process backend is not initialized")
        return self._session

    @property
    def queue_id(self) -> int:
        return int(self._require_session().id)

    @property
    def history_markers(self) -> tuple[dict[str, object], ...]:
        """当前轨迹已接受的死亡事件；游标始终按完整动作历史计数。"""
        with self._lifecycle_lock:
            self._require_session()
            return tuple(dict(marker) for marker in self._history_markers)

    def init(
        self,
        *,
        actual_base_gcd: float | None = None,
        fight_remaining: float | None = None,
        max_history: int | None = None,
        initial_timestamp: float | None = None,
    ) -> float:
        """初始化或重置 Rust 状态机，省略的历史上限和起始时刻沿用既有值。"""
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("in-process backend is closed")
            history = self._max_history if max_history is None else max_history
            initial = self._initial_timestamp if initial_timestamp is None else float(initial_timestamp)
            if self._session is None:
                self._session = self._engine._require_engine().create_session(
                    history, actual_base_gcd, initial, fight_remaining,
                )
            else:
                self._require_session().reset(history, actual_base_gcd, initial, fight_remaining)
            self._max_history = history
            self._initial_timestamp = initial
            self._history_markers.clear()
            return initial

    def advance_to(self, timestamp: float) -> TimelinePoint:
        return TimelinePoint(*self._require_session().advance_to(float(timestamp)))

    def submit_action(
        self,
        timestamp: float,
        action: str,
        *,
        actual_cast_seconds: float | None = None,
    ) -> ActionSubmissionResult:
        return ActionSubmissionResult(
            *self._require_session().submit_action(float(timestamp), action, actual_cast_seconds),
        )

    def validate_at(self, timestamp: float, action: str) -> ValidationResult:
        return ValidationResult(*self._require_session().validate_at(float(timestamp), action))

    def record_policy_action(
        self,
        timestamp: float,
        action: str,
        next_observation_timestamp: float,
    ) -> PolicyDecisionResult:
        return PolicyDecisionResult(
            *self._require_session().record_policy_action(
                float(timestamp), action, float(next_observation_timestamp),
            ),
        )

    def apply_external_events(self, events) -> ExternalEventResult:
        """同刻事实一次交给 session；Rust 完成整批校验、入队和统一推进。"""
        batch = []
        kinds = []
        for event in events:
            kind = str(event["event_kind"])
            kinds.append(kind)
            batch.append((
                float(event["timestamp"]), kind, event.get("value"),
                event.get("target_count"), event.get("remaining_seconds"),
                event.get("mp"),
            ))
        with self._lifecycle_lock:
            accepted, reason, timestamp, next_event, token, history_cursor = (
                self._require_session().apply_external_events(batch)
            )
            receipt = ExternalEventResult(
                accepted=accepted,
                reason=reason,
                event_kind=kinds[0] if len(kinds) == 1 else "scene_batch",
                timestamp=timestamp,
                next_scheduled_event_time=next_event,
                token=token,
                history_cursor=history_cursor,
            )
            if receipt.accepted and receipt.token == "<dead/>":
                self._history_markers.append({
                    "token": receipt.token,
                    "timestamp": receipt.timestamp,
                    "after_history_length": receipt.history_cursor,
                })
            return receipt

    def observe_at(
        self,
        timestamp: float,
        *,
        format: str = "seconds",
        next_observation_timestamp: float | None = None,
    ) -> ObservationResult:
        if format not in ("vector", "seconds", "gcd"):
            raise ValueError(f"unsupported observe format: {format}; supported=gcd, seconds, vector")
        if format == "vector":
            if next_observation_timestamp is None:
                raise ValueError("observe_at(format='vector') 需要 next_observation_timestamp")
            if next_observation_timestamp < timestamp:
                raise ValueError(
                    "next_observation_timestamp must not precede observation timestamp"
                )

        response = self._require_session().observe_at(float(timestamp), format, next_observation_timestamp)
        return self._decode_observation(response, format)

    def observe_vector_delta_at(
        self,
        timestamp: float,
        *,
        next_observation_timestamp: float,
        after_sequence: int,
    ) -> ObservationResult:
        """编译内部读取新增冻结历史；公开完整观测仍使用 observe_at。"""
        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int) or after_sequence < 0:
            raise ValueError("after_sequence must be a non-negative integer")
        response = self._require_session().observe_vector_delta_at(
            float(timestamp), float(next_observation_timestamp), after_sequence,
        )
        return self._decode_observation(response, "vector")

    def _decode_observation(self, response, format: str) -> ObservationResult:
        timestamp, next_event, encoded = response
        # Rust 计算与编码时释放 GIL；数据只来自当前加载的原生扩展。
        python_context = pickle.loads(encoded)
        return ObservationResult(
            timestamp=timestamp,
            format=format,
            next_scheduled_event_time=next_event,
            context=python_context,
        )

    def replay_policy_sequence(
        self,
        requests: list[dict[str, object]],
        facts: list[dict[str, object]],
        *,
        fight_duration: float,
        policy_action_key: str,
    ) -> list[dict[str, object]]:
        """一次原生调用推进完整请求序列，逐帧只传新增历史。"""
        response = self._require_session().replay_policy_sequence(
            json.dumps(requests, allow_nan=False, separators=(",", ":")),
            json.dumps(facts, allow_nan=False, separators=(",", ":")),
            float(fight_duration), policy_action_key,
        )
        payload = self._decode_observation(response, "vector").context
        return payload["observations"]

    def replay_policy_sequence_packet(
        self,
        requests_json: str,
        facts_json: str,
        *,
        fight_duration: float,
        policy_action_key: str,
    ) -> bytes:
        """返回原生基础值数据包，供进程间转发；不在当前 Python 进程展开观测对象。"""
        response = self._require_session().replay_policy_sequence(
            requests_json, facts_json, float(fight_duration), policy_action_key,
        )
        return response[2]

    def close(self) -> None:
        """立即从引擎注销队列并释放历史；关闭共享队列不影响其余任务。"""
        with self._lifecycle_lock:
            self._closed = True
            if self._session is not None:
                self._session.close()
                self._session = None

    def statistics(self) -> dict[str, int | float]:
        """读取容量诊断，不复制战斗历史，也不推进时钟。"""
        fields = (
            "timestamp", "action_history_count", "policy_history_count",
            "pending_event_count", "queue_entry_count", "pending_settlement_count",
        )
        return dict(zip(fields, self._require_session().statistics(), strict=True))

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
