"""通过 Python.NET 在当前 Python 进程中直接调用 C# 战斗状态机。"""

from __future__ import annotations

import json
import os
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
_CONFIGURATION = os.environ.get("COMBAT_SIM_CONFIGURATION", "Debug")
_DOTNET_OUTPUT = (
    _PROJECT_ROOT
    / "Combat.Sim"
    / "PythonBridge"
    / "bin"
    / _CONFIGURATION
    / "net10.0"
)
_RUNTIME_CONFIG = _DOTNET_OUTPUT / "FightEngine.PythonBridge.runtimeconfig.json"
_FIGHT_ENGINE_DLL = _DOTNET_OUTPUT / "FightEngine.dll"
_YAML_DOTNET_DLL = _DOTNET_OUTPUT / "YamlDotNet.dll"
_PYTHON_BRIDGE_DLL = _DOTNET_OUTPUT / "FightEngine.PythonBridge.dll"

_DOTNET_LOCK = threading.Lock()
_DOTNET_TYPES: tuple[Any, Any, Any] | None = None
# 与 C# SimulationEngine.MaxCapacity 一致；实际并发由所属工作流的 .env 参数决定。
MAX_QUEUE_CAPACITY = 100


def _load_dotnet_types() -> tuple[Any, Any, Any]:
    """只初始化一次 .NET 10，并加载 C# 状态机及其 YAML 依赖。"""
    global _DOTNET_TYPES
    if _DOTNET_TYPES is not None:
        return _DOTNET_TYPES

    with _DOTNET_LOCK:
        if _DOTNET_TYPES is not None:
            return _DOTNET_TYPES

        required_files = (
            _RUNTIME_CONFIG,
            _FIGHT_ENGINE_DLL,
            _YAML_DOTNET_DLL,
            _PYTHON_BRIDGE_DLL,
        )
        missing_files = [path for path in required_files if not path.is_file()]
        if missing_files:
            missing = ", ".join(str(path) for path in missing_files)
            raise FileNotFoundError(
                f"C# 状态机运行文件缺失：{missing}；"
                "请执行 ./setup.sh 安装项目依赖，再运行 "
                "dotnet build Combat.Sim/PythonBridge/PythonBridge.csproj。"
            )

        try:
            from pythonnet import get_runtime_info, load

            runtime_info = get_runtime_info()
            if runtime_info is None:
                load("coreclr", runtime_config=str(_RUNTIME_CONFIG))
            elif getattr(runtime_info, "kind", None) != "CoreCLR":
                raise RuntimeError(
                    "当前 Python 进程已经初始化了非 CoreCLR 的 .NET runtime，"
                    "无法再加载 FightEngine net10.0。"
                )

            import clr

            # PythonBridge 输出目录包含运行时配置与 FightEngine 的托管依赖。
            clr.AddReference(str(_YAML_DOTNET_DLL))
            clr.AddReference(str(_FIGHT_ENGINE_DLL))
            clr.AddReference(str(_PYTHON_BRIDGE_DLL))

            from Combat.Sim.Config import SchemaConfigLoader
            from Combat.Sim.Models.Timeline import ExternalCombatEvent
            from Combat.Sim.PythonBridge import ContextPacketEncoder
            from Combat.Sim.Sessions import SimulationEngine
        except Exception as exc:
            raise RuntimeError(
                "无法在 Python 进程内加载 FightEngine/PythonBridge；请确认两个程序集来自当前工作树，"
                "且本 Python 进程尚未加载其他版本的 .NET runtime。"
            ) from exc

        assembly_version = int(SchemaConfigLoader.AssemblySidecarContractVersion)
        if assembly_version != SIDECAR_CONTRACT_VERSION:
            raise RuntimeError(
                "实际加载的 FightEngine DLL 契约版本与当前 Python schema 不匹配："
                f"expected={SIDECAR_CONTRACT_VERSION}, dll={assembly_version}。"
                "请使用当前工作树重新构建 PythonBridge。"
            )
        _DOTNET_TYPES = (SimulationEngine, ExternalCombatEvent, ContextPacketEncoder)
        return _DOTNET_TYPES


def _build_death_token(result: ExternalEventResult) -> dict[str, object]:
    """复用 C# 死亡 token 装配；首次可见步号由转换回放层补充。"""
    *_, context_packet_encoder = _load_dotnet_types()
    from Combat.Sim.Outputs.TokenBuilders import DeathTokenBuilder

    encoded = context_packet_encoder.Encode(DeathTokenBuilder.Build(result.timestamp, result.history_cursor))
    with memoryview(encoded) as buffer:
        return pickle.loads(buffer)


class InProcessEngine:
    """一个 C# 引擎管理多个独立队列，供转换、验证和回放共同使用。

    同一进程的多个 Python 线程可共享引擎；每个任务持有自己的 backend。
    capacity 限制同时存活的队列数，满载立即报错，调度和任务等待由调用方负责。
    """

    def __init__(self, job_tag: str, *, capacity: int = MAX_QUEUE_CAPACITY):
        if (
            isinstance(capacity, bool) or not isinstance(capacity, int)
            or not 1 <= capacity <= MAX_QUEUE_CAPACITY
        ):
            raise ValueError(f"capacity must be an integer between 1 and {MAX_QUEUE_CAPACITY}")
        engine_type, _, _ = _load_dotnet_types()
        self.job_tag = job_tag
        self.capacity = capacity
        self._engine = engine_type(str(_PROJECT_ROOT), job_tag, capacity)

    def _require_engine(self) -> Any:
        if self._engine is None:
            raise RuntimeError("in-process engine is closed")
        return self._engine

    @property
    def active_count(self) -> int:
        return int(self._require_engine().ActiveCount)

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
            engine.Dispose()

    def __enter__(self) -> Self:
        self._require_engine()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class InProcessBackend:
    """共享 C# 引擎中的一个队列句柄；只释放队列，引擎由调用方管理。"""

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

    def _types(self) -> tuple[Any, Any, Any]:
        return _load_dotnet_types()

    def _require_session(self) -> Any:
        if self._closed or self._engine._engine is None:
            raise RuntimeError("in-process backend is closed")
        if self._session is None:
            raise RuntimeError("in-process backend is not initialized")
        return self._session

    @property
    def queue_id(self) -> int:
        return int(self._require_session().Id)

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
        """初始化或重置 C# 状态机，省略的历史上限和起始时刻沿用既有值。"""
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("in-process backend is closed")
            history = self._max_history if max_history is None else max_history
            initial = self._initial_timestamp if initial_timestamp is None else float(initial_timestamp)
            if self._session is None:
                self._session = self._engine._require_engine().CreateSession(
                    history, actual_base_gcd, initial, fight_remaining,
                )
            else:
                self._require_session().Reset(history, actual_base_gcd, initial, fight_remaining)
            self._max_history = history
            self._initial_timestamp = initial
            self._history_markers.clear()
            return initial

    @staticmethod
    def _optional(value: Any) -> float | None:
        return None if value is None else float(value)

    def advance_to(self, timestamp: float) -> TimelinePoint:
        result = self._require_session().AdvanceTo(float(timestamp))
        return TimelinePoint(
            timestamp=float(result.Timestamp),
            next_scheduled_event_time=self._optional(result.NextScheduledEventTime),
        )

    def submit_action(
        self,
        timestamp: float,
        action: str,
        *,
        actual_cast_seconds: float | None = None,
    ) -> ActionSubmissionResult:
        result = self._require_session().SubmitAction(float(timestamp), action, actual_cast_seconds).Value
        action_id = result.ActionInstanceId
        return ActionSubmissionResult(
            accepted=bool(result.Accepted),
            queued=bool(result.Queued),
            reason=str(result.Reason),
            action_instance_id=None if action_id is None else str(action_id),
            request_timestamp=float(result.RequestTimestamp),
            accepted_timestamp=self._optional(result.AcceptedTimestamp),
            effect_timestamp=self._optional(result.EffectTimestamp),
            next_scheduled_event_time=self._optional(result.NextScheduledEventTime),
        )

    def validate_at(self, timestamp: float, action: str) -> ValidationResult:
        response = self._require_session().ValidateActionAt(float(timestamp), action)
        result = response.Value
        return ValidationResult(
            legal=bool(result.Ok),
            reason=str(result.Reason),
            timestamp=float(response.Timestamp),
            next_scheduled_event_time=self._optional(response.NextScheduledEventTime),
        )

    def record_policy_action(
        self,
        timestamp: float,
        action: str,
        next_observation_timestamp: float,
    ) -> PolicyDecisionResult:
        decision = self._require_session().RecordPolicyAction(
            float(timestamp),
            action,
            float(next_observation_timestamp),
        ).Value
        return PolicyDecisionResult(
            action=str(decision.Action.Key),
            timestamp=float(decision.Timestamp),
            next_observation_timestamp=float(next_observation_timestamp),
            gcd_index=int(decision.GcdIndex),
        )

    def apply_external_events(self, events) -> ExternalEventResult:
        """同刻事实一次交给 session；C# 完成整批校验、入队和统一推进。"""
        _, external_event, _ = self._types()
        from System.Collections.Generic import List

        batch = List[external_event]()
        kinds = []
        for event in events:
            kind = str(event["event_kind"])
            kinds.append(kind)
            batch.Add(external_event(
                float(event["timestamp"]), kind, event.get("value"),
                event.get("target_count"), event.get("remaining_seconds"),
                event.get("mp"),
            ))
        with self._lifecycle_lock:
            response = self._require_session().ApplyExternalEvents(batch)
            result = response.Value
            receipt = ExternalEventResult(
                accepted=bool(result.Accepted),
                reason=str(result.Reason),
                event_kind=kinds[0] if len(kinds) == 1 else "scene_batch",
                timestamp=float(result.Timestamp),
                next_scheduled_event_time=self._optional(response.NextScheduledEventTime),
                token=None if result.Token is None else str(result.Token),
                history_cursor=None if result.HistoryCursor is None else int(result.HistoryCursor),
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

        response = self._require_session().ObserveAt(float(timestamp), format, next_observation_timestamp)
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
        response = self._require_session().ObserveVectorDeltaAt(
            float(timestamp), float(next_observation_timestamp), after_sequence,
        )
        return self._decode_observation(response, "vector")

    def _decode_observation(self, response, format: str) -> ObservationResult:
        *_, context_packet_encoder = self._types()
        # C# 编码不持有 GIL；借用托管数组缓冲区，交给标准库批量还原原生容器。
        # 数据只来自当前加载的桥接程序集，不接受文件或外部 pickle 输入。
        encoded = context_packet_encoder.Encode(response.Value)
        with memoryview(encoded) as buffer:
            python_context = pickle.loads(buffer)
        return ObservationResult(
            timestamp=float(response.Timestamp),
            format=format,
            next_scheduled_event_time=self._optional(response.NextScheduledEventTime),
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
        """一次托管调用推进完整请求序列，逐帧只传新增历史。"""
        response = self._require_session().ReplayPolicySequence(
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
        """返回托管基础值数据包，供进程间转发；不在当前 Python 进程展开观测对象。"""
        response = self._require_session().ReplayPolicySequence(
            requests_json, facts_json, float(fight_duration), policy_action_key,
        )
        *_, context_packet_encoder = self._types()
        encoded = context_packet_encoder.Encode(response.Value)
        with memoryview(encoded) as buffer:
            return bytes(buffer)

    def close(self) -> None:
        """立即从引擎注销队列并释放历史；关闭共享队列不影响其余任务。"""
        with self._lifecycle_lock:
            self._closed = True
            if self._session is not None:
                self._session.Dispose()
                self._session = None

    def statistics(self) -> dict[str, int | float]:
        """读取容量诊断，不复制战斗历史，也不推进时钟。"""
        stats = self._require_session().GetStatistics()
        return {
            "timestamp": float(stats.Timestamp),
            "action_history_count": int(stats.ActionHistoryCount),
            "policy_history_count": int(stats.PolicyHistoryCount),
            "pending_event_count": int(stats.PendingEventCount),
            "queue_entry_count": int(stats.QueueEntryCount),
            "pending_settlement_count": int(stats.PendingSettlementCount),
        }

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
