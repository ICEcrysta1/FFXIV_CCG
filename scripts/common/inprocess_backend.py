"""通过 Python.NET 在当前 Python 进程中直接调用 C# 战斗状态机。"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Self

from common.contracts import SIDECAR_CONTRACT_VERSION
from scripts.common.cs_backend import (
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
    / "SidecarHost"
    / "bin"
    / _CONFIGURATION
    / "net10.0"
)
_RUNTIME_CONFIG = _DOTNET_OUTPUT / "SidecarHost.runtimeconfig.json"
_FIGHT_ENGINE_DLL = _DOTNET_OUTPUT / "FightEngine.dll"
_YAML_DOTNET_DLL = _DOTNET_OUTPUT / "YamlDotNet.dll"

_DOTNET_LOCK = threading.Lock()
_DOTNET_TYPES: tuple[Any, Any, Any, Any, Any] | None = None


def _load_dotnet_types() -> tuple[Any, Any, Any, Any, Any]:
    """只初始化一次 .NET 10，并加载 C# 状态机及其 YAML 依赖。"""
    global _DOTNET_TYPES
    if _DOTNET_TYPES is not None:
        return _DOTNET_TYPES

    with _DOTNET_LOCK:
        if _DOTNET_TYPES is not None:
            return _DOTNET_TYPES

        required_files = (_RUNTIME_CONFIG, _FIGHT_ENGINE_DLL, _YAML_DOTNET_DLL)
        missing_files = [path for path in required_files if not path.is_file()]
        if missing_files:
            missing = ", ".join(str(path) for path in missing_files)
            raise FileNotFoundError(
                f"C# 状态机运行文件缺失：{missing}；请先运行 "
                "dotnet build Combat.Sim/SidecarHost/SidecarHost.csproj"
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

            # SidecarHost 输出目录同时包含运行时配置与 FightEngine 的托管依赖；
            # 这里只加载程序集，不启动 SidecarHost 进程。
            clr.AddReference(str(_YAML_DOTNET_DLL))
            clr.AddReference(str(_FIGHT_ENGINE_DLL))
            clr.AddReference("System.Text.Json")

            from Combat.Sim.Config import SchemaConfigLoader
            from Combat.Sim.Facade import JobSimulator
            from Combat.Sim.Models.Timeline import ExternalCombatEvent
            from Combat.Sim.Policy import PolicySession
            from System.Text.Json import JsonSerializer
        except Exception as exc:
            raise RuntimeError(
                "无法在 Python 进程内加载 FightEngine；请确认当前工作树已构建 "
                "net10.0 SidecarHost，且本 Python 进程尚未加载其他版本的 .NET runtime。"
            ) from exc

        _DOTNET_TYPES = (
            JobSimulator,
            PolicySession,
            ExternalCombatEvent,
            SchemaConfigLoader,
            JsonSerializer,
        )
        return _DOTNET_TYPES


class InProcessBackend:
    """SidecarBackend 兼容接口的进程内实现；不创建子进程、不传输 JSON。"""

    def __init__(
        self,
        job_tag: str,
        *,
        actual_base_gcd: float | None = None,
        fight_remaining: float | None = None,
        max_history: int | None = None,
        initial_timestamp: float | None = None,
    ):
        self.job_tag = job_tag
        self._max_history = max_history
        self._initial_timestamp = float(initial_timestamp or 0.0)
        self._simulator: Any | None = None
        self._policy: Any | None = None
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

    def _types(self) -> tuple[Any, Any, Any, Any, Any]:
        return _load_dotnet_types()

    def _require_simulator(self) -> Any:
        if self._closed:
            raise RuntimeError("in-process backend is closed")
        if self._simulator is None:
            raise RuntimeError("in-process backend is not initialized")
        return self._simulator

    def init(
        self,
        *,
        actual_base_gcd: float | None = None,
        fight_remaining: float | None = None,
        max_history: int | None = None,
        initial_timestamp: float | None = None,
    ) -> float:
        """初始化或重置 C# 状态机，省略的历史上限和起始时刻沿用既有值。"""
        if self._closed:
            raise RuntimeError("in-process backend is closed")
        if max_history is not None:
            self._max_history = max_history
        if initial_timestamp is not None:
            self._initial_timestamp = float(initial_timestamp)

        job_simulator, policy_session, _, schema_loader, _ = self._types()
        try:
            assembly_version = int(schema_loader.AssemblySidecarContractVersion)
        except Exception as exc:
            self._simulator = None
            self._policy = None
            raise RuntimeError(
                "实际加载的 FightEngine DLL 缺少可验证的程序集契约版本；"
                "请使用当前工作树重新构建 SidecarHost。"
            ) from exc
        if assembly_version != SIDECAR_CONTRACT_VERSION:
            self._simulator = None
            self._policy = None
            raise RuntimeError(
                "实际加载的 FightEngine DLL 契约版本与当前 Python schema 不匹配："
                f"expected={SIDECAR_CONTRACT_VERSION}, dll={assembly_version}。"
                "请使用当前工作树重新构建 SidecarHost。"
            )

        simulator = job_simulator.Create(
            str(_PROJECT_ROOT),
            self.job_tag,
            actual_base_gcd,
            self._max_history,
            self._initial_timestamp,
            fight_remaining,
        )
        policy = policy_session.Create(str(_PROJECT_ROOT), simulator)
        self._simulator = simulator
        self._policy = policy
        return float(simulator.Time)

    @staticmethod
    def _optional(value: Any) -> float | None:
        return None if value is None else float(value)

    def advance_to(self, timestamp: float) -> TimelinePoint:
        simulator = self._require_simulator()
        simulator.AdvanceTo(float(timestamp))
        return TimelinePoint(
            timestamp=float(simulator.Time),
            next_scheduled_event_time=self._optional(simulator.GetNextScheduledEventTime()),
        )

    def submit_action(
        self,
        timestamp: float,
        action: str,
        *,
        actual_cast_seconds: float | None = None,
    ) -> ActionSubmissionResult:
        simulator = self._require_simulator()
        result = simulator.SubmitAction(float(timestamp), action, actual_cast_seconds)
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
        simulator = self._require_simulator()
        result = simulator.ValidateActionAt(float(timestamp), action)
        return ValidationResult(
            legal=bool(result.Ok),
            reason=str(result.Reason),
            timestamp=float(simulator.Time),
            next_scheduled_event_time=self._optional(simulator.GetNextScheduledEventTime()),
        )

    def record_policy_action(
        self,
        timestamp: float,
        action: str,
        next_observation_timestamp: float,
    ) -> PolicyDecisionResult:
        simulator = self._require_simulator()
        decision = self._policy.Record(
            simulator,
            float(timestamp),
            action,
            float(next_observation_timestamp),
        )
        return PolicyDecisionResult(
            action=str(decision.Action.Key),
            timestamp=float(decision.Timestamp),
            next_observation_timestamp=float(next_observation_timestamp),
            gcd_index=int(decision.GcdIndex),
        )

    def apply_external_event(
        self,
        timestamp: float,
        event_kind: str,
        *,
        value: bool | None = None,
        target_count: int | None = None,
        remaining_seconds: float | None = None,
    ) -> ExternalEventResult:
        simulator = self._require_simulator()
        _, _, external_event, _, _ = self._types()
        result = simulator.ApplyExternalEvent(
            external_event(
                float(timestamp),
                event_kind,
                value,
                target_count,
                remaining_seconds,
            )
        )
        return ExternalEventResult(
            accepted=bool(result.Accepted),
            reason=str(result.Reason),
            event_kind=event_kind,
            timestamp=float(result.Timestamp),
            next_scheduled_event_time=self._optional(simulator.GetNextScheduledEventTime()),
        )

    def observe_at(
        self,
        timestamp: float,
        *,
        format: str = "seconds",
        next_observation_timestamp: float | None = None,
    ) -> ObservationResult:
        simulator = self._require_simulator()
        if format not in ("vector", "seconds", "gcd"):
            raise ValueError(f"unsupported observe format: {format}; supported=gcd, seconds, vector")
        if format == "vector":
            if next_observation_timestamp is None:
                raise ValueError("observe_at(format='vector') 需要 next_observation_timestamp")
            if next_observation_timestamp < timestamp:
                raise ValueError(
                    "next_observation_timestamp must not precede observation timestamp"
                )

        simulator.ObserveAt(float(timestamp))
        if format == "vector":
            context = self._policy.BuildVectorContext(
                simulator,
                float(next_observation_timestamp),
            )
        else:
            context = simulator.FormatState(format)

        *_, json_serializer = self._types()
        # Vector 包含多层候选与历史容器；在 CLR 侧一次序列化后由 Python 解码，
        # 比逐字段跨 Python.NET 反射枚举快，且只发生在当前进程内。
        python_context = json.loads(str(json_serializer.Serialize(context)))
        return ObservationResult(
            timestamp=float(simulator.Time),
            format=format,
            next_scheduled_event_time=self._optional(simulator.GetNextScheduledEventTime()),
            context=python_context,
        )

    def close(self) -> None:
        """释放会话对象；CoreCLR 由 Python.NET 在当前进程内共享。"""
        self._closed = True
        self._policy = None
        self._simulator = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
