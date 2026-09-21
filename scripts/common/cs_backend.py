"""C# 状态机后端客户端（SidecarHost 长驻进程，JSON Lines over stdin/stdout）。

协议是绝对时间的：动作提交、只读探测、policy 决策、外部事实和观测分别调用，
宿主只返回各自的轻量元数据，不返回状态镜像；完整模型输入只由
`observe_at(format="vector")` 返回。调用方自己持有当前时间线位置。

用法：
    with SidecarBackend(job_tag="black_mage") as backend:
        backend.submit_action(0.0, "fire_iii")
        backend.advance_to(3.0)
        context = backend.observe_at(3.0, format="vector",
                                     next_observation_timestamp=5.0).context
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from common.contracts import SIDECAR_CONTRACT_VERSION

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_SIDECAR_EXE = (
    _PROJECT_ROOT
    / "Combat.Sim"
    / "SidecarHost"
    / "bin"
    / os.environ.get("COMBAT_SIM_CONFIGURATION", "Debug")
    / "net10.0"
    / "SidecarHost.exe"
)


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


class SidecarBackend:
    """SidecarHost 长驻进程客户端（一次性 init，多阶段可重新 init 复用）。"""

    def __init__(
        self,
        job_tag: str,
        *,
        actual_base_gcd: float | None = None,
        fight_remaining: float | None = None,
        max_history: int | None = None,
        initial_timestamp: float | None = None,
    ):
        if not _SIDECAR_EXE.exists():
            raise FileNotFoundError(
                f"SidecarHost 未构建：{_SIDECAR_EXE}；请先运行 "
                "dotnet build Combat.Sim/SidecarHost/SidecarHost.csproj"
            )
        self._proc = subprocess.Popen(
            [str(_SIDECAR_EXE)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._seq = 0
        self.job_tag = job_tag
        # 多阶段转换会重复 init；省略参数时继续沿用构造时的值。
        self._max_history = max_history
        self._initial_timestamp = float(initial_timestamp or 0.0)
        try:
            self.init(
                actual_base_gcd=actual_base_gcd,
                fight_remaining=fight_remaining,
                max_history=max_history,
                initial_timestamp=initial_timestamp,
            )
        except BaseException:
            # 构造失败也必须回收子进程，否则 SidecarHost 变成孤儿进程；
            # 清理是尽力而为，close 自身的意外异常不得覆盖原始构造异常
            try:
                self.close()
            except BaseException as close_error:
                logger.warning(
                    "SidecarBackend 构造失败后 close 清理异常被抑制（原始异常继续抛出）",
                    exc_info=close_error,
                )
            raise

    def _call(self, op: str, **payload: object) -> dict[str, object]:
        self._seq += 1
        request = json.dumps({"op": op, "seq": self._seq, **payload}, ensure_ascii=False)
        self._proc.stdin.write(request + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError(f"SidecarHost 进程已退出（op={op}）")
        response = json.loads(line)
        if response.get("seq") != self._seq:
            raise RuntimeError(f"SidecarHost 响应序号错乱：期望 {self._seq}，实际 {response.get('seq')}")
        if not response.get("ok"):
            raise RuntimeError(f"SidecarHost {op} 失败: {response.get('error')}")
        return response

    # ---- 协议命令 ----

    def init(
        self,
        *,
        actual_base_gcd: float | None = None,
        fight_remaining: float | None = None,
        max_history: int | None = None,
        initial_timestamp: float | None = None,
    ) -> float:
        """初始化状态机并返回初始逻辑时刻；未显式传入的参数沿用既有值。"""
        if max_history is not None:
            self._max_history = max_history
        if initial_timestamp is not None:
            self._initial_timestamp = float(initial_timestamp)
        payload: dict[str, object] = {
            "job_tag": self.job_tag,
            "initial_timestamp": self._initial_timestamp,
        }
        if actual_base_gcd is not None:
            payload["actual_base_gcd"] = actual_base_gcd
        if fight_remaining is not None:
            payload["fight_remaining"] = fight_remaining
        if self._max_history is not None:
            payload["max_history"] = self._max_history
        response = self._call("init", **payload)
        actual_version = response.get("sidecar_contract_version")
        if actual_version != SIDECAR_CONTRACT_VERSION:
            raise RuntimeError(
                "SidecarHost DLL 与当前 Python 转换/回放契约不匹配："
                f"expected={SIDECAR_CONTRACT_VERSION}, actual={actual_version!r}。"
                "请先使用当前仓库重建 Combat.Sim/SidecarHost/SidecarHost.csproj，"
                "并确保新版 DLL 与 config/schema.yaml 来自同一版本。"
            )
        return float(response["timestamp"])

    @staticmethod
    def _optional(value: object) -> float | None:
        return None if value is None else float(value)  # type: ignore[arg-type]

    def advance_to(self, timestamp: float) -> TimelinePoint:
        """把逻辑时钟单调推进到绝对时刻并排空到期事件。"""
        response = self._call("advance_to", timestamp=float(timestamp))
        return TimelinePoint(
            timestamp=float(response["timestamp"]),
            next_scheduled_event_time=self._optional(response.get("next_scheduled_event_time")),
        )

    def submit_action(
        self,
        timestamp: float,
        action: str,
        *,
        actual_cast_seconds: float | None = None,
    ) -> ActionSubmissionResult:
        """在绝对请求时刻提交真实游戏动作；校验与接受在宿主内原子完成。"""
        payload: dict[str, object] = {"timestamp": float(timestamp), "action": action}
        if actual_cast_seconds is not None:
            payload["actual_cast_seconds"] = float(actual_cast_seconds)
        response = self._call("submit_action", **payload)
        return ActionSubmissionResult(
            accepted=bool(response["accepted"]),
            queued=bool(response["queued"]),
            reason=str(response.get("reason", "")),
            action_instance_id=(
                None if response.get("action_instance_id") is None else str(response["action_instance_id"])
            ),
            request_timestamp=float(response["request_timestamp"]),
            accepted_timestamp=self._optional(response.get("accepted_timestamp")),
            effect_timestamp=self._optional(response.get("effect_timestamp")),
            next_scheduled_event_time=self._optional(response.get("next_scheduled_event_time")),
        )

    def validate_at(self, timestamp: float, action: str) -> ValidationResult:
        """只读探测动作在指定时刻是否合法。

        探测会推进时钟且不接受过去时刻；先前请求仍被排队时会返回
        `action_queue_occupied`。仅用于合法性断言，不作为提交前置。
        """
        response = self._call("validate_at", timestamp=float(timestamp), action=action)
        return ValidationResult(
            legal=bool(response["legal"]),
            reason=str(response.get("reason", "")),
            timestamp=float(response["timestamp"]),
            next_scheduled_event_time=self._optional(response.get("next_scheduled_event_time")),
        )

    def record_policy_action(
        self,
        timestamp: float,
        action: str,
        next_observation_timestamp: float,
    ) -> PolicyDecisionResult:
        """记录一条不提交给游戏状态机的策略决策。

        `timestamp` 必须等于状态机当前时刻（先 `advance_to` 到决策时刻）。
        """
        response = self._call(
            "record_policy_action",
            timestamp=float(timestamp),
            action=action,
            next_observation_timestamp=float(next_observation_timestamp),
        )
        return PolicyDecisionResult(
            action=str(response["action"]),
            timestamp=float(response["timestamp"]),
            next_observation_timestamp=float(response["next_observation_timestamp"]),
            gcd_index=int(response["gcd_index"]),
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
        """提交带绝对时间戳的外部战斗事实。

        支持 Boss 可选中、移动、目标数和团辅窗口四类外部事实。自回归、PPG 与 GRPO
        调用方提交经过滑步豁免的 movement_changed；转换链路仍由输出层按 scene 上下文
        合成移动字段，不通过本客户端向状态机注入移动事实。
        """
        payload: dict[str, object] = {
            "timestamp": float(timestamp),
            "event_kind": event_kind,
        }
        if value is not None:
            payload["value"] = bool(value)
        if target_count is not None:
            payload["target_count"] = int(target_count)
        if remaining_seconds is not None:
            payload["remaining_seconds"] = float(remaining_seconds)
        response = self._call("apply_external_event", **payload)
        return ExternalEventResult(
            accepted=bool(response["accepted"]),
            reason=str(response.get("reason", "")),
            event_kind=str(response["event_kind"]),
            timestamp=float(response["timestamp"]),
            next_scheduled_event_time=self._optional(response.get("next_scheduled_event_time")),
        )

    def observe_at(
        self,
        timestamp: float,
        *,
        format: str = "seconds",
        next_observation_timestamp: float | None = None,
    ) -> ObservationResult:
        """在绝对时刻观测；`format="vector"` 才返回完整模型输入。

        `next_observation_timestamp` 只影响 policy 候选与 policy 历史的状态，
        真实技能候选的 after 由宿主按技能类型自行约定，与调用方无关。
        """
        payload: dict[str, object] = {"timestamp": float(timestamp), "format": format}
        if format == "vector":
            if next_observation_timestamp is None:
                raise ValueError("observe_at(format='vector') 需要 next_observation_timestamp")
            payload["next_observation_timestamp"] = float(next_observation_timestamp)
        response = self._call("observe_at", **payload)
        return ObservationResult(
            timestamp=float(response["timestamp"]),
            format=str(response["format"]),
            next_scheduled_event_time=self._optional(response.get("next_scheduled_event_time")),
            context=response["context"],
        )

    def close(self) -> None:
        try:
            self._call("close")
        except (BrokenPipeError, RuntimeError):
            pass
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()

    def __enter__(self) -> "SidecarBackend":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
