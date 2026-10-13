"""训练样本回放 —— 逐步展开战斗并生成逐决策样本（绝对时间线版）。

状态机的推进、校验与结算由 C# FightEngine 独占，通过进程内后端直接调用；
公开回放逐步提交请求与场景事实；编译回放把时间线与等待决策整批交给 C#，
本模块统一负责标签装配、场景字段合成与历史引用恢复。

时间轴以日志为准：转换层已经把整场最早请求平移到 0；动作严格在归一化后的
请求时刻提交。调用方不钳制请求时间，也不按拒绝原因补推进。

转换与实时复用同一场景执行视图和同刻批次；移动合法性由状态机读取事实。
只有停机 ETA/剩余由输出层按每段冻结时间纯合成，日志读条覆盖保持原职责。
"""

from __future__ import annotations

from dataclasses import asdict

from common.contracts import SCENE_EPSILON
from common.policy.data.schema import TRAINING_SAMPLE_SCHEMA_VERSION

from scripts.common.scene_state import SceneFactScheduler, rewrite_scene_player_state
from scripts.common.state_machine_types import ActionSubmissionResult

from ..config.constants import _round_time
from ..scene.scene_context import build_scene_context_view, normalize_scene_context, resolve_anchor
from .history_bank import HistoryPrefixInterner

OGCD_WAIT_ACTION_KEY = "ogcd_wait"


class _TrainingObserver:
    """转换专用观测：先合成新增状态的场景字段，再恢复完整历史引用。"""

    def __init__(self, backend, scheduler: SceneFactScheduler, *, incremental: bool):
        self._backend = backend
        self._scheduler = scheduler
        self._history = HistoryPrefixInterner() if incremental else None
        self._scene_view = build_scene_context_view(scheduler.scene_context) if incremental else None

    def observe(self, *, timestamp: float, next_observation_timestamp: float) -> dict[str, object]:
        if self._history is None:
            observation = self._backend.observe_at(
                timestamp, format="vector", next_observation_timestamp=next_observation_timestamp,
            )
        else:
            observation = self._backend.observe_vector_delta_at(
                timestamp, next_observation_timestamp=next_observation_timestamp,
                after_sequence=self._history.cursor,
            )
        return self.restore(observation.context)

    def restore(self, canonical: dict[str, object]) -> dict[str, object]:
        context = rewrite_scene_player_state(dict(canonical), scene_state_at=self._scheduler.state_at)
        if self._history is not None:
            self._history.append_delta(context)
            context["scene_context"] = self._scene_view
        else:
            context["scene_context"] = build_scene_context_view(self._scheduler.scene_context)
        return context


def build_training_samples(
    backend,
    skill_book,
    fight_payload: dict[str, object],
    *,
    policy_action_key: str = OGCD_WAIT_ACTION_KEY,
    deduplicate_history: bool = False,
) -> dict[str, object]:
    """把一场战斗展开成逐决策样本；编译侧可共享只读的冻结历史行。"""
    actions = fight_payload.get("actions", [])
    if not isinstance(actions, list) or not actions:
        raise ValueError("fight payload must contain at least one action")

    scene_context = normalize_scene_context(fight_payload.get("scene_context"))
    scheduler = SceneFactScheduler(scene_context)
    fight_duration = float(fight_payload.get("duration", 0.0))
    # 转换层已把最早请求归一化到 0。
    cursor = resolve_initial_timestamp(fight_payload)
    scheduler.sync_through(backend, cursor)

    if deduplicate_history:
        samples, resolved_sequence = _collect_native_samples(
            backend, skill_book, fight_payload, scheduler,
            fight_duration=fight_duration, policy_action_key=policy_action_key,
        )
        return _training_payload(backend, fight_payload, scene_context, samples, resolved_sequence)

    samples: list[dict[str, object]] = []
    observer = _TrainingObserver(backend, scheduler, incremental=deduplicate_history)
    resolved_sequence: list[str] = []
    next_step_index = 1

    for source_step, raw_action in enumerate(actions, 1):
        if not isinstance(raw_action, dict):
            raise ValueError(f"action at source step {source_step} must be an object")
        next_request_offset = (
            _request_time_offset(actions[source_step]) if source_step < len(actions) else None
        )

        wait_sample, cursor = _build_ogcd_wait_sample(
            backend,
            scheduler,
            skill_book,
            observer=observer,
            cursor=cursor,
            next_action=raw_action,
            action_request_offset=_request_time_offset(raw_action),
            fight_duration=fight_duration,
            policy_action_key=policy_action_key,
        )
        if wait_sample is not None:
            wait_sample["step"] = next_step_index
            wait_sample["source_step"] = source_step
            samples.append(wait_sample)
            resolved_sequence.append(policy_action_key)
            next_step_index += 1

        sample, cursor = _run_real_action(
            backend,
            scheduler,
            skill_book,
            raw_action,
            observer=observer,
            cursor=cursor,
            next_request_offset=next_request_offset,
            fight_duration=fight_duration,
            step_index=next_step_index,
            source_step=source_step,
            resolved_sequence=resolved_sequence,
            policy_action_key=policy_action_key,
        )
        sample["step"] = next_step_index
        sample["source_step"] = source_step
        samples.append(sample)
        resolved_sequence.append(str(raw_action["action_key"]))
        next_step_index += 1

    return _training_payload(backend, fight_payload, scene_context, samples, resolved_sequence)


def _training_payload(
    backend, fight_payload: dict[str, object], scene_context: dict[str, object],
    samples: list[dict[str, object]], resolved_sequence: list[str],
) -> dict[str, object]:
    return {
        "sample_schema_version": TRAINING_SAMPLE_SCHEMA_VERSION,
        "job_tag": str(fight_payload.get("job_tag", backend.job_tag)),
        "fight_id": str(fight_payload.get("fight_id", "unknown_fight")),
        "player": fight_payload.get("player"),
        "encounter": fight_payload.get("encounter"),
        "ranking": fight_payload.get("ranking"),
        "annotation_status": fight_payload.get("annotation_status", "unannotated"),
        "fight_scene_context": scene_context,
        "num_samples": len(samples),
        "resolved_sequence": resolved_sequence,
        "samples": samples,
    }


def _collect_native_samples(
    backend, skill_book, fight_payload: dict[str, object], scheduler: SceneFactScheduler,
    *, fight_duration: float, policy_action_key: str,
) -> tuple[list[dict[str, object]], list[str]]:
    """托管执行只负责时间线和决策；样本、标签及场景合成仍留在转换模块。"""
    actions = fight_payload["actions"]
    start = resolve_initial_timestamp(fight_payload)
    requests = []
    for source_step, action in enumerate(actions, 1):
        if not isinstance(action, dict):
            raise ValueError(f"action at source step {source_step} must be an object")
        cast_seconds = action.get("actual_cast_seconds")
        requests.append({
            "timestamp": _request_time_offset(action),
            "skill_key": str(action["action_key"]),
            "actual_cast_seconds": None if cast_seconds is None else float(cast_seconds),
        })
    facts = [asdict(fact) for fact in scheduler.lookup.facts() if fact.timestamp > start]
    frames = backend.replay_policy_sequence(
        requests, facts, fight_duration=fight_duration, policy_action_key=policy_action_key,
    )
    observer = _TrainingObserver(backend, scheduler, incremental=True)
    samples: list[dict[str, object]] = []
    resolved_sequence: list[str] = []
    for frame in frames:
        source_index = int(frame["source_index"])
        raw_action = actions[source_index]
        timestamp = float(frame["timestamp"])
        context = observer.restore(frame["context"])
        step_index = len(samples) + 1
        if frame["is_wait"]:
            action_key = policy_action_key
            sample = _wait_sample(context, scheduler, timestamp, fight_duration, policy_action_key)
        else:
            action_key = str(raw_action["action_key"])
            _assert_no_label_leak(
                context, expected_history_sequence=resolved_sequence, policy_action_key=policy_action_key,
                current_action_key=action_key, current_step=step_index,
            )
            raw_submission = frame["submission"]
            submission = ActionSubmissionResult(
                accepted=bool(raw_submission["accepted"]), queued=bool(raw_submission["queued"]),
                reason=str(raw_submission["reason"]), action_instance_id=None, request_timestamp=timestamp,
                accepted_timestamp=raw_submission["accepted_timestamp"],
                effect_timestamp=raw_submission["effect_timestamp"], next_scheduled_event_time=None,
            )
            if not submission.accepted:
                raise ValueError(
                    "logged action was rejected by the state machine: "
                    f"step={step_index}, action={action_key}, request_time={timestamp:.4f}, "
                    f"reason={submission.reason}"
                )
            sample = _real_action_sample(
                raw_action, skill_book.get(action_key), context, timestamp, submission, fight_duration,
            )
        sample["step"] = step_index
        sample["source_step"] = source_index + 1
        samples.append(sample)
        resolved_sequence.append(action_key)
    return samples, resolved_sequence


def _run_real_action(
    backend,
    scheduler: SceneFactScheduler,
    skill_book,
    raw_action: dict[str, object],
    *,
    observer: _TrainingObserver,
    cursor: float,
    next_request_offset: float | None,
    fight_duration: float,
    step_index: int,
    source_step: int,
    resolved_sequence: list[str],
    policy_action_key: str,
) -> tuple[dict[str, object], float]:
    """在日志请求时刻提交一条真实动作并装配对应样本。"""
    action_key = str(raw_action["action_key"])
    skill = skill_book.get(action_key)
    logged_request = _request_time_offset(raw_action)
    if logged_request < 0.0 - SCENE_EPSILON:
        raise ValueError(
            "normalized action request timestamp must be non-negative: "
            f"step={step_index}, action={action_key}, request_time={logged_request:.4f}"
        )
    request_time, _ = _advance_to_with_facts(backend, scheduler, logged_request)

    observation_timestamp = _resolve_next_observation(next_request_offset, request_time, fight_duration)
    context = observer.observe(
        timestamp=request_time,
        next_observation_timestamp=observation_timestamp,
    )
    _assert_no_label_leak(
        context,
        expected_history_sequence=resolved_sequence,
        policy_action_key=policy_action_key,
        current_action_key=action_key,
        current_step=step_index,
    )

    observed_cast_seconds = raw_action.get("actual_cast_seconds")
    submission = backend.submit_action(
        request_time,
        action_key,
        actual_cast_seconds=(
            None if observed_cast_seconds is None else float(observed_cast_seconds)
        ),
    )
    if not submission.accepted:
        raise ValueError(
            "logged action was rejected by the state machine: "
            f"step={step_index}, action={action_key}, request_time={request_time:.4f}, "
            f"reason={submission.reason}"
        )

    # 只把状态机声明的生效时刻留给下一轮判断是否需要 policy wait。这里不主动
    # 推进，否则预测生效时刻一旦晚于下一条日志请求，就会迫使整场时间轴后移。
    effect_timestamp = (
        request_time
        if submission.effect_timestamp is None
        else float(submission.effect_timestamp)
    )

    return _real_action_sample(raw_action, skill, context, request_time, submission, fight_duration), effect_timestamp


def _real_action_sample(raw_action, skill, context, request_time, submission, fight_duration):
    action_key = str(raw_action["action_key"])
    observed_cast_seconds = raw_action.get("actual_cast_seconds")
    return {
        "time_offset": _round_time(request_time),
        "label": {
            "action_key": action_key,
            "skill_id": skill.game_id,
            "skill_name": skill.name,
            "raw_event_index": raw_action.get("raw_event_index"),
            "quality_labels": list(raw_action.get("quality_labels", [])),
            "action_index": _find_action_index(context["action_keys"], action_key),
            "cast_time_seconds": None if observed_cast_seconds is None else float(observed_cast_seconds),
            "is_legal": True,
            "invalid_reason": "",
            "time_gap": round(float(raw_action.get("time_gap", 0.0)), 4),
            "anchor": str(raw_action.get("anchor", "combat")),
            "fight_remaining": _round_time(max(0.0, fight_duration - request_time)),
            "timing_retry_count": 0,
            "request_time_clamped": 0.0,
            "cast_timing_source": str(raw_action.get("cast_timing_source", "")),
            "request_order_adjustment_seconds": _round_time(
                float(raw_action.get("request_order_adjustment_seconds", 0.0))
            ),
            # 状态机与日志的时序偏差证据：接受时刻/生效时刻都来自状态机。
            "queued": bool(submission.queued),
            "accepted_offset": _offset_or_zero(submission.accepted_timestamp, request_time),
            "effect_offset": _offset_or_zero(submission.effect_timestamp, request_time),
        },
        "context": context,
    }


def _build_ogcd_wait_sample(
    backend,
    scheduler: SceneFactScheduler,
    skill_book,
    *,
    observer: _TrainingObserver,
    cursor: float,
    next_action: dict[str, object],
    action_request_offset: float,
    fight_duration: float,
    policy_action_key: str,
) -> tuple[dict[str, object] | None, float]:
    """在当前 weave 链结束处补一条 policy `ogcd_wait` 决策样本。

    只在"仍处于 GCD 窗口内、且下一条真实动作不是同一窗口内的 oGCD"时注入，
    把"停止继续织入"显式化。policy 动作不提交状态机，时钟也不推进。
    """
    if cursor >= action_request_offset - SCENE_EPSILON:
        return None, cursor

    decision_time, cursor = _advance_to_with_facts(backend, scheduler, cursor)
    probe = backend.observe_at(decision_time, format="seconds").context
    gcd_remaining = float(probe.get("gcd_remaining_seconds") or 0.0)
    if gcd_remaining <= SCENE_EPSILON:
        return None, cursor

    window_end = decision_time + gcd_remaining
    next_skill = skill_book.get(str(next_action["action_key"]))
    if next_skill.kind.value == "ogcd" and action_request_offset < window_end - SCENE_EPSILON:
        return None, cursor

    observation_timestamp = max(window_end, action_request_offset)
    context = observer.observe(
        timestamp=decision_time,
        next_observation_timestamp=observation_timestamp,
    )
    backend.record_policy_action(decision_time, policy_action_key, observation_timestamp)
    return _wait_sample(context, scheduler, decision_time, fight_duration, policy_action_key), cursor


def _wait_sample(context, scheduler, decision_time, fight_duration, policy_action_key):
    return {
        "time_offset": _round_time(decision_time),
        "label": {
            "action_key": policy_action_key,
            "skill_id": 0,
            "skill_name": "",
            "action_index": _find_action_index(context["action_keys"], policy_action_key),
            "cast_time_seconds": 0.0,
            "is_legal": True,
            "invalid_reason": "",
            "time_gap": 0.0,
            "anchor": resolve_anchor(decision_time, scheduler.scene_context),
            "fight_remaining": _round_time(max(0.0, fight_duration - decision_time)),
            "timing_retry_count": 0,
            "request_time_clamped": 0.0,
            "cast_timing_source": "",
        },
        "context": context,
    }


def _advance_to_with_facts(
    backend,
    scheduler: SceneFactScheduler,
    timestamp: float,
) -> tuple[float, float]:
    """提交所有不晚于 `timestamp` 的场景事实，再把时钟推进到该时刻。

    同刻事实全部入队后才统一推进。返回 (请求时刻, 状态机当前时刻)。
    """
    scheduler.sync_through(backend, timestamp)
    return timestamp, backend.advance_to(timestamp).timestamp


def _resolve_next_observation(
    next_request_offset: float | None,
    request_time: float,
    fight_duration: float,
) -> float:
    """下一次观测时刻：下一条动作的请求时刻，末条动作则取战斗结束。"""
    if next_request_offset is None:
        return max(request_time, fight_duration)
    return max(request_time, next_request_offset)


def resolve_initial_timestamp(fight_payload: dict[str, object]) -> float:
    """返回归一化后的初始时刻；合法转换载荷必须从 0 开始。"""
    actions = fight_payload.get("actions", [])
    if not isinstance(actions, list) or not actions:
        raise ValueError("fight payload must contain at least one action")
    initial = min(_request_time_offset(action) for action in actions)
    if abs(initial) > SCENE_EPSILON:
        raise ValueError(f"fight payload request timeline must start at 0, got {initial:.4f}")
    return 0.0


def _request_time_offset(action: dict[str, object]) -> float:
    return float(action.get("request_time_offset", action.get("time_offset", 0.0)))


def _offset_or_zero(timestamp: float | None, base: float) -> float:
    """状态机给出的时刻相对请求时刻的偏移；缺失时记 0。"""
    return 0.0 if timestamp is None else _round_time(float(timestamp) - base)


def _find_action_index(action_keys: list[str], action_key: str) -> int:
    """在固定动作输出词表中取得监督索引，缺失时显式失败。"""
    try:
        return action_keys.index(action_key)
    except ValueError as exc:
        raise ValueError(f"action {action_key} is not present in action_keys") from exc


def _assert_no_label_leak(
    context: dict[str, object],
    *,
    expected_history_sequence: list[str],
    policy_action_key: str,
    current_action_key: str,
    current_step: int,
) -> None:
    """验证当前样本输入只包含 step<t 的真实动作历史。

    policy 决策与真实动作共用同一条历史，但只有真实动作会被提交状态机，
    因此两侧都先剔除 policy 条目再对齐。动作在接受之后要等到生效时刻才写入
    历史，所以历史始终是已解析序列的**前缀**，不能用尾部对齐。
    """
    skill_history = context["skill_history_context"]
    assert isinstance(skill_history, list)
    exported_sequence = [
        str(token["skill_key"])
        for token in skill_history
        if not _is_policy_entry(token)
    ]
    expected_runtime_sequence = [
        action_key
        for action_key in expected_history_sequence
        if action_key != policy_action_key
    ]
    expected_sequence = expected_runtime_sequence[: len(exported_sequence)]
    if exported_sequence != expected_sequence:
        raise AssertionError(
            "label leak: exported skill_history_context is not a prefix of the resolved "
            f"pre-label sequence: step={current_step}, action={current_action_key}, "
            f"expected={expected_sequence}, actual={exported_sequence}"
        )

    state_history_count = len(context["state_history_context"]["tokens"])
    if state_history_count != len(skill_history):
        raise AssertionError(
            "label leak: exported state_history_context length does not match skill history: "
            f"step={current_step}, action={current_action_key}, "
            f"skill_count={len(skill_history)}, state_count={state_history_count}"
        )

    if len(context["current_state_context"]["tokens"]) != 1:
        raise AssertionError(
            "current request state must contain exactly one token: "
            f"step={current_step}, action={current_action_key}, "
        )


def _is_policy_entry(token: object) -> bool:
    """policy 控制动作不是游戏技能，raw skill id 固定为 0。"""
    if not isinstance(token, dict):
        return False
    skill_id = token.get("skill_id")
    return skill_id is not None and float(skill_id) == 0.0  # type: ignore[arg-type]
