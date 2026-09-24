"""训练样本回放 —— 逐步展开战斗并生成逐决策样本（绝对时间线版）。

状态机的推进、校验与结算由 C# FightEngine 独占，通过进程内后端直接调用；
本模块只保留决策逻辑：
请求时刻解析、场景事实提交、policy 动作合成与样本装配。

时间轴以日志为准：转换层已经把整场最早请求平移到 0；动作严格在归一化后的
请求时刻提交。调用方不钳制请求时间，也不按拒绝原因补推进。

转换路径只有三类场景事实进入状态机（Boss 可选中、目标数、团辅窗口），因为它们参与
内部数值结算；移动状态仍由输出层按场景上下文改写并应用滑步豁免，不进入转换状态机。
在线自回归、PPG 与 GRPO 的 `SceneTemplateProvider` 采用另一条口径：提交经过滑步豁免的
`movement_changed`，由状态机负责请求时移动合法性判断。
"""

from __future__ import annotations

from common.contracts import SCENE_EPSILON

from scripts.common.scene_state import SceneFactScheduler, rewrite_scene_player_state

from ..config.constants import _round_time
from ..scene.scene_context import build_scene_context_view, normalize_scene_context, resolve_anchor

OGCD_WAIT_ACTION_KEY = "ogcd_wait"

def build_training_samples(
    backend,
    skill_book,
    fight_payload: dict[str, object],
    *,
    policy_action_key: str = OGCD_WAIT_ACTION_KEY,
) -> dict[str, object]:
    """把一场战斗展开成逐决策训练样本。"""
    actions = fight_payload.get("actions", [])
    if not isinstance(actions, list) or not actions:
        raise ValueError("fight payload must contain at least one action")

    scene_context = normalize_scene_context(fight_payload.get("scene_context"))
    scheduler = SceneFactScheduler(scene_context)
    fight_duration = float(fight_payload.get("duration", 0.0))
    # 转换层已把最早请求归一化到 0。
    cursor = resolve_initial_timestamp(fight_payload)

    samples: list[dict[str, object]] = []
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
            cursor=cursor,
            next_request_offset=next_request_offset,
            fight_duration=fight_duration,
            step_index=next_step_index,
            source_step=source_step,
            resolved_sequence=resolved_sequence,
            policy_action_key=policy_action_key,
        )
        samples.append(sample)
        resolved_sequence.append(str(raw_action["action_key"]))
        next_step_index += 1

    return {
        "sample_schema_version": 7,
        "job_tag": str(fight_payload.get("job_tag", backend.job_tag)),
        "fight_id": str(fight_payload.get("fight_id", "unknown_fight")),
        "player": fight_payload.get("player"),
        "encounter": fight_payload.get("encounter"),
        "fight_scene_context": scene_context,
        "num_samples": len(samples),
        "resolved_sequence": resolved_sequence,
        "samples": samples,
    }


def _run_real_action(
    backend,
    scheduler: SceneFactScheduler,
    skill_book,
    raw_action: dict[str, object],
    *,
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
    context = _observe_canonical(
        backend,
        scheduler,
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

    sample = {
        "time_offset": _round_time(request_time),
        "label": {
            "action_key": action_key,
            "skill_id": skill.game_id,
            "skill_name": skill.name,
            "candidate_index": _find_candidate_index(context["candidate_skill_context"], action_key),
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
    return sample, effect_timestamp


def _build_ogcd_wait_sample(
    backend,
    scheduler: SceneFactScheduler,
    skill_book,
    *,
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
    context = _observe_canonical(
        backend,
        scheduler,
        timestamp=decision_time,
        next_observation_timestamp=observation_timestamp,
    )
    backend.record_policy_action(decision_time, policy_action_key, observation_timestamp)
    return (
        {
            "time_offset": _round_time(decision_time),
            "label": {
                "action_key": policy_action_key,
                "skill_id": 0,
                "skill_name": "",
                "candidate_index": _find_candidate_index(
                    context["candidate_skill_context"],
                    policy_action_key,
                ),
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
        },
        cursor,
    )


def _advance_to_with_facts(
    backend,
    scheduler: SceneFactScheduler,
    timestamp: float,
) -> tuple[float, float]:
    """提交所有不晚于 `timestamp` 的场景事实，再把时钟推进到该时刻。

    事实必须先提交：状态机拒绝过去事件，而 `apply_external_event` 自身会把
    时钟推进到事实时刻。返回 (请求时刻, 状态机当前时刻)。
    """
    for fact in scheduler.pop_facts_through(timestamp):
        backend.apply_external_event(
            fact.timestamp,
            fact.event_kind,
            value=fact.value,
            target_count=fact.target_count,
            remaining_seconds=fact.remaining_seconds,
        )
    return timestamp, backend.advance_to(timestamp).timestamp


def _observe_canonical(
    backend,
    scheduler: SceneFactScheduler,
    *,
    timestamp: float,
    next_observation_timestamp: float,
) -> dict[str, object]:
    """取一次完整模型输入，并按场景上下文改写 player 向量的场景字段。"""
    observation = backend.observe_at(
        timestamp,
        format="vector",
        next_observation_timestamp=next_observation_timestamp,
    )
    context = dict(observation.context)  # type: ignore[arg-type]
    rewrite_scene_player_state(
        context,
        observation_timestamp=timestamp,
        next_observation_timestamp=next_observation_timestamp,
        scene_state_at=scheduler.state_at,
    )
    context["scene_context"] = build_scene_context_view(scheduler.scene_context)
    return context


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


def _find_candidate_index(candidate_skill_context: list[dict[str, object]], action_key: str) -> int:
    """在候选技能上下文里找到与目标动作同源的候选下标。

    候选集必须覆盖全部已启用技能（含 policy 候选），缺失即抛错，
    避免训练标签静默错位污染数据。
    """
    for index, entry in enumerate(candidate_skill_context):
        if str(entry.get("skill_key")) == action_key:
            return index
    raise ValueError(f"action {action_key} is not present in candidate_skill_context")


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

    candidate_skill_count = len(context["candidate_skill_context"])
    candidate_state_count = len(context["candidate_state_context"]["tokens"])
    if candidate_skill_count != candidate_state_count:
        raise AssertionError(
            "candidate context mismatch: "
            f"step={current_step}, action={current_action_key}, "
            f"candidate_skill_count={candidate_skill_count}, candidate_state_count={candidate_state_count}"
        )


def _is_policy_entry(token: object) -> bool:
    """policy 控制动作不是游戏技能，raw skill id 固定为 0。"""
    if not isinstance(token, dict):
        return False
    skill_id = token.get("skill_id")
    return skill_id is not None and float(skill_id) == 0.0  # type: ignore[arg-type]
