"""FFLogs 事件提取：动作识别、请求时刻解析、移动标注、GCD 检测。"""

from __future__ import annotations

import statistics

from collections import Counter

from common.contracts import SCENE_EPSILON, SLIDECAST_WINDOW_SECONDS

from ..config import GcdDetectionConfig
from ..config.constants import MOVE_DIST_THRESHOLD, MOVEMENT_MERGE_GAP, POTION_BUFF_ID, POTION_SKILL_ID, _round_time
from ..utils import merge_timestamps_to_windows

# 开怪预读的 begincast 会被战斗窗口裁掉：实测首个 cast 落在开怪后 0~800ms。
# 落在该窗口内且缺少 begincast 的硬读条按技能表读条时长回拨请求时刻；
# 其余缺少 begincast 的硬读条一律按瞬发化（三连/迅速）处理。
_PREPULL_WINDOW_SECONDS = 1.5


def _get_ability_id(event: dict[str, object]) -> int:
    """统一提取 FFLogs 事件里的技能 ID。"""
    if "abilityGameID" in event:
        return int(event["abilityGameID"])
    ability = event.get("ability", {})
    return int(ability.get("guid", 0)) if isinstance(ability, dict) else 0


def _get_ability_name(event: dict[str, object]) -> str:
    """统一提取 FFLogs 事件里的技能名。"""
    ability = event.get("ability", {})
    if isinstance(ability, dict):
        return str(ability.get("name", "?"))
    return "?"


def extract_supported_actions(
    report_payload: dict[str, object],
    source_id: int,
    skill_book,
    *,
    actual_base_gcd: float,
    skill_table_base_gcd: float,
    action_queue_window_seconds: float,
) -> tuple[list[dict[str, object]], Counter[tuple[int, str]]]:
    """从原始事件中提取技能表可识别的动作，并解析每个动作的请求时刻与读条时长。

    FFLogs 的 ``cast`` 时刻是服务器结算时刻。完整硬读条直接以配对的
    ``begincast`` 作为请求时刻；缺少 ``begincast`` 的战斗中动作按瞬发处理。
    只有开怪首个硬读条需要按技能表推断被战斗窗口裁掉的预读请求。
    """
    events = report_payload.get("events", [])
    actions, ignored_skill_counts = _collect_actions(events, source_id, skill_book)
    if not actions:
        return actions, ignored_skill_counts

    actions.sort(key=lambda action: action["timestamp"])
    _resolve_request_timing(
        actions,
        _collect_begincasts(events, source_id),
        actual_base_gcd=actual_base_gcd,
        skill_table_base_gcd=skill_table_base_gcd,
        prepull_cutoff=_prepull_cutoff(report_payload),
        begincast_match_tolerance_seconds=action_queue_window_seconds,
    )
    _stabilize_request_order(
        actions,
        action_queue_window_seconds=action_queue_window_seconds,
    )
    annotate_action_movement(actions)
    return actions, ignored_skill_counts


def _prepull_cutoff(report_payload: dict[str, object]) -> float:
    """战斗开始后仍可能出现预读的窗口末端绝对秒。

    事件时间戳以整份 report 为原点，首个 cast 通常就落在战斗起点上；
    预读动作的 begincast 发生在开怪前并被战斗窗口裁掉，因此只能靠
    "紧邻开怪 + 硬读条 + 无 begincast" 三个条件推断它是预读。
    """
    fights = report_payload.get("fights", [])
    if isinstance(fights, list) and fights and isinstance(fights[0], dict):
        start_time = fights[0].get("start_time")
        if isinstance(start_time, (int, float)):
            return float(start_time) / 1000.0 + _PREPULL_WINDOW_SECONDS
    return float("inf")


def _collect_actions(
    events: list[object],
    source_id: int,
    skill_book,
) -> tuple[list[dict[str, object]], Counter[tuple[int, str]]]:
    """把支持的 cast/buff 事件翻译成动作条目；``timestamp`` 为服务器结算时刻。"""
    actions: list[dict[str, object]] = []
    ignored_skill_counts: Counter[tuple[int, str]] = Counter()

    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("sourceID") != source_id:
            continue

        event_type = event.get("type")
        skill = None

        if event_type == "applybuff" and _get_ability_id(event) == POTION_BUFF_ID:
            if POTION_SKILL_ID in skill_book:
                skill = skill_book.get(POTION_SKILL_ID)
        elif event_type == "cast":
            raw_name = _get_ability_name(event)
            if raw_name in {"攻击", "Attack"} or event.get("melee"):
                continue
            ability_id = _get_ability_id(event)
            if ability_id in skill_book:
                skill = skill_book.get(ability_id)
            else:
                ignored_skill_counts[(ability_id, raw_name)] += 1
                continue
        else:
            continue

        if skill is None or not skill.enabled:
            continue

        source_resources = event.get("sourceResources", {})
        if not isinstance(source_resources, dict):
            source_resources = {}

        actions.append(
            {
                "timestamp": float(event["timestamp"]) / 1000.0,
                "action_key": skill.key,
                "skill_id": skill.game_id,
                "skill_name": skill.name,
                "raw_skill_name": _get_ability_name(event),
                "cast_time": float(skill.cast_time),
                "is_damaging": bool(skill.potency > 0 or skill.dot_potency > 0),
                "x": float(source_resources.get("x", 0.0)),
                "y": float(source_resources.get("y", 0.0)),
            }
        )

    return actions, ignored_skill_counts


def _collect_begincasts(
    events: list[object],
    source_id: int,
) -> dict[int, list[tuple[float, float]]]:
    """按技能 ID 收集读条开始时刻与实际读条时长。

    ``begincast`` 不携带 ``packetID``，无法与 ``cast`` 直接配对；这里改为按
    技能分组后时序邻近消费：每个 cast 取走其之前最后一个尚未配对的 begincast。
    同技能不会重叠读条，因此该配对在正常日志里唯一。
    """
    begincasts: dict[int, list[tuple[float, float]]] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("type") != "begincast" or event.get("sourceID") != source_id:
            continue
        ability_id = _get_ability_id(event)
        timestamp = float(event["timestamp"]) / 1000.0
        duration = float(event.get("duration", 0.0)) / 1000.0
        begincasts.setdefault(ability_id, []).append((timestamp, duration))

    for entries in begincasts.values():
        entries.sort(key=lambda entry: entry[0])
    return begincasts


def _resolve_request_timing(
    actions: list[dict[str, object]],
    begincasts: dict[int, list[tuple[float, float]]],
    *,
    actual_base_gcd: float,
    skill_table_base_gcd: float,
    prepull_cutoff: float,
    begincast_match_tolerance_seconds: float,
) -> None:
    """为每个动作写入 ``request_timestamp`` / ``actual_cast_seconds`` / ``cast_timing_source``。

    完整硬读条的 ``begincast`` 就是日志能够提供的真实请求时刻，直接使用它，
    不再为了让状态机生效事件精确贴合 ``cast`` 而反推请求。日志与游戏结算之间
    的毫秒级偏差由状态机容量一动作队列吸收。

    开怪首个硬读条是唯一例外：它的 ``begincast`` 发生在战斗窗口之前，原始事件
    已经将其裁掉，因此只能用实测 GCD 缩放后的读条时长与游戏滑步窗口回拨。

    ``cast_timing_source`` 记录读条时长的来源，避免推断值被当成日志权威事实：
    ``begincast_duration``（日志直读）、``instant_skill``（技能表瞬发）、
    ``prepull_estimated``（预读按实测 GCD 缩放回退）、
    ``instant_cast_inferred``（三连/迅速瞬发化推断）。
    """
    cursors: dict[int, int] = {}
    for index, action in enumerate(actions):
        skill_id = int(action["skill_id"])
        effect_timestamp = float(action["timestamp"])
        paired = _take_begincast(
            begincasts.get(skill_id),
            cursors,
            skill_id,
            effect_timestamp=effect_timestamp,
            match_tolerance_seconds=begincast_match_tolerance_seconds,
        )
        if paired is not None:
            request_timestamp = paired[0]
            cast_seconds = paired[1]
            source = "begincast_duration"
        else:
            nominal_cast = float(action["cast_time"])
            if nominal_cast <= SCENE_EPSILON:
                cast_seconds = 0.0
                source = "instant_skill"
            elif index == 0 and effect_timestamp <= prepull_cutoff:
                cast_seconds = _scale_nominal_cast(
                    nominal_cast,
                    actual_base_gcd=actual_base_gcd,
                    skill_table_base_gcd=skill_table_base_gcd,
                )
                source = "prepull_estimated"
            else:
                cast_seconds = 0.0
                source = "instant_cast_inferred"

            request_lead = (
                max(0.0, cast_seconds - SLIDECAST_WINDOW_SECONDS)
                if source == "prepull_estimated"
                else 0.0
            )
            request_timestamp = effect_timestamp - request_lead

        action["actual_cast_seconds"] = _round_time(cast_seconds)
        action["request_timestamp"] = _round_time(request_timestamp)
        action["cast_timing_source"] = source


def _take_begincast(
    entries: list[tuple[float, float]] | None,
    cursors: dict[int, int],
    skill_id: int,
    *,
    effect_timestamp: float,
    match_tolerance_seconds: float,
) -> tuple[float, float] | None:
    """消费该 cast 之前且能按读条时长闭环的最后一个 begincast。

    取消的读条没有对应 cast，会残留在同技能队列里。只有按游戏滑步窗口推算的
    结算时刻与实际 cast 相差不超过动作队列窗口时，才视为同一次读条。
    """
    if not entries:
        return None
    cursor = cursors.get(skill_id, 0)
    matched: tuple[float, float] | None = None
    while cursor < len(entries) and entries[cursor][0] <= effect_timestamp + SCENE_EPSILON:
        candidate = entries[cursor]
        expected_effect = candidate[0] + max(
            0.0,
            candidate[1] - SLIDECAST_WINDOW_SECONDS,
        )
        if abs(effect_timestamp - expected_effect) <= match_tolerance_seconds + SCENE_EPSILON:
            matched = candidate
        cursor += 1
    cursors[skill_id] = cursor
    return matched


def _stabilize_request_order(
    actions: list[dict[str, object]],
    *,
    action_queue_window_seconds: float,
) -> None:
    """消除队列窗口内由服务器事件排序造成的毫秒级请求倒序。

    FFLogs 的瞬发效果事件偶尔会比下一条硬读条的 ``begincast`` 晚几毫秒，
    但动作序列仍明确表示瞬发动作先发生。这里保持该顺序，把后一个请求抬到
    前一个请求的同一时刻；随后由状态机队列处理锁。超过配置队列窗口的倒序
    不是网络抖动，直接报错，禁止把真实时序问题静默压平。
    """
    if action_queue_window_seconds < 0.0:
        raise ValueError("action_queue_window_seconds must be non-negative")

    previous_request: float | None = None
    for index, action in enumerate(actions):
        request = float(action["request_timestamp"])
        adjustment = 0.0
        if previous_request is not None and request < previous_request - SCENE_EPSILON:
            adjustment = previous_request - request
            if (
                adjustment > action_queue_window_seconds + SCENE_EPSILON
            ):
                raise ValueError(
                    "inferred action request order exceeds action queue window: "
                    f"index={index}, action={action['action_key']}, "
                    f"previous={previous_request:.4f}, request={request:.4f}, "
                    f"delta={adjustment:.4f}, window={action_queue_window_seconds:.4f}"
                )
            request = previous_request
            action["request_timestamp"] = _round_time(request)
        action["request_order_adjustment_seconds"] = _round_time(adjustment)
        previous_request = request


def _scale_nominal_cast(
    nominal_cast: float,
    *,
    actual_base_gcd: float,
    skill_table_base_gcd: float,
) -> float:
    """按实测基础 GCD 缩放技能表标称读条时长（对照状态机的 ScaleNominalGcdSeconds）。"""
    if skill_table_base_gcd <= 0:
        return nominal_cast
    return nominal_cast * actual_base_gcd / skill_table_base_gcd


def annotate_action_movement(actions: list[dict[str, object]]) -> None:
    """按前后坐标变化给动作打移动标记。

    如果动作已经补充了 ``actual_cast_seconds``，这里优先使用动态真实读条时长；
    否则回退到技能表静态 ``cast_time``。
    """
    if not actions:
        return

    actions[0]["moved"] = False
    actions[0]["forced_move"] = False
    actions[0]["instant_move"] = False

    for index in range(1, len(actions)):
        previous = actions[index - 1]
        current = actions[index]
        dx = float(current["x"]) - float(previous["x"])
        dy = float(current["y"]) - float(previous["y"])
        distance = (dx * dx + dy * dy) ** 0.5
        moved = distance > MOVE_DIST_THRESHOLD
        previous_cast_time = float(previous.get("actual_cast_seconds", previous["cast_time"]))

        current["moved"] = moved
        current["forced_move"] = moved and previous_cast_time > SLIDECAST_WINDOW_SECONDS
        current["instant_move"] = moved and previous_cast_time <= SLIDECAST_WINDOW_SECONDS


def detect_forced_movement_windows(actions: list[dict[str, object]]) -> list[tuple[float, float]]:
    """把强制移动动作合并成原始时间窗口。

    当前转换链路把 FFLogs ``cast`` 事件的 timestamp 统一视为动作生效时刻。
    因此当某个动作被标成 ``forced_move`` 时，真正允许开始滑步的时刻应落在
    前一个读条动作的生效前 0.5 秒，而不是旧语义里的“起读条时刻 + 读条时长 - 0.5”。
    """
    move_timestamps: list[float] = []
    for index, action in enumerate(actions):
        if not action.get("forced_move") or index == 0:
            continue
        previous = actions[index - 1]
        prev_cast_time = float(previous.get("actual_cast_seconds", previous["cast_time"]))
        if prev_cast_time <= SLIDECAST_WINDOW_SECONDS:
            continue
        # 在 cast=生效时刻 语义下，滑步开始时刻 = 生效前 0.5s。
        movement_time = float(previous["timestamp"]) - SLIDECAST_WINDOW_SECONDS
        move_timestamps.append(movement_time)
    return merge_timestamps_to_windows(move_timestamps, gap_seconds=MOVEMENT_MERGE_GAP)


def detect_gcd_from_logs(
    events: list[dict[str, object]],
    source_id: int,
    *,
    gcd_detection: GcdDetectionConfig,
    haste_multiplier: float | None = None,
    skill_table_base_gcd: float = 2.5,
) -> float:
    """按职业配置的探针技能与排除 Buff 估算实际基础 GCD。"""
    fire_iv_timestamps: list[int] = []
    fire_iv_begincast_timestamps: list[int] = []
    fire_iv_begincast_durations: list[tuple[int, int]] = []
    for event in events:
        if event.get("sourceID") != source_id:
            continue
        if _get_ability_id(event) == gcd_detection.probe_skill_game_id:
            if event.get("type") == "cast":
                fire_iv_timestamps.append(int(event["timestamp"]))
            elif event.get("type") == "begincast":
                fire_iv_begincast_timestamps.append(int(event["timestamp"]))
                if event.get("duration") is not None:
                    fire_iv_begincast_durations.append(
                        (int(event["timestamp"]), int(event["duration"]))
                    )

    if (
        len(fire_iv_timestamps) < gcd_detection.min_probe_casts
        and len(fire_iv_begincast_timestamps) < gcd_detection.min_probe_casts
    ):
        return gcd_detection.fallback_seconds

    ley_lines_periods: list[list[int]] = []
    for event in events:
        if event.get("sourceID") != source_id or event.get("targetID") != source_id:
            continue
        if _get_ability_id(event) not in gcd_detection.haste_excluded_buff_game_ids:
            continue
        if event.get("type") == "applybuff":
            timestamp = int(event["timestamp"])
            duration = int(event.get("duration", 20000))
            ley_lines_periods.append([timestamp, timestamp + duration])

    if ley_lines_periods:
        ley_lines_periods.sort()
        merged_periods = [ley_lines_periods[0][:]]
        for start, end in ley_lines_periods[1:]:
            if start <= merged_periods[-1][1]:
                merged_periods[-1][1] = max(merged_periods[-1][1], end)
            else:
                merged_periods.append([start, end])
        ley_lines_periods = merged_periods

    # 请求时间现在优先取 begincast。若日志同时包含足够多的黑魔纹外样本，
    # 直接用这些未加速间隔估算基准 GCD，避免把带宽分箱后的急速间隔再除以
    # 0.85 后累积出几十毫秒的系统性偏大误差。只有外部样本不足时才回退到
    # 黑魔纹内的间隔反推路径（保留无外部窗口回放的兼容性）。
    normal_timestamps = (
        fire_iv_begincast_timestamps
        if len(fire_iv_begincast_timestamps) >= gcd_detection.min_probe_casts
        else fire_iv_timestamps
    )
    normal_gaps = _collect_gcd_gaps(
        normal_timestamps,
        ley_lines_periods,
        require_contained=False,
    )
    normal_mode_ms = _estimate_gcd_mode_ms(normal_gaps, gcd_detection)
    if gcd_detection.probe_skill_cast_time_seconds and fire_iv_begincast_durations:
        normal_duration_candidates = [
            duration / 1000.0 / gcd_detection.probe_skill_cast_time_seconds * skill_table_base_gcd
            for timestamp, duration in fire_iv_begincast_durations
            if not any(start <= timestamp <= end for start, end in ley_lines_periods)
        ]
        if len(normal_duration_candidates) > gcd_detection.min_gap_samples:
            return round(statistics.median(normal_duration_candidates), 3)
    if normal_mode_ms is not None and len(normal_gaps) > gcd_detection.min_gap_samples:
        return round(normal_mode_ms / 1000.0, 2)

    if haste_multiplier is not None:
        if haste_multiplier <= 0.0:
            raise ValueError("haste_multiplier must be > 0")
        haste_gaps = _collect_gcd_gaps(
            fire_iv_begincast_timestamps,
            ley_lines_periods,
            require_contained=True,
        )
        haste_mode_ms = _estimate_gcd_mode_ms(haste_gaps, gcd_detection)
        if haste_mode_ms is not None:
            return round(haste_mode_ms / 1000.0 / haste_multiplier, 2)

    gaps = _collect_gcd_gaps(
        fire_iv_timestamps,
        ley_lines_periods,
        require_contained=False,
    )
    mode_ms = _estimate_gcd_mode_ms(gaps, gcd_detection)
    if mode_ms is None:
        return gcd_detection.fallback_seconds
    return round(mode_ms / 1000.0, 2)


def _collect_gcd_gaps(
    timestamps: list[int],
    haste_periods: list[list[int]],
    *,
    require_contained: bool,
) -> list[int]:
    """收集探针相邻时间差并过滤 haste 窗口边界样本。"""
    ordered = sorted(timestamps)
    gaps: list[int] = []
    for index in range(1, len(ordered)):
        previous = ordered[index - 1]
        current = ordered[index]
        if require_contained:
            in_haste = any(
                previous >= period_start and current <= period_end
                for period_start, period_end in haste_periods
            )
        else:
            in_haste = any(
                previous < period_end and current > period_start
                for period_start, period_end in haste_periods
            )
        include_gap = in_haste if require_contained else not in_haste
        if include_gap:
            gaps.append(current - previous)
    return gaps


def _estimate_gcd_mode_ms(
    gaps: list[int],
    gcd_detection: GcdDetectionConfig,
) -> float | None:
    """从候选间隔中提取配置直方图的众数中心。"""
    if len(gaps) < gcd_detection.min_gap_samples:
        return None

    histogram_lower = gcd_detection.histogram_lower_ms
    histogram_upper = gcd_detection.histogram_upper_ms
    histogram_bin_width = gcd_detection.histogram_bin_width_ms
    histogram = [0] * ((histogram_upper - histogram_lower) // histogram_bin_width)
    for gap in gaps:
        if histogram_lower <= gap < histogram_upper:
            histogram[(gap - histogram_lower) // histogram_bin_width] += 1

    max_count = max(histogram, default=0)
    if max_count <= 0:
        return None

    mode_index = histogram.index(max_count)
    mode_start = histogram_lower + mode_index * histogram_bin_width
    mode_end = mode_start + histogram_bin_width
    # 直方图只负责选择主峰；主峰内部使用实际样本中位数，避免把 10ms
    # 分箱中心当成真实 GCD，在长回放中累积出可见的时间漂移。
    modal_samples = [gap for gap in gaps if mode_start <= gap < mode_end]
    return statistics.median(modal_samples) if modal_samples else mode_start
