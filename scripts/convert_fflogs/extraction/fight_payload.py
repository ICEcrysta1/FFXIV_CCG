"""战斗载荷装配：把原始动作组装成带 scene_context 的完整训练载荷。"""

from __future__ import annotations

from ..config import MovementDetectionConfig
from ..config.constants import DOWNTIME_BOUNDARY_MARGIN, _round_time
from ..scene.scene_context import (
    build_forced_movement_context,
    build_raid_buff_window_context,
    build_target_count_window_context,
    build_scene_context,
    build_targetable_window_context,
    compute_downtime_total,
    resolve_anchor,
)
from .downtime import _collect_party_damage_actions, detect_downtime_windows
from .extraction import _get_ability_id, extract_gcd_cast_windows
from .movement import detect_forced_movement_windows


def build_output_fight_id(report_payload: dict[str, object], report_code: str) -> str:
    """构造稳定的 fight_id。"""
    raw_fight_id = report_payload.get("fight_id")
    if raw_fight_id is not None:
        return f"{report_code[:12]}_fight{raw_fight_id}"

    fights_meta = report_payload.get("fights", [])
    if isinstance(fights_meta, list) and len(fights_meta) == 1 and isinstance(fights_meta[0], dict):
        meta_fight_id = fights_meta[0].get("id")
        if meta_fight_id is not None:
            return f"{report_code[:12]}_fight{meta_fight_id}"

    return report_code[:12]


def _extract_lifecycle_events(fight_actions, raw_events, *, source_id, fight_start, fight_end):
    """死亡配对实际复活保护；缺少保护时用玩家首次重新起读或施法兜底。"""
    deaths = sorted({
        _round_time(float(event["timestamp"]) / 1000.0 - fight_start)
        for event in raw_events
        if event.get("type") == "death" and event.get("targetID") == source_id
        and fight_start <= float(event["timestamp"]) / 1000.0 <= fight_end
    })
    if not deaths:
        return []
    requests = {
        action["raw_event_index"]: float(action.get("request_timestamp", action["timestamp"]))
        for action in fight_actions if action.get("raw_event_index") is not None
    }
    protections, activities = [], []
    for index, event in enumerate(raw_events):
        protection = (event.get("type") == "applybuff" and event.get("targetID") == source_id
                      and _get_ability_id(event) == 1000418)
        activity = event.get("type") in {"begincast", "cast"} and event.get("sourceID") == source_id
        if not protection and not activity:
            continue
        timestamp = float(event["timestamp"]) / 1000.0
        if not fight_start <= timestamp <= fight_end:
            continue
        offset = _round_time(timestamp - fight_start)
        if protection:
            protections.append((offset, event))
        else:
            request = _round_time(requests.get(index, timestamp) - fight_start)
            activities.append((request, offset, event))

    result = []
    for index, death in enumerate(deaths):
        next_death = deaths[index + 1] if index + 1 < len(deaths) else float("inf")
        result.append({"timestamp": death, "event_kind": "player_death"})
        protection = min((item for item in protections if death <= item[0] < next_death),
                         key=lambda item: item[0], default=None)
        if protection is not None:
            timestamp, event = protection
            observed_at, resources = timestamp, event.get("targetResources")
        else:
            activity = min((item for item in activities
                            if death < item[0] < next_death and death < item[1] < next_death),
                           key=lambda item: item[0], default=None)
            if activity is None:
                continue
            timestamp, observed_at, event = activity
            resources = event.get("sourceResources")
        revival = {"timestamp": timestamp, "event_kind": "player_revived"}
        # 请求可能早于日志 cast；只采纳复活同刻的资源，不能把未来 MP 倒填到起读时刻。
        if timestamp == observed_at and isinstance(resources, dict):
            observed_mp, maximum = resources.get("mp"), resources.get("maxMP", 10000)
            if type(observed_mp) is int and type(maximum) is int and 0 <= observed_mp <= maximum:
                revival["mp"] = observed_mp
        result.append(revival)
    return sorted(result, key=lambda event: event["timestamp"])


def build_fight_payload(
    fight_actions: list[dict[str, object]],
    *,
    fight_id: str,
    player_name: str,
    encounter_name: str | None,
    report_code: str,
    source_id: int,
    job_tag: str,
    gcd_time: float,
    generated_at: str,
    downtime_gap_seconds: float,
    raid_buff_marker_keys: tuple[str, ...],
    raid_buff_window_duration: float,
    movement_detection: MovementDetectionConfig,
    downtime_boundary_margin: float = DOWNTIME_BOUNDARY_MARGIN,
    target_count_window_context: dict[str, object] | None = None,
    raw_events: list[dict[str, object]] | None = None,
    raw_report: dict[str, object] | None = None,
    skill_book=None,
    ranking: dict[str, object] | None = None,
    annotation_status: str = "unannotated",
) -> dict[str, object]:
    """把一场完整战斗动作转成以最早请求为 0 的训练输入载荷。"""
    # 请求时刻可能因开怪预读落在首个 cast 之前。整场只做一次仿射平移，动作、
    # 场景事实和战斗终点共用同一原点；持续时间与任意两个事件的间隔保持不变。
    fight_start = min(
        float(action.get("request_timestamp", action["timestamp"]))
        for action in fight_actions
    )
    fight_end = float(fight_actions[-1]["timestamp"])
    duration = fight_end - fight_start

    party_actions = _collect_party_damage_actions(
        raw_report, fight_actions, source_id=source_id, fight_start=fight_start,
        fight_end=fight_end, match_tolerance_seconds=downtime_boundary_margin,
    ) if raw_report is not None else None
    downtime_windows = detect_downtime_windows(
        fight_actions if party_actions is None else party_actions,
        fight_start=fight_start,
        fight_end=fight_end,
        gap_seconds=downtime_gap_seconds,
        boundary_margin=downtime_boundary_margin,
    )
    forced_movement_windows = detect_forced_movement_windows(
        raw_events or [], source_id=source_id, actual_base_gcd=gcd_time,
        movement_detection=movement_detection, fight_start=fight_start, fight_end=fight_end,
        hardcast_windows=extract_gcd_cast_windows(
            raw_events or [], fight_actions, source_id=source_id, skill_book=skill_book,
        ) if skill_book is not None else [],
    )
    targetable_window_context = build_targetable_window_context(
        fight_start,
        fight_end,
        downtime_windows,
    )
    if target_count_window_context is None and raw_events is not None and skill_book is not None:
        target_count_window_context = build_target_count_window_context(
            raw_events,
            source_id=source_id,
            skill_book=skill_book,
            fight_start=fight_start,
            fight_end=fight_end,
        )
    scene_context = build_scene_context(
        targetable_window_context=targetable_window_context,
        forced_movement_context=build_forced_movement_context(
            forced_movement_windows,
            fight_start=fight_start,
        ),
        raid_buff_window_context=build_raid_buff_window_context(
            fight_actions,
            fight_start=fight_start,
            window_duration=raid_buff_window_duration,
            marker_keys=raid_buff_marker_keys,
        ),
        target_count_window_context=target_count_window_context,
    )

    annotated_actions: list[dict[str, object]] = []
    previous_timestamp = fight_start
    for action in fight_actions:
        timestamp = float(action["timestamp"])
        time_offset = timestamp - fight_start
        time_gap = timestamp - previous_timestamp
        previous_timestamp = timestamp
        request_timestamp = float(action.get("request_timestamp", timestamp))

        annotated_actions.append(
            {
                "time_offset": _round_time(time_offset),
                "request_time_offset": _round_time(request_timestamp - fight_start),
                "action_key": action["action_key"],
                "skill_id": int(action["skill_id"]),
                "raw_event_index": action.get("raw_event_index"),
                "quality_labels": list(action.get("quality_labels", [])),
                "skill_name": action["skill_name"],
                "actual_cast_seconds": _round_time(float(action.get("actual_cast_seconds", 0.0))),
                "actual_is_instant": bool(action.get("actual_is_instant", False)),
                "cast_timing_source": str(action.get("cast_timing_source", "")),
                "request_order_adjustment_seconds": _round_time(
                    float(action.get("request_order_adjustment_seconds", 0.0))
                ),
                "time_gap": _round_time(max(0.0, time_gap)),
                "anchor": resolve_anchor(time_offset, scene_context),
                "fight_remaining": _round_time(max(0.0, fight_end - timestamp)),
                "forced_move": bool(action.get("forced_move", False)),
                "instant_move": bool(action.get("instant_move", False)),
                "moved": bool(action.get("moved", False)),
            }
        )

    downtime_total = compute_downtime_total(targetable_window_context)

    return {
        "data_schema_version": 4,
        "job_tag": job_tag,
        "fight_id": fight_id,
        "report_code": report_code,
        "source_id": source_id,
        "player": player_name,
        "encounter": encounter_name,
        "date": generated_at,
        "duration": _round_time(duration),
        "downtime_total": downtime_total,
        "downtime_pct": _round_time((downtime_total / duration) * 100.0) if duration > 0 else 0.0,
        "num_actions": len(annotated_actions),
        "gcd_time": _round_time(gcd_time),
        "scene_context": scene_context,
        "external_events": _extract_lifecycle_events(
            fight_actions, raw_events or [], source_id=source_id, fight_start=fight_start, fight_end=fight_end,
        ),
        "ranking": ranking,
        "annotation_status": annotation_status,
        "actions": annotated_actions,
    }
