"""战斗载荷装配：把原始动作组装成带 scene_context 的完整训练载荷。"""

from __future__ import annotations

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
from .downtime import detect_downtime_windows
from .extraction import detect_forced_movement_windows


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
    downtime_boundary_margin: float = DOWNTIME_BOUNDARY_MARGIN,
    target_count_window_context: dict[str, object] | None = None,
    raw_events: list[dict[str, object]] | None = None,
    skill_book=None,
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

    downtime_windows = detect_downtime_windows(
        fight_actions,
        fight_start=fight_start,
        fight_end=fight_end,
        gap_seconds=downtime_gap_seconds,
        boundary_margin=downtime_boundary_margin,
    )
    forced_movement_windows = detect_forced_movement_windows(fight_actions)
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
        "actions": annotated_actions,
    }
