"""Boss 停手窗口检测。"""

from __future__ import annotations

from ..config.constants import DOWNTIME_BOUNDARY_MARGIN
from .extraction import _collect_begincasts, _get_ability_id, _take_begincast


def _collect_party_cast_actions(
    report: dict[str, object], *, fight_start: float, fight_end: float,
    match_tolerance_seconds: float,
) -> list[dict[str, object]] | None:
    """收集本场玩家对敌方的 cast，恢复请求优先使用配对的 begincast。"""
    friendlies, enemies = report.get("friendlies"), report.get("enemies")
    fight_id = report.get("fight_id")
    if fight_id is None:
        fights = report.get("fights", [])
        if len(fights) == 1:
            fight_id = fights[0].get("id")
    # 旧的手工载荷没有完整原始角色表，保留原有单玩家动作退回路径。
    if not friendlies or not enemies or fight_id is None:
        return None

    def in_fight(actor):
        return any(member.get("id") == fight_id for member in actor.get("fights", []))

    players = {actor["id"] for actor in friendlies if in_fight(actor)
               and actor.get("type") not in {"Pet", "NPC", "Boss", "LimitBreak"}
               and actor.get("petOwner") is None}
    targets = {actor["id"] for actor in enemies if in_fight(actor)
               and actor.get("type") != "Pet" and actor.get("petOwner") is None}
    events_by_player = {actor_id: [] for actor_id in players}
    for event in report.get("events", []):
        if (not isinstance(event, dict) or event.get("type") not in {"cast", "begincast"}
                or event.get("sourceID") not in players
                or event.get("fight", fight_id) != fight_id):
            continue
        events_by_player[event["sourceID"]].append(event)

    result = []
    for actor_id, events in events_by_player.items():
        begincasts = _collect_begincasts(events, actor_id)
        cursors = {}
        for event in sorted(events, key=lambda item: item["timestamp"]):
            if event.get("type") != "cast" or event.get("targetID") not in targets:
                continue
            timestamp = float(event["timestamp"]) / 1000.0
            if not fight_start <= timestamp <= fight_end:
                continue
            skill_id = _get_ability_id(event)
            begin = _take_begincast(
                begincasts.get(skill_id), cursors, skill_id,
                effect_timestamp=timestamp, match_tolerance_seconds=match_tolerance_seconds,
            )
            # 左界仍取 cast；右界优先起读时刻，瞬发或缺少有效起读时退回 cast。
            result.append({"timestamp": timestamp, "request_timestamp": begin[0] if begin else timestamp,
                           "is_damaging": True})
    return sorted(result, key=lambda action: action["timestamp"])


def detect_downtime_windows(
    actions: list[dict[str, object]],
    *,
    fight_start: float,
    fight_end: float,
    gap_seconds: float,
    boundary_margin: float = DOWNTIME_BOUNDARY_MARGIN,
) -> list[dict[str, float]]:
    """通过攻击事件间隔检测 Boss 停手窗口。

    正式载荷左界取停手前最后一个 cast；右界取恢复段中最早的配对起读时刻，
    没有有效 begincast 的动作使用 cast，再以 boundary_margin 内缩窗口。
    缺少完整角色表的手工载荷沿用个人动作的请求时刻退回路径。
    """
    damaging_actions = sorted((action for action in actions if action["is_damaging"]),
                             key=lambda action: action["timestamp"])
    if not damaging_actions:
        return [
            _narrow_window(
                {"start": fight_start, "end": fight_end, "duration": fight_end - fight_start},
                boundary_margin,
            )
        ]

    phases = [[damaging_actions[0]]]
    for action in damaging_actions[1:]:
        if float(action["timestamp"]) - float(phases[-1][-1]["timestamp"]) > gap_seconds:
            phases.append([])
        phases[-1].append(action)

    windows: list[dict[str, float]] = []
    for previous_phase, next_phase in zip(phases, phases[1:]):
        previous = float(previous_phase[-1]["timestamp"])
        # 恢复段第一发完成的动作未必最早起读，右界取全队最早的有效请求。
        end = min(_request_timestamp(action) for action in next_phase)
        if end > previous:
            windows.append({"start": previous, "end": end, "duration": end - previous})

    first_damaging_timestamp = float(damaging_actions[0]["timestamp"])
    if first_damaging_timestamp - fight_start > gap_seconds:
        windows.insert(
            0,
            {
                "start": fight_start,
                "end": max(fight_start, min(_request_timestamp(action) for action in phases[0])),
                "duration": max(0.0, min(_request_timestamp(action) for action in phases[0]) - fight_start),
            },
        )

    last_damaging_timestamp = float(damaging_actions[-1]["timestamp"])
    if fight_end - last_damaging_timestamp > gap_seconds:
        windows.append(
            {
                "start": last_damaging_timestamp,
                "end": fight_end,
                "duration": fight_end - last_damaging_timestamp,
            }
        )

    if not windows:
        return []

    merged_windows = [windows[0]]
    for window in windows[1:]:
        previous = merged_windows[-1]
        if window["start"] <= previous["end"] + 1.0:
            previous["end"] = max(previous["end"], window["end"])
            previous["duration"] = previous["end"] - previous["start"]
        else:
            merged_windows.append(window)
    return [_narrow_window(window, boundary_margin) for window in merged_windows]


def _narrow_window(window: dict[str, float], margin: float) -> dict[str, float]:
    """把停手窗口收成开区间，端点动作仍算在可选中区间内。"""
    start = window["start"] + margin
    end = max(start, window["end"] - margin)
    return {"start": start, "end": end, "duration": end - start}


def _request_timestamp(action: dict[str, object]) -> float:
    """动作的请求时刻；解析层缺失时退回生效时刻。"""
    return float(action.get("request_timestamp", action["timestamp"]))
