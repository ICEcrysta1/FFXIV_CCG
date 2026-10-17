"""Boss 停手窗口检测。"""

from __future__ import annotations

from ..config.constants import DOWNTIME_BOUNDARY_MARGIN
from .extraction import _collect_begincasts, _get_ability_id, _take_begincast


def _collect_party_damage_actions(
    report: dict[str, object], own_actions: list[dict[str, object]], *,
    source_id: int, fight_start: float, fight_end: float, match_tolerance_seconds: float,
) -> list[dict[str, object]] | None:
    """从本场玩家对敌人的实际直伤恢复攻击时刻，不用目标职业技能表识别队友。"""
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
        if (not isinstance(event, dict) or event.get("sourceID") not in players
                or event.get("fight", fight_id) != fight_id):
            continue
        timestamp = float(event["timestamp"]) / 1000.0
        if fight_start <= timestamp <= fight_end:
            events_by_player[event["sourceID"]].append(event)

    own_requests = {(int(action["skill_id"]), float(action["timestamp"])):
                    _request_timestamp(action) for action in own_actions}
    result = []
    for actor_id, events in events_by_player.items():
        begincasts = _collect_begincasts(events, actor_id)
        cursors = {}
        casts_by_packet = {}
        for event in sorted(events, key=lambda item: item["timestamp"]):
            if event.get("type") != "cast":
                continue
            skill_id = _get_ability_id(event)
            timestamp = float(event["timestamp"]) / 1000.0
            begin = _take_begincast(begincasts.get(skill_id), cursors, skill_id,
                                   effect_timestamp=timestamp,
                                   match_tolerance_seconds=match_tolerance_seconds)
            request = begin[0] if begin is not None else timestamp
            if actor_id == source_id:
                request = own_requests.get((skill_id, timestamp), request)
            cast = (timestamp, request)
            if event.get("packetID") is not None:
                casts_by_packet[event["packetID"], skill_id] = cast

        for event in events:
            # calculateddamage 是同次攻击的提前计算，不是额外命中；DoT tick 也不能证明可选中。
            if (event.get("type") != "damage" or event.get("tick")
                    or event.get("targetID") not in targets or float(event.get("amount", 0)) <= 0):
                continue
            timestamp = float(event["timestamp"]) / 1000.0
            skill_id = _get_ability_id(event)
            cast = casts_by_packet.get((event.get("packetID"), skill_id))
            # 自动攻击的正伤同样证明目标可攻击；无配对 cast 时仅以真实命中作为请求上界。
            result.append({"timestamp": timestamp, "request_timestamp": cast[1] if cast else timestamp,
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
    """通过伤害动作间隔检测 Boss 不可输出窗口，正式载荷使用全队实际命中。

    窗口端点就是"停手前的最后一击"和"恢复后的第一击"。两者都发生在 Boss 可选中时，
    因此停手区间是**开区间**，且两端的参照时刻不同：

    - 起点用最后一击的**生效时刻**——停手发生在那次命中结算之后。
    - 终点用第一个恢复动作的**请求时刻**——状态机的合法性校验发生在请求时刻，
      读条技能的请求时刻天然早于生效时刻，用生效时刻会把恢复动作圈进窗口。

    边界额外内缩 `boundary_margin`，覆盖端点动作被状态机推迟执行的最坏情况。
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
        # 第一发伤害不一定来自最早起读的队友，右界取恢复输出段中最早的真实请求。
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
