"""Boss 停手窗口检测。"""

from __future__ import annotations

from .constants import DOWNTIME_BOUNDARY_MARGIN


def detect_downtime_windows(
    actions: list[dict[str, object]],
    *,
    fight_start: float,
    fight_end: float,
    gap_seconds: float,
    boundary_margin: float = DOWNTIME_BOUNDARY_MARGIN,
) -> list[dict[str, float]]:
    """通过伤害动作间隔检测 Boss 不可输出窗口。

    窗口端点就是"停手前的最后一击"和"恢复后的第一击"。两者都发生在 Boss 可选中时，
    因此停手区间是**开区间**，且两端的参照时刻不同：

    - 起点用最后一击的**生效时刻**——停手发生在那次命中结算之后。
    - 终点用第一个恢复动作的**请求时刻**——状态机的合法性校验发生在请求时刻，
      读条技能的请求时刻天然早于生效时刻，用生效时刻会把恢复动作圈进窗口。

    边界额外内缩 `boundary_margin`，覆盖端点动作被状态机推迟执行的最坏情况。
    """
    damaging_actions = [action for action in actions if action["is_damaging"]]
    if not damaging_actions:
        return [
            _narrow_window(
                {"start": fight_start, "end": fight_end, "duration": fight_end - fight_start},
                boundary_margin,
            )
        ]

    windows: list[dict[str, float]] = []
    for index in range(1, len(damaging_actions)):
        previous = float(damaging_actions[index - 1]["timestamp"])
        current = float(damaging_actions[index]["timestamp"])
        gap = current - previous
        if gap > gap_seconds:
            end = min(current, _request_timestamp(damaging_actions[index]))
            windows.append({"start": previous, "end": end, "duration": end - previous})

    first_damaging_timestamp = float(damaging_actions[0]["timestamp"])
    if first_damaging_timestamp - fight_start > gap_seconds:
        windows.insert(
            0,
            {
                "start": fight_start,
                "end": first_damaging_timestamp,
                "duration": first_damaging_timestamp - fight_start,
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
