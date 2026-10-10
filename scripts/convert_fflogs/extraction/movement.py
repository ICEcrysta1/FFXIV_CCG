"""从玩家坐标的连续速度曲线提取移动窗口。"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import median

from common.contracts import SCENE_EPSILON, SLIDECAST_WINDOW_SECONDS

from ..config import MovementDetectionConfig
from .movement_curve import sample_bezier_speed, threshold_windows


def detect_forced_movement_windows(
    events: list[dict[str, object]],
    *,
    source_id: int,
    actual_base_gcd: float,
    movement_detection: MovementDetectionConfig,
    fight_start: float,
    fight_end: float,
    hardcast_windows: list[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """按贝塞尔连续速度阈值截取、GCD 合并、扣实际读条并剔除短窗口。

    不设速度上限或观测间隔门槛。贝塞尔曲线决定阈值交点，间隔不超过
    实测一 GCD 的候选先合并；扣除硬读条后不重新跨读条合并。
    """
    if not math.isfinite(actual_base_gcd) or actual_base_gcd <= 0.0:
        raise ValueError("actual_base_gcd must be positive and finite")
    if not math.isfinite(fight_start) or not math.isfinite(fight_end):
        raise ValueError("fight bounds must be finite")
    positions = _collect_player_positions(
        events, source_id=source_id, coordinate_scale=movement_detection.coordinate_scale,
        fight_start=fight_start, fight_end=fight_end,
    )
    if len(positions) < 2:
        return []
    origin = positions[0][0]
    duration = positions[-1][0] - origin
    times, speed = sample_bezier_speed(
        positions, sample_step_seconds=movement_detection.sample_step_seconds,
    )
    candidates = threshold_windows(
        times, speed, threshold=movement_detection.speed_threshold, duration=duration,
    )
    merge_gap = movement_detection.merge_gap_gcds * actual_base_gcd
    merged: list[tuple[float, float]] = []
    for start, end in candidates:
        if merged and start - merged[-1][1] <= merge_gap + SCENE_EPSILON:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    minimum_length = (
        movement_detection.minimum_window_gcds * actual_base_gcd - SLIDECAST_WINDOW_SECONDS
    )
    cuts = sorted((start-origin, end-origin) for start, end in (hardcast_windows or []))
    result = []
    for start, end in merged:
        for left, right in _subtract_windows(start, end, cuts):
            if right-left > SCENE_EPSILON and right-left+SCENE_EPSILON >= minimum_length:
                result.append((origin+left, origin+right))
    return result


def _subtract_windows(
    start: float, end: float, cuts: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """从单个窗口扣除有序读条区间，扣除后不重新合并跨越读条的片段。"""
    result = []
    cursor = start
    for left, right in cuts:
        if right <= cursor:
            continue
        if left >= end:
            break
        if left > cursor:
            result.append((cursor, min(left, end)))
        cursor = max(cursor, right)
        if cursor >= end:
            break
    if cursor < end:
        result.append((cursor, end))
    return result


def _collect_player_positions(
    events: list[dict[str, object]],
    *,
    source_id: int,
    coordinate_scale: float,
    fight_start: float,
    fight_end: float,
) -> list[tuple[float, float, float]]:
    """只收集完整有限坐标，同刻各轴取中位数；缺失值不补零。"""
    observations: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for event in events:
        if not isinstance(event, dict):
            continue
        try:
            timestamp = float(event["timestamp"]) / 1000.0
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(timestamp) or not fight_start <= timestamp <= fight_end:
            continue
        for actor_key, resources_key in (
            ("sourceID", "sourceResources"),
            ("targetID", "targetResources"),
        ):
            resources = event.get(resources_key)
            if event.get(actor_key) != source_id or not isinstance(resources, dict):
                continue
            try:
                x = float(resources["x"]) / coordinate_scale
                y = float(resources["y"]) / coordinate_scale
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                observations[timestamp].append((x, y))
    return sorted(
        (timestamp, median(x for x, _ in positions), median(y for _, y in positions))
        for timestamp, positions in observations.items()
    )
