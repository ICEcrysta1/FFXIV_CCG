"""从玩家来源和目标资源坐标中提取并聚合移动窗口。"""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import pairwise
from statistics import median

from common.contracts import SCENE_EPSILON, SLIDECAST_WINDOW_SECONDS

from ..config import MovementDetectionConfig


def detect_forced_movement_windows(
    events: list[dict[str, object]],
    *,
    source_id: int,
    actual_base_gcd: float,
    movement_detection: MovementDetectionConfig,
    fight_start: float,
    fight_end: float,
) -> list[tuple[float, float]]:
    """先按实测 GCD 合并坐标变化区间，再剔除短于 GCD 减滑步时间的窗口。

    不设速度上限，也不按动作是否瞬发排除位移；保留原有 scene context 名称。
    合并填入的间隙描述移动片段的聚集范围，不保证整段都在持续移动。
    """
    if not math.isfinite(actual_base_gcd) or actual_base_gcd <= 0.0:
        raise ValueError("actual_base_gcd must be positive and finite")
    positions = _collect_player_positions(
        events,
        source_id=source_id,
        coordinate_scale=movement_detection.coordinate_scale,
        fight_start=fight_start,
        fight_end=fight_end,
    )
    candidates: list[tuple[float, float]] = []
    for previous, current in pairwise(positions):
        gap = current[0] - previous[0]
        distance = math.hypot(current[1] - previous[1], current[2] - previous[2])
        if (
            0.0
            < gap
            <= movement_detection.maximum_observation_gap_seconds + SCENE_EPSILON
            and distance + SCENE_EPSILON >= movement_detection.minimum_displacement
        ):
            candidates.append((previous[0], current[0]))

    merge_gap = movement_detection.merge_gap_gcds * actual_base_gcd
    merged: list[tuple[float, float]] = []
    for start, end in candidates:
        # 绝对时间相减和 GCD 减滑步时间都会有舍入误差，等号边界复用场景容差。
        if merged and start - merged[-1][1] <= merge_gap + SCENE_EPSILON:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    minimum_length = (
        movement_detection.minimum_window_gcds * actual_base_gcd
        - SLIDECAST_WINDOW_SECONDS
    )
    return [
        (start, end)
        for start, end in merged
        if end - start + SCENE_EPSILON >= minimum_length
    ]


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
