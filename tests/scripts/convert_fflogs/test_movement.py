"""移动窗口的坐标来源、先合并后过滤和 GCD／滑步边界回归。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from common.contracts import SLIDECAST_WINDOW_SECONDS
from scripts.convert_fflogs import (
    build_skill_book,
    convert_report_payload,
    detect_forced_movement_windows,
    load_convert_fflogs_config,
    load_job_project_config,
)
from scripts.convert_fflogs.extraction import movement as movement_module


def _position(timestamp, x, *, actor=7, target=False):
    """测试坐标使用游戏单位，事件载荷按 FFLogs 放大 100 倍。"""
    return {
        "timestamp": round(timestamp * 1000),
        "type": "heal" if target else "damage",
        "targetID" if target else "sourceID": actor,
        "targetResources" if target else "sourceResources": {"x": x * 100, "y": 10000},
    }


def _detect(events, *, gcd=2.5, config=None, start=0.0, end=20.0):
    return detect_forced_movement_windows(
        events,
        source_id=7,
        actual_base_gcd=gcd,
        movement_detection=config or load_convert_fflogs_config().movement_detection,
        fight_start=start,
        fight_end=end,
    )


@pytest.mark.parametrize("gap, expected", [(2.5, [(0.0, 3.5)]), (2.501, [])])
def test_short_intervals_merge_before_filter_with_gcd_gap_boundary(gap, expected):
    events = [
        _position(0, 100),
        _position(0.5, 101),
        _position(0.5 + gap, 101),
        _position(1 + gap, 102),
    ]
    assert _detect(events) == expected


def test_merge_is_transitive_and_uses_last_end():
    events = [
        _position(t, x)
        for t, x in [(0, 100), (0.5, 101), (3, 101), (3.5, 102), (6, 102), (6.5, 103)]
    ]
    assert _detect(events) == [(0.0, 6.5)]


def test_merge_uses_extracted_gcd_instead_of_fixed_seconds():
    events = [
        _position(t, x) for t, x in [(0, 100), (0.5, 101), (2.95, 101), (3.45, 102)]
    ]
    assert _detect(events, gcd=2.5) == [(0.0, 3.45)]
    assert _detect(events, gcd=2.4) == []


@pytest.mark.parametrize(
    "duration, retained", [(1.999, False), (2.0, True), (2.001, True)]
)
def test_minimum_length_is_gcd_minus_shared_slidecast(duration, retained):
    gcd = 2.0 + SLIDECAST_WINDOW_SECONDS
    events = [_position(t, 100 + i) for i, t in enumerate([0, 0.5, 1, 1.5, duration])]
    assert _detect(events, gcd=gcd) == ([(0.0, duration)] if retained else [])


def test_minimum_length_reads_shared_slidecast_constant(monkeypatch):
    events = [_position(t, 100 + i) for i, t in enumerate([0, 0.5, 1, 1.5, 1.9])]
    assert _detect(events) == []
    monkeypatch.setattr(movement_module, "SLIDECAST_WINDOW_SECONDS", 0.75)
    assert _detect(events) == [(0.0, 1.9)]


@pytest.mark.parametrize("origin", [0.0, 114.552, 1000000.0])
def test_decimal_gcd_and_minimum_boundaries_survive_absolute_timestamps(origin):
    events = [
        _position(origin + t, 100 + i) for i, t in enumerate([0, 0.5, 1, 1.5, 1.942])
    ]
    windows = _detect(events, gcd=2.442, start=origin, end=origin + 20)
    assert len(windows) == 1
    assert windows[0] == pytest.approx((origin, origin + 1.942))

    events = [
        _position(origin + t, x)
        for t, x in [(0, 100), (0.5, 101), (2.942, 101), (3.442, 102)]
    ]
    windows = _detect(events, gcd=2.442, start=origin, end=origin + 20)
    assert len(windows) == 1
    assert windows[0] == pytest.approx((origin, origin + 3.442))


def test_target_resources_and_fast_displacement_are_retained():
    # 模拟传送式坐标跳变；不能因速度或事件不是玩家 cast 而丢弃。
    events = [
        _position(t, 100 + i * 1000, target=True)
        for i, t in enumerate([0, 0.1, 0.6, 1.1, 1.6, 2.1])
    ]
    assert _detect(list(reversed(events))) == [(0.0, 2.1)]


def test_same_timestamp_median_rejects_outlier_and_ignores_other_players():
    events = []
    for t in [0, 0.5, 1, 1.5, 2]:
        events.extend(
            [
                _position(t, 100, target=True),
                _position(t, 100),
                _position(t, 900 if t % 1 == 0 else 10),
                _position(t, 1000 + t * 100, actor=8),
            ]
        )
    assert _detect(events) == []


def test_missing_nonfinite_and_out_of_fight_coordinates_do_not_make_windows():
    events = [_position(t, 100) for t in [0, 0.5, 1, 1.5, 2]]
    events += [
        {"timestamp": 250, "sourceID": 7},
        {"timestamp": 750, "sourceID": 7, "sourceResources": {"x": 0}},
        _position(1.25, float("nan")),
        _position(1.75, float("inf")),
        _position(-0.5, 0),
        _position(2.5, 0),
        {"timestamp": float("nan"), "sourceID": 7, "sourceResources": {"x": 0, "y": 0}},
    ]
    assert _detect(events, start=0, end=2) == []


def test_observation_gap_and_displacement_threshold_are_applied_before_merge():
    events = [_position(t, 100 + i * 0.25) for i, t in enumerate([0, 0.5, 1, 1.5, 2])]
    config = load_convert_fflogs_config().movement_detection
    assert _detect(events, config=config) == [(0.0, 2.0)]
    assert _detect(events, config=replace(config, minimum_displacement=0.251)) == []
    assert (
        _detect(events, config=replace(config, maximum_observation_gap_seconds=0.499))
        == []
    )


@pytest.mark.parametrize("gcd", [0, -1, float("nan"), float("inf")])
def test_invalid_gcd_is_rejected(gcd):
    with pytest.raises(ValueError, match="actual_base_gcd"):
        _detect([], gcd=gcd)


def test_conversion_uses_coordinate_windows_and_preserves_request_anchor():
    project = load_job_project_config("black_mage")
    events = [
        _position(t, 100 + i, target=True)
        for i, t in enumerate([10, 10.5, 11, 11.5, 12])
    ]
    events.extend(
        {"type": "cast", "sourceID": 7, "timestamp": t, "abilityGameID": 16507}
        for t in [10000, 13000]
    )
    payload, _ = convert_report_payload(
        {"events": events, "fight_id": 1},
        job_tag="black_mage",
        project_config=project,
        skill_book=build_skill_book(project),
        source_id=7,
        encounter_name="Demo",
        report_code="demo",
        player_name="Tester",
        generated_at="2026-10-10T00:00:00Z",
    )
    assert payload["scene_context"]["forced_movement_context"]["tokens"] == [
        [0.0, 2.0, 2.0]
    ]
    assert payload["actions"][0]["request_time_offset"] == 0.0
