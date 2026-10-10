"""连续速度移动窗口、真实读条扣除及 GCD／滑步边界回归。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
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
from scripts.convert_fflogs.extraction.extraction import extract_gcd_cast_windows
from scripts.convert_fflogs.extraction.movement_curve import sample_bezier_speed, smooth_speed_envelope


def _position(timestamp, x, *, y=100, actor=7, target=False):
    """测试坐标使用游戏单位，事件载荷按 FFLogs 放大 100 倍。"""
    return {
        "timestamp": round(timestamp * 1000),
        "type": "heal" if target else "damage",
        "targetID" if target else "sourceID": actor,
        "targetResources" if target else "sourceResources": {"x": x * 100, "y": y * 100},
    }


def _detect(events, *, gcd=2.5, config=None, start=0.0, end=20.0, casts=None):
    return detect_forced_movement_windows(
        events, source_id=7, actual_base_gcd=gcd,
        movement_detection=config or load_convert_fflogs_config().movement_detection,
        fight_start=start, fight_end=end, hardcast_windows=casts,
    )


def _sample_rectangles(monkeypatch, spans, *, duration=8.0):
    """用已知交点的速度输入隔离验证合并及裁回，仍运行真实平滑与阈值提取。"""
    times = np.arange(round(duration / 0.01) + 1) * 0.01
    speed = np.zeros_like(times)
    for start, end in spans:
        speed[round(start/0.01):round(end/0.01)+1] = 1.0
    monkeypatch.setattr(movement_module, "sample_bezier_speed", lambda *_a, **_kw: (times, speed))
    return times, speed


def test_long_observation_gaps_and_small_individual_displacements_are_retained():
    # 长间隔和每次不足 0.25 的坐标变化都通过真实平均速度判断。
    assert _detect([_position(t, 100+t) for t in [0, 2, 4]]) == [(0.0, 4.0)]
    events = [_position(i/5, 100+i/10) for i in range(16)]
    assert _detect(events) == [(0.0, 3.0)]


def test_velocity_uses_both_coordinate_axes():
    events = [_position(t, 100+t/4, y=100+t/4) for t in [0, 0.5, 1, 1.5, 2, 2.5, 3]]
    assert _detect(events) == [(0.0, 3.0)]
    assert _detect([_position(t, 100+t/4) for t in [0, 1, 2, 3]]) == []


def test_threshold_is_speed_instead_of_per_observation_distance():
    events = [_position(t, 100+t/2) for t in [0, 0.5, 1, 1.5, 2]]
    config = load_convert_fflogs_config().movement_detection
    assert _detect(events, config=replace(config, speed_threshold=0.5)) == [(0.0, 2.0)]
    assert _detect(events, config=replace(config, speed_threshold=0.5001)) == []


def test_gcd_merge_boundary_uses_unexpanded_crossings(monkeypatch):
    _sample_rectangles(monkeypatch, [(1, 3), (3.3, 5)])
    events = [_position(0, 100), _position(8, 108)]
    split = _detect(events, gcd=0.1)
    assert len(split) == 2
    gap = split[1][0] - split[0][1]
    assert _detect(events, gcd=gap) == [(split[0][0], split[1][1])]
    assert _detect(events, gcd=gap-0.001) == split


def test_expansion_cannot_merge_a_gap_above_one_gcd(monkeypatch):
    times, speed = _sample_rectangles(monkeypatch, [(1, 3), (3.3, 5)])
    config = load_convert_fflogs_config().movement_detection
    smooth = smooth_speed_envelope(speed, sample_step_seconds=config.sample_step_seconds,
        maximum_expansion_per_side_seconds=config.maximum_expansion_per_side_seconds)
    assert len(movement_module.threshold_windows(times, smooth,
        threshold=config.speed_threshold, duration=8.0)) == 1
    # 平滑已经连成一段，外扩前仍有超过 GCD 的间隙，最终必须保留两段。
    windows = _detect([_position(0, 100), _position(8, 108)], gcd=0.28)
    assert len(windows) == 2
    assert windows[1][0] - windows[0][1] > 0.28


def test_gcd_merge_is_transitive(monkeypatch):
    _sample_rectangles(monkeypatch, [(1, 2), (2.8, 3.8), (4.6, 5.6)])
    windows = _detect([_position(0, 100), _position(8, 108)], gcd=0.8)
    assert len(windows) == 1
    assert windows[0] == pytest.approx((1-0.02/3, 5.6+0.02/3))


def test_trim_restores_actual_edges_and_preserves_internal_bridge(monkeypatch):
    _sample_rectangles(monkeypatch, [(1, 3), (3.3, 5)])
    windows = _detect([_position(0, 100), _position(8, 108)], gcd=2.5)
    assert len(windows) == 1
    # 阈值为 1/3，两端交点来自 0.01 秒网格，不统一减去 0.5 秒。
    assert windows[0] == pytest.approx((1-0.02/3, 5+0.02/3))


def test_short_window_is_filtered_after_trimming_expansion(monkeypatch):
    _sample_rectangles(monkeypatch, [(2, 3.8)])
    assert _detect([_position(0, 100), _position(8, 108)]) == []


@pytest.mark.parametrize("duration, retained", [(1.999, False), (2.0, True), (2.001, True)])
def test_minimum_length_is_gcd_minus_shared_slidecast(duration, retained):
    gcd = 2.0 + SLIDECAST_WINDOW_SECONDS
    events = [_position(0, 100), _position(duration, 100+duration)]
    assert _detect(events, gcd=gcd) == ([(0.0, duration)] if retained else [])


def test_minimum_length_reads_shared_slidecast_constant(monkeypatch):
    events = [_position(0, 100), _position(1.9, 101.9)]
    assert _detect(events) == []
    monkeypatch.setattr(movement_module, "SLIDECAST_WINDOW_SECONDS", 0.75)
    assert _detect(events) == [(0.0, 1.9)]


@pytest.mark.parametrize("origin", [0.0, 114.552, 1000000.0])
def test_absolute_time_translation_preserves_windows_and_duration_boundary(origin):
    events = [_position(origin+t, 100+t) for t in [0, 0.5, 1, 1.5, 1.942]]
    windows = _detect(events, gcd=2.442, start=origin, end=origin+20)
    assert len(windows) == 1
    assert windows[0] == pytest.approx((origin, origin+1.942))


def test_curve_does_not_extrapolate_movement_outside_observations():
    assert _detect([_position(10, 100), _position(12, 102)]) == [(10.0, 12.0)]


def test_target_resources_and_teleport_speed_are_retained():
    events = [_position(t, 100+i*1000, target=True)
        for i, t in enumerate([0, 0.1, 0.6, 1.1, 1.6, 2.1])]
    assert _detect(list(reversed(events))) == [(0.0, 2.1)]


def test_same_timestamp_median_rejects_outlier_and_ignores_other_players():
    events = []
    for t in [0, 0.5, 1, 1.5, 2]:
        events.extend([_position(t, 100, target=True), _position(t, 100),
            _position(t, 900 if t % 1 == 0 else 10), _position(t, 1000+t*100, actor=8)])
    assert _detect(events) == []


def test_missing_nonfinite_and_out_of_fight_coordinates_do_not_make_windows():
    events = [_position(t, 100) for t in [0, 0.5, 1, 1.5, 2]]
    events += [{"timestamp": 250, "sourceID": 7},
        {"timestamp": 750, "sourceID": 7, "sourceResources": {"x": 0}},
        _position(1.25, float("nan")), _position(1.75, float("inf")),
        _position(-0.5, 0), _position(2.5, 0),
        {"timestamp": float("nan"), "sourceID": 7, "sourceResources": {"x": 0, "y": 0}}]
    assert _detect(events, start=0, end=2) == []


@pytest.mark.parametrize("events", [[], [_position(1, 100)]])
def test_insufficient_coordinates_have_no_movement(events):
    assert _detect(events) == []


def test_hardcast_subtraction_does_not_remerge_across_casts():
    events = [_position(t, 100+t) for t in range(13)]
    windows = _detect(events, casts=[(7, 8), (3, 4), (3.5, 4.5)])
    assert windows == [(0.0, 3.0), (4.5, 7.0), (8.0, 12.0)]


def test_short_fragments_are_filtered_after_hardcast_subtraction():
    assert _detect([_position(0, 100), _position(5, 105)], casts=[(1, 3)]) == [(3.0, 5.0)]


def test_bezier_and_envelope_do_not_create_negative_or_excessive_peaks():
    times, speed = sample_bezier_speed([(0, 0, 0), (1, 0, 0), (2, 10, 0), (3, 10, 0)],
        sample_step_seconds=0.01)
    assert np.min(speed) >= 0.0
    assert np.max(speed) == pytest.approx(10.0)
    assert speed[round(1.5/0.01)] == pytest.approx(10.0)
    smooth = smooth_speed_envelope(speed, sample_step_seconds=0.01,
        maximum_expansion_per_side_seconds=0.5)
    assert np.all(smooth >= speed-1e-9)
    assert np.max(smooth) <= 10.0+1e-9
    assert np.all(smooth[times <= 0.0] == 0.0)


def test_actual_cast_windows_include_cancelled_begins_and_estimated_prepull():
    skills = build_skill_book(load_job_project_config("black_mage"))
    events = [{"type": "begincast", "sourceID": 7, "abilityGameID": 3577,
        "timestamp": t, "duration": duration} for t, duration in [(10000, 3000), (11000, 2000)]]
    actions = [{"skill_id": 152, "cast_timing_source": "prepull_estimated",
        "request_timestamp": 5.0, "actual_cast_seconds": 3.0},
        {"skill_id": 3577, "cast_timing_source": "instant_cast_inferred",
        "request_timestamp": 15.0, "actual_cast_seconds": 0.0}]
    assert extract_gcd_cast_windows(events, actions, source_id=7, skill_book=skills) == [
        (5.0, 8.0), (10.0, 11.0), (11.0, 13.0)]


@pytest.mark.parametrize("gcd", [0, -1, float("nan"), float("inf")])
def test_invalid_gcd_is_rejected(gcd):
    with pytest.raises(ValueError, match="actual_base_gcd"):
        _detect([], gcd=gcd)


def test_conversion_preserves_instant_gcds_and_removes_only_actual_hardcast():
    project = load_job_project_config("black_mage")
    events = [_position(t/2, 100+t/2, target=True) for t in range(20, 45)]
    events += [{"type": "cast", "sourceID": 7, "timestamp": t, "abilityGameID": skill_id}
        for t, skill_id in [(10000, 16507), (15500, 3577), (18000, 3577), (22000, 16507)]]
    events.append({"type": "begincast", "sourceID": 7, "timestamp": 14000,
        "abilityGameID": 3577, "duration": 2000})
    payload, _ = convert_report_payload({"events": events, "fight_id": 1}, job_tag="black_mage",
        project_config=project, skill_book=build_skill_book(project), source_id=7,
        encounter_name="Demo", report_code="demo", player_name="Tester",
        generated_at="2026-10-10T00:00:00Z")
    assert payload["scene_context"]["forced_movement_context"]["tokens"] == [
        [0.0, 4.0, 4.0], [6.0, 12.0, 6.0]]
    assert payload["actions"][0]["request_time_offset"] == 0.0
    instant_fire = next(a for a in payload["actions"] if a["time_offset"] == 8.0)
    assert instant_fire["actual_cast_seconds"] == 0.0
    assert instant_fire["cast_timing_source"] == "instant_cast_inferred"
