"""FFLogs 请求时刻还原与整场归零测试。"""

from __future__ import annotations

import pytest

from common.contracts import SLIDECAST_WINDOW_SECONDS
from scripts.convert_fflogs import build_fight_payload, extract_supported_actions
from tests.helpers import DEFAULT_BASE_GCD


def test_hardcast_request_uses_paired_begincast(cs_skill_book):
    payload = {
        "events": [
            {
                "type": "begincast",
                "sourceID": 10,
                "timestamp": 1000,
                "duration": 1926,
                "abilityGameID": 3577,
                "ability": {"name": "炽炎"},
            },
            {
                "type": "cast",
                "sourceID": 10,
                "timestamp": 2382,
                "abilityGameID": 3577,
                "ability": {"name": "炽炎"},
            },
            {
                "type": "cast",
                "sourceID": 10,
                "timestamp": 2500,
                "abilityGameID": 25796,
                "ability": {"name": "详述"},
            },
        ]
    }

    actions, ignored = extract_supported_actions(
        payload,
        10,
        cs_skill_book,
        actual_base_gcd=DEFAULT_BASE_GCD,
        skill_table_base_gcd=DEFAULT_BASE_GCD,
        action_queue_window_seconds=0.5,
    )

    assert ignored == {}
    assert actions[0]["cast_timing_source"] == "begincast_duration"
    assert actions[0]["actual_cast_seconds"] == pytest.approx(1.926)
    assert actions[0]["request_timestamp"] == pytest.approx(1.0)
    assert actions[1]["cast_timing_source"] == "instant_skill"
    assert actions[1]["request_timestamp"] == pytest.approx(2.5)


def test_prepull_hardcast_recovers_cropped_request_with_slidecast_rule(cs_skill_book):
    payload = {
        "fights": [{"start_time": 1000}],
        "events": [
            {
                "type": "cast",
                "sourceID": 10,
                "timestamp": 1000,
                "abilityGameID": 152,
                "ability": {"name": "爆炎"},
            }
        ],
    }

    actions, ignored = extract_supported_actions(
        payload,
        10,
        cs_skill_book,
        actual_base_gcd=DEFAULT_BASE_GCD,
        skill_table_base_gcd=DEFAULT_BASE_GCD,
        action_queue_window_seconds=0.5,
    )

    assert ignored == {}
    assert actions[0]["cast_timing_source"] == "prepull_estimated"
    assert actions[0]["request_timestamp"] == pytest.approx(
        1.0 - 3.5 + SLIDECAST_WINDOW_SECONDS
    )


def test_request_order_uses_queue_window_for_server_event_jitter(cs_skill_book):
    payload = {
        "events": [
            {
                "type": "begincast",
                "sourceID": 10,
                "timestamp": 12980,
                "duration": 1660,
                "abilityGameID": 3577,
                "ability": {"name": "炽炎"},
            },
            {
                "type": "applybuff",
                "sourceID": 10,
                "timestamp": 12983,
                "abilityGameID": 1000049,
                "ability": {"name": "强化药"},
            },
            {
                "type": "cast",
                "sourceID": 10,
                "timestamp": 14140,
                "abilityGameID": 3577,
                "ability": {"name": "炽炎"},
            },
        ]
    }

    actions, ignored = extract_supported_actions(
        payload,
        10,
        cs_skill_book,
        actual_base_gcd=DEFAULT_BASE_GCD,
        skill_table_base_gcd=DEFAULT_BASE_GCD,
        action_queue_window_seconds=0.5,
    )

    assert ignored == {}
    assert [action["action_key"] for action in actions] == ["potion", "fire_iv"]
    assert actions[0]["request_timestamp"] == pytest.approx(12.983)
    assert actions[1]["request_timestamp"] == pytest.approx(12.983)
    assert actions[1]["request_order_adjustment_seconds"] == pytest.approx(0.003)


def test_cancelled_begincast_is_not_paired_with_a_later_instant_cast(cs_skill_book):
    payload = {
        "events": [
            {
                "type": "begincast",
                "sourceID": 10,
                "timestamp": 1000,
                "duration": 1709,
                "abilityGameID": 154,
                "ability": {"name": "冰封"},
            },
            {
                "type": "cast",
                "sourceID": 10,
                "timestamp": 29000,
                "abilityGameID": 7561,
                "ability": {"name": "即刻咏唱"},
            },
            {
                "type": "cast",
                "sourceID": 10,
                "timestamp": 29713,
                "abilityGameID": 154,
                "ability": {"name": "冰封"},
            },
        ]
    }

    actions, ignored = extract_supported_actions(
        payload,
        10,
        cs_skill_book,
        actual_base_gcd=DEFAULT_BASE_GCD,
        skill_table_base_gcd=DEFAULT_BASE_GCD,
        action_queue_window_seconds=0.5,
    )

    assert ignored == {}
    assert actions[1]["cast_timing_source"] == "instant_cast_inferred"
    assert actions[1]["actual_cast_seconds"] == 0.0
    assert actions[1]["request_timestamp"] == pytest.approx(29.713)


def test_fight_payload_translates_all_requests_to_earliest_zero():
    raw_requests = (-3.50, 0.11, 0.23, 0.25, 2.60)
    actions = [
        {
            "timestamp": request + (3.0 if index == 0 else 0.0),
            "request_timestamp": request,
            "action_key": "fire_iii" if index == 0 else "amplifier",
            "skill_id": 152 if index == 0 else 25796,
            "skill_name": "爆炎" if index == 0 else "详述",
            "actual_cast_seconds": 3.5 if index == 0 else 0.0,
            "is_damaging": index == 0,
            "x": 0.0,
            "y": 0.0,
        }
        for index, request in enumerate(raw_requests)
    ]
    actions.sort(key=lambda action: float(action["timestamp"]))

    payload = build_fight_payload(
        actions,
        fight_id="normalized",
        player_name="Tester",
        encounter_name="Demo",
        report_code="demo",
        source_id=10,
        job_tag="black_mage",
        gcd_time=DEFAULT_BASE_GCD,
        generated_at="2026-09-15T00:00:00+00:00",
        downtime_gap_seconds=30.0,
        raid_buff_marker_keys=("raid_marker",),
        raid_buff_window_duration=20.0,
    )

    normalized = [action["request_time_offset"] for action in payload["actions"]]
    assert normalized == pytest.approx([0.0, 3.61, 3.73, 3.75, 6.10])
    assert payload["duration"] == pytest.approx(6.10)
