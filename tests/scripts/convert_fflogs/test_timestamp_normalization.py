"""FFLogs 请求时刻还原与整场归零测试。"""

from __future__ import annotations

import pytest

from common.context.contracts import SLIDECAST_WINDOW_SECONDS
from scripts.convert_fflogs import build_fight_payload, extract_supported_actions, load_convert_fflogs_config
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
        movement_detection=load_convert_fflogs_config().movement_detection,
        raw_events=[
            {"type": "death", "targetID": 10, "timestamp": 1000},
            {"type": "death", "targetID": 10, "timestamp": -3500},
            {"type": "death", "targetID": 10, "timestamp": 1000},
            {"type": "death", "sourceID": 10, "targetID": 20, "timestamp": 500},
            {"type": "resurrect", "targetID": 10, "timestamp": 2000},
            {"type": "death", "targetID": 10, "timestamp": -4000},
            {"type": "death", "targetID": 10, "timestamp": 3000},
        ],
    )

    normalized = [action["request_time_offset"] for action in payload["actions"]]
    assert normalized == pytest.approx([0.0, 3.61, 3.73, 3.75, 6.10])
    assert payload["duration"] == pytest.approx(6.10)
    # 死亡和动作共用请求时间原点，只保留当前玩家、战斗范围内的死亡并去重排序。
    assert payload["external_events"] == [
        {"timestamp": 0.0, "event_kind": "player_death"},
        {"timestamp": 4.5, "event_kind": "player_death"},
    ]
    assert "player_death" not in payload["scene_context"]


def _lifecycle_payload(events, *, cast_action=None):
    actions = [
        {"timestamp": timestamp, "request_timestamp": timestamp,
         "action_key": "amplifier", "skill_id": 25796, "skill_name": "详述",
         "is_damaging": False, "x": 0.0, "y": 0.0}
        for timestamp in (100.0, 140.0)
    ]
    if cast_action is not None:
        actions.insert(1, {**actions[0], **cast_action})
    return build_fight_payload(
        actions, fight_id="lifecycle", player_name="Tester", encounter_name="Demo",
        report_code="demo", source_id=10, job_tag="black_mage", gcd_time=DEFAULT_BASE_GCD,
        generated_at="2026-10-18", downtime_gap_seconds=30.0,
        raid_buff_marker_keys=("raid_marker",), raid_buff_window_duration=20.0,
        movement_detection=load_convert_fflogs_config().movement_detection, raw_events=events,
    )


def test_lifecycle_prefers_actual_protection_and_preserves_sorted_deduplicated_deaths():
    """复活邀请不等于复活，重复保护只产生一个边界，未复活尾部保留死亡。"""
    protection = {"type": "applybuff", "sourceID": -1, "targetID": 10,
                  "timestamp": 115000, "abilityGameID": 1000418,
                  "targetResources": {"mp": 1800, "maxMP": 10000}}
    payload = _lifecycle_payload([
        {"type": "death", "targetID": 10, "timestamp": 130000},
        {"type": "applybuff", "targetID": 10, "timestamp": 105000, "abilityGameID": 1000418},
        protection,
        {"type": "death", "targetID": 10, "timestamp": 110000},
        {"type": "applybuff", "targetID": 10, "timestamp": 112000, "abilityGameID": 1000148,
         "targetResources": {"mp": 9000}},
        {"type": "applybuff", "targetID": 20, "timestamp": 113000, "abilityGameID": 1000418},
        {"type": "cast", "sourceID": 10, "timestamp": 117000, "abilityGameID": 7561,
         "sourceResources": {"mp": 2600}},
        dict(protection),
        {"type": "death", "targetID": 10, "timestamp": 110000},
        {"type": "death", "targetID": 20, "timestamp": 125000},
        {"type": "death", "targetID": 10, "timestamp": 99000},
        {"type": "applybuff", "targetID": 10, "timestamp": 141000, "abilityGameID": 1000418},
    ])

    assert payload["external_events"] == [
        {"timestamp": 10.0, "event_kind": "player_death"},
        {"timestamp": 15.0, "event_kind": "player_revived", "mp": 1800},
        {"timestamp": 30.0, "event_kind": "player_death"},
    ]


@pytest.mark.parametrize(("resources", "expected_mp"), [
    (None, 2000), ({"mp": 0, "maxMP": 10000}, 0),
    ({"mp": 2600, "maxMP": 10000}, 2600), ({"mp": True}, 2000),
    ({"mp": 12000, "maxMP": 10000}, 2000),
])
def test_protection_uses_only_its_own_reliable_target_mp(resources, expected_mp):
    payload = _lifecycle_payload([
        {"type": "death", "targetID": 10, "timestamp": 110000},
        {"type": "applybuff", "sourceID": -1, "targetID": 10,
         "timestamp": 115000, "ability": {"guid": 1000418}, "targetResources": resources,
         "sourceResources": {"mp": 9000}},
        {"type": "cast", "sourceID": 10, "timestamp": 117000,
         "abilityGameID": 7561, "sourceResources": {"mp": 2800}},
    ])

    assert payload["external_events"][1] == {
        "timestamp": 15.0, "event_kind": "player_revived", "mp": expected_mp,
    }


@pytest.mark.parametrize(("cast_seconds", "request_seconds", "expected_mp"), [
    (120.0, 120.0, 2400), (122.0, 120.0, 2000),
])
def test_revival_without_protection_uses_extracted_cast_request_before_submission(
    cast_seconds, request_seconds, expected_mp,
):
    """实际 cast 晚于起读时，既不把复活放在技能后，也不倒填未来 MP。"""
    payload = _lifecycle_payload([
        {"type": "death", "targetID": 10, "timestamp": 110000},
        {"type": "applybuff", "targetID": 10, "timestamp": 112000, "abilityGameID": 1000148},
        {"type": "cast", "sourceID": 10, "targetID": 20,
         "timestamp": cast_seconds * 1000, "abilityGameID": 152,
         "sourceResources": {"mp": 2400}, "targetResources": {"mp": 10000}},
    ], cast_action={"timestamp": cast_seconds, "request_timestamp": request_seconds,
                    "raw_event_index": 2, "action_key": "fire_iii", "skill_id": 152})

    assert payload["external_events"] == [
        {"timestamp": 10.0, "event_kind": "player_death"},
        {"timestamp": 20.0, "event_kind": "player_revived", "mp": expected_mp},
    ]
    assert payload["external_events"][1]["timestamp"] == payload["actions"][1]["request_time_offset"]


def test_revival_falls_back_to_first_own_begincast_but_not_party_actions_or_damage():
    payload = _lifecycle_payload([
        {"type": "death", "targetID": 10, "timestamp": 110000},
        {"type": "cast", "sourceID": 11, "targetID": 10, "timestamp": 111000, "abilityGameID": 125},
        {"type": "damage", "sourceID": 10, "timestamp": 112000, "amount": 100},
        {"type": "begincast", "sourceID": 10, "timestamp": 118000, "abilityGameID": 152},
        {"type": "cast", "sourceID": 10, "timestamp": 120000,
         "abilityGameID": 152, "sourceResources": {"mp": 2400}},
    ])

    assert payload["external_events"] == [
        {"timestamp": 10.0, "event_kind": "player_death"},
        {"timestamp": 18.0, "event_kind": "player_revived", "mp": 2000},
    ]


@pytest.mark.parametrize("event_type", ["cast", "begincast"])
@pytest.mark.parametrize("has_later_activity", [False, True])
def test_same_timestamp_action_does_not_revive(event_type, has_later_activity):
    """死亡同刻的旧动作不能充当复活证据，只能采用严格晚于死亡的新活动。"""
    events = [
        {"type": "death", "targetID": 10, "timestamp": 110000},
        {"type": event_type, "sourceID": 10, "timestamp": 110000,
         "abilityGameID": 152, "sourceResources": {"mp": 9000}},
    ]
    expected = [{"timestamp": 10.0, "event_kind": "player_death"}]
    if has_later_activity:
        events.append({"type": "begincast", "sourceID": 10, "timestamp": 118000,
                       "abilityGameID": 152})
        expected.append({"timestamp": 18.0, "event_kind": "player_revived", "mp": 2000})

    assert _lifecycle_payload(events)["external_events"] == expected


def test_death_timestamp_request_does_not_revive_even_when_cast_lands_later():
    """日志 cast 晚于死亡时，仍不能用死亡同刻的旧请求推断复活。"""
    payload = _lifecycle_payload([
        {"type": "death", "targetID": 10, "timestamp": 110000},
        {"type": "cast", "sourceID": 10, "timestamp": 112000,
         "abilityGameID": 152, "sourceResources": {"mp": 2400}},
    ], cast_action={"timestamp": 112.0, "request_timestamp": 110.0,
                    "raw_event_index": 1, "action_key": "fire_iii", "skill_id": 152})

    assert payload["external_events"] == [
        {"timestamp": 10.0, "event_kind": "player_death"},
    ]


def test_same_timestamp_death_precedes_revival_and_each_later_death_can_revive_once():
    payload = _lifecycle_payload([
        {"type": "applybuff", "targetID": 10, "timestamp": 110000, "abilityGameID": 1000418},
        {"type": "death", "targetID": 10, "timestamp": 110000},
        {"type": "death", "targetID": 10, "timestamp": 125000},
        {"type": "applybuff", "targetID": 10, "timestamp": 130000, "abilityGameID": 1000418},
    ])

    assert payload["external_events"] == [
        {"timestamp": 10.0, "event_kind": "player_death"},
        {"timestamp": 10.0, "event_kind": "player_revived", "mp": 2000},
        {"timestamp": 25.0, "event_kind": "player_death"},
        {"timestamp": 30.0, "event_kind": "player_revived", "mp": 2000},
    ]
