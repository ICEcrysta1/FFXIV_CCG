"""FFLogs 转换脚本测试。"""

from __future__ import annotations

from scripts.convert_fflogs import (
    load_job_project_config,
    GcdDetectionConfig,
    annotate_action_movement,
    build_fight_payload,
    convert_report_payload,
    convert_report_to_training_payload,
    detect_gcd_from_logs,
    extract_supported_actions,
    build_raid_buff_window_context,
    build_target_count_window_context,
    build_target_count_window_token,
)
from tests.helpers import DEFAULT_BASE_GCD, forced_movement_window_token, raid_buff_window_token, targetable_window_token


def test_extract_supported_actions_resolves_action_keys_and_potion(cs_backend, cs_skill_book):
    raw_payload = {
        "events": [
            {
                "type": "cast",
                "sourceID": 10,
                "timestamp": 1000,
                "abilityGameID": 152,
                "ability": {"name": "爆炎"},
                "sourceResources": {"x": 0.0, "y": 0.0},
            },
            {
                "type": "applybuff",
                "sourceID": 10,
                "timestamp": 1100,
                "abilityGameID": 1000049,
                "ability": {"name": "强化药"},
                "sourceResources": {"x": 1.0, "y": 1.0},
            },
            {
                "type": "cast",
                "sourceID": 10,
                "timestamp": 1200,
                "abilityGameID": 123456,
                "ability": {"name": "未知技能"},
            },
        ]
    }

    actions, ignored = extract_supported_actions(
        raw_payload,
        10,
        cs_skill_book,
        actual_base_gcd=DEFAULT_BASE_GCD,
        skill_table_base_gcd=DEFAULT_BASE_GCD,
        action_queue_window_seconds=0.5,
    )

    assert [action["action_key"] for action in actions] == ["fire_iii", "potion"]
    assert actions[0]["skill_name"] == "爆炎"
    assert actions[1]["skill_name"] == "爆发药"
    assert ignored[(123456, "未知技能")] == 1


def test_build_fight_payload_uses_new_scene_context_shell(cs_backend, cs_skill_book):
    fight_actions = [
        {
            "timestamp": 100.0,
            "action_key": "fire_iii",
            "skill_id": 152,
            "skill_name": "爆炎",
            "raw_skill_name": "爆炎",
            "cast_time": 3.5,
            "is_damaging": True,
            "x": 0.0,
            "y": 0.0,
        },
        {
            "timestamp": 104.0,
            "action_key": "amplifier",
            "skill_id": 25796,
            "skill_name": "详述",
            "raw_skill_name": "详述",
            "cast_time": 0.0,
            "is_damaging": False,
            "x": 4.5,
            "y": 0.0,
        },
        {
            "timestamp": 111.5,
            "action_key": "xenoglossy",
            "skill_id": 16507,
            "skill_name": "异言",
            "raw_skill_name": "异言",
            "cast_time": 0.0,
            "is_damaging": True,
            "x": 4.5,
            "y": 0.0,
        },
    ]
    annotate_action_movement(fight_actions)

    payload = build_fight_payload(
        fight_actions,
        fight_id="demo_F1",
        player_name="Tester",
        encounter_name="Futures Rewritten (Ultimate)",
        report_code="demo",
        source_id=10,
        job_tag=cs_backend.job_tag,
        gcd_time=DEFAULT_BASE_GCD,
        generated_at="2026-07-02T00:00:00+00:00",
        downtime_gap_seconds=6.0,
        raid_buff_marker_keys=("amplifier",),
        raid_buff_window_duration=20.0,
    )

    assert payload["job_tag"] == "black_mage"
    assert payload["data_schema_version"] == 4
    assert set(payload["scene_context"]) == {
        "targetable_window_context",
        "forced_movement_context",
        "raid_buff_window_context",
        "target_count_window_context",
    }
    # 在 cast=生效时刻 语义下，滑步起点 = prev_timestamp - 0.5 = 100-0.5 = 99.5，offset=-0.5
    assert payload["scene_context"]["forced_movement_context"]["tokens"] == [
        forced_movement_window_token(-0.5, -0.5)
    ]
    assert payload["scene_context"]["raid_buff_window_context"]["tokens"] == [
        raid_buff_window_token(4.0, 24.0)
    ]
    assert payload["scene_context"]["target_count_window_context"]["tokens"] == []
    assert any(
        token[3] == 0.0
        for token in payload["scene_context"]["targetable_window_context"]["tokens"]
    )
    assert payload["actions"][0]["action_key"] == "fire_iii"
    assert payload["actions"][1]["forced_move"] is True


def test_raid_buff_markers_average_same_window():
    context = build_raid_buff_window_context(
        [
            {"timestamp": 100.0, "action_key": "amplifier"},
            {"timestamp": 102.0, "action_key": "raid_buff_b"},
            {"timestamp": 220.0, "action_key": "amplifier"},
        ],
        fight_start=100.0,
        window_duration=20.0,
        marker_keys=("amplifier", "raid_buff_b"),
    )

    assert context["feature_keys"] == [
        "start_offset_seconds",
        "end_offset_seconds",
        "duration_seconds",
        "source.amplifier",
        "source.raid_buff_b",
    ]
    assert context["tokens"] == [
        [1.0, 21.0, 20.0, 1.0, 1.0],
        [120.0, 140.0, 20.0, 1.0, 0.0],
    ]


def test_target_count_window_context_marks_only_replayed_multi_target_windows(cs_backend, cs_skill_book):
    context = build_target_count_window_context(
        [
            {"type": "damage", "sourceID": 10, "timestamp": 1000, "abilityGameID": 25794, "targetID": 201},
            {"type": "damage", "sourceID": 10, "timestamp": 5000, "abilityGameID": 25794, "targetID": 201},
            {"type": "damage", "sourceID": 10, "timestamp": 5000, "abilityGameID": 25794, "targetID": 202},
            {"type": "damage", "sourceID": 10, "timestamp": 9000, "abilityGameID": 25794, "targetID": 201},
            {"type": "damage", "sourceID": 10, "timestamp": 9000, "abilityGameID": 25794, "targetID": 202},
            {"type": "damage", "sourceID": 10, "timestamp": 19000, "abilityGameID": 25794, "targetID": 201},
        ],
        source_id=10,
        skill_book=cs_skill_book,
        fight_start=0.0,
        fight_end=20.0,
    )

    assert context["feature_keys"] == [
        "start_offset_seconds",
        "end_offset_seconds",
        "duration_seconds",
        "target_count",
    ]
    assert context["tokens"] == [
        build_target_count_window_token(5.0, 6.5, 2),
        build_target_count_window_token(9.0, 10.5, 2),
    ]


def test_annotate_action_movement_prefers_runtime_cast_seconds():
    actions = [
        {
            "timestamp": 100.0,
            "action_key": "blizzard_iii",
            "cast_time": 3.5,
            "actual_cast_seconds": 0.0,
            "x": 0.0,
            "y": 0.0,
        },
        {
            "timestamp": 101.0,
            "action_key": "blizzard_iv",
            "cast_time": 2.0,
            "x": 4.5,
            "y": 0.0,
        },
    ]

    annotate_action_movement(actions)

    assert actions[1]["moved"] is True
    assert actions[1]["forced_move"] is False
    assert actions[1]["instant_move"] is True


def test_convert_report_payload_uses_runtime_instant_cast_for_movement(cs_backend, cs_skill_book):
    raw_payload = {
        "report_code": "demo_report",
        "fight_id": 19,
        "events": [
            {
                "type": "cast",
                "sourceID": 10,
                "timestamp": 1000,
                "abilityGameID": 7421,
                "ability": {"name": "三连咏唱"},
                "sourceResources": {"x": 0.0, "y": 0.0},
            },
            {
                "type": "cast",
                "sourceID": 10,
                "timestamp": 2000,
                "abilityGameID": 154,
                "ability": {"name": "冰封"},
                "sourceResources": {"x": 0.0, "y": 0.0},
            },
            {
                "type": "cast",
                "sourceID": 10,
                "timestamp": 3000,
                "abilityGameID": 3576,
                "ability": {"name": "冰澈"},
                "sourceResources": {"x": 4.5, "y": 0.0},
            },
        ],
    }

    payload, ignored = convert_report_payload(
        raw_payload,
        job_tag=cs_backend.job_tag,
        project_config=load_job_project_config("black_mage"),
        skill_book=cs_skill_book,
        source_id=10,
        encounter_name="Demo",
        report_code="demo_report",
        player_name="Tester",
        generated_at="2026-07-05T00:00:00+00:00",
    )

    assert ignored == {}
    assert payload is not None
    assert payload["scene_context"]["forced_movement_context"]["tokens"] == []
    assert payload["actions"][2]["moved"] is True
    assert payload["actions"][2]["forced_move"] is False
    assert payload["actions"][2]["instant_move"] is True


def test_convert_report_payload_keeps_one_complete_fight_even_with_long_downtime(cs_backend, cs_skill_book):
    raw_payload = {
        "report_code": "demo_report",
        "fight_id": 17,
        "events": [
            {
                "type": "cast",
                "sourceID": 10,
                "timestamp": 1000,
                "abilityGameID": 152,
                "ability": {"name": "爆炎"},
                "sourceResources": {"x": 0.0, "y": 0.0},
            },
            {
                "type": "cast",
                "sourceID": 10,
                "timestamp": 121000,
                "abilityGameID": 16507,
                "ability": {"name": "异言"},
                "sourceResources": {"x": 0.0, "y": 0.0},
            },
        ],
    }

    payload, ignored = convert_report_payload(
        raw_payload,
        job_tag=cs_backend.job_tag,
        project_config=load_job_project_config("black_mage"),
        skill_book=cs_skill_book,
        source_id=10,
        encounter_name="Futures Rewritten (Ultimate)",
        report_code="demo_report",
        player_name="Tester",
        generated_at="2026-07-02T00:00:00+00:00",
    )

    assert ignored == {}
    assert payload is not None
    assert payload["fight_id"] == "demo_report_fight17"
    assert payload["num_actions"] == 2
    assert [action["action_key"] for action in payload["actions"]] == ["fire_iii", "xenoglossy"]
    assert any(
        token[3] == 0.0
        for token in payload["scene_context"]["targetable_window_context"]["tokens"]
    )


def test_convert_report_to_training_payload_invokes_build_training_samples(cs_backend, cs_skill_book, monkeypatch):
    import scripts.convert_fflogs.pipeline as pipeline_module

    captured: dict[str, object] = {}

    def _fake_build_training_samples(backend, skill_book, fight_payload):
        captured["job_tag"] = backend.job_tag
        captured["fight_id"] = fight_payload["fight_id"]
        return {
            "fight_id": fight_payload["fight_id"],
            "num_samples": 1,
        }

    monkeypatch.setattr(pipeline_module, "build_training_samples", _fake_build_training_samples)

    raw_payload = {
        "report_code": "demo_report",
        "fight_id": 19,
        "events": [
            {
                "type": "cast",
                "sourceID": 10,
                "timestamp": 1000,
                "abilityGameID": 152,
                "ability": {"name": "爆炎"},
                "sourceResources": {"x": 0.0, "y": 0.0},
            },
        ],
    }

    payload, ignored = convert_report_to_training_payload(
        raw_payload,
        backend=cs_backend,
        project_config=load_job_project_config("black_mage"),
        skill_book=cs_skill_book,
        source_id=10,
        encounter_name="Demo",
        report_code="demo_report",
        player_name="Tester",
        generated_at="2026-07-07T00:00:00+00:00",
    )

    assert ignored == {}
    assert payload == {"fight_id": "demo_report_fight19", "num_samples": 1}
    assert captured == {
        "job_tag": "black_mage",
        "fight_id": "demo_report_fight19",
    }


def test_extract_supported_actions_includes_flare(cs_backend, cs_skill_book):
    raw_payload = {
        "events": [
            {
                "type": "cast",
                "sourceID": 10,
                "timestamp": 1000,
                "abilityGameID": 162,
                "ability": {"name": "鏍哥垎"},
                "sourceResources": {"x": 0.0, "y": 0.0},
            },
        ],
    }

    actions, ignored = extract_supported_actions(
        raw_payload,
        10,
        cs_skill_book,
        actual_base_gcd=DEFAULT_BASE_GCD,
        skill_table_base_gcd=DEFAULT_BASE_GCD,
        action_queue_window_seconds=0.5,
    )

    assert ignored == {}
    assert len(actions) == 1
    assert actions[0]["action_key"] == "flare"
    assert actions[0]["skill_id"] == 162


def test_detect_gcd_from_logs_uses_configured_probe_skill_and_haste_exclusions():
    gcd_detection = GcdDetectionConfig(
        probe_skill_game_id=999001,
        haste_excluded_buff_game_ids=(999777,),
        fallback_seconds=DEFAULT_BASE_GCD,
        min_probe_casts=4,
        min_gap_samples=3,
        histogram_lower_ms=1800,
        histogram_upper_ms=2600,
        histogram_bin_width_ms=10,
    )
    events = [
        {"type": "cast", "sourceID": 10, "timestamp": 1000, "abilityGameID": 999001},
        {"type": "cast", "sourceID": 10, "timestamp": 3500, "abilityGameID": 999001},
        {"type": "applybuff", "sourceID": 10, "targetID": 10, "timestamp": 5600, "duration": 3000, "abilityGameID": 999777},
        {"type": "cast", "sourceID": 10, "timestamp": 6000, "abilityGameID": 999001},
        {"type": "cast", "sourceID": 10, "timestamp": 8100, "abilityGameID": 999001},
        {"type": "cast", "sourceID": 10, "timestamp": 10600, "abilityGameID": 999001},
        {"type": "cast", "sourceID": 10, "timestamp": 13100, "abilityGameID": 999001},
        {"type": "cast", "sourceID": 10, "timestamp": 15600, "abilityGameID": 999001},
        {"type": "cast", "sourceID": 10, "timestamp": 18100, "abilityGameID": 999001},
    ]

    assert detect_gcd_from_logs(events, 10, gcd_detection=gcd_detection) == DEFAULT_BASE_GCD


def test_detect_gcd_from_logs_unscales_black_mage_haste_window():
    gcd_detection = GcdDetectionConfig(
        probe_skill_game_id=3577,
        haste_excluded_buff_game_ids=(737,),
        fallback_seconds=DEFAULT_BASE_GCD,
        min_probe_casts=4,
        min_gap_samples=3,
        histogram_lower_ms=1800,
        histogram_upper_ms=2600,
        histogram_bin_width_ms=10,
    )
    events = [
        *[
            {"type": "cast", "sourceID": 10, "timestamp": timestamp, "abilityGameID": 3577}
            for timestamp in (1000, 3450, 5900, 8350)
        ],
        {
            "type": "applybuff",
            "sourceID": 10,
            "targetID": 10,
            "timestamp": 10000,
            "duration": 10000,
            "abilityGameID": 737,
        },
        *[
            {"type": "begincast", "sourceID": 10, "timestamp": timestamp, "abilityGameID": 3577}
            for timestamp in (11000, 13055, 15110, 17165)
        ],
    ]

    assert (
        detect_gcd_from_logs(
            events,
            10,
            gcd_detection=gcd_detection,
            haste_multiplier=0.85,
        )
        == 2.42
    )


def test_detect_gcd_from_logs_uses_probe_cast_duration_for_precise_base_gcd():
    gcd_detection = GcdDetectionConfig(
        probe_skill_game_id=3577,
        haste_excluded_buff_game_ids=(),
        fallback_seconds=DEFAULT_BASE_GCD,
        min_probe_casts=4,
        min_gap_samples=3,
        histogram_lower_ms=1800,
        histogram_upper_ms=2600,
        histogram_bin_width_ms=10,
        probe_skill_cast_time_seconds=2.0,
    )
    events = [
        {
            "type": "begincast",
            "sourceID": 10,
            "timestamp": timestamp,
            "abilityGameID": 3577,
            "duration": 1948,
        }
        for timestamp in (1000, 3450, 5900, 8350, 10800)
    ]

    assert detect_gcd_from_logs(
        events,
        10,
        gcd_detection=gcd_detection,
        skill_table_base_gcd=2.5,
    ) == 2.435
