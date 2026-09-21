"""训练样本构建的 scene 上下文与目标可选中测试。"""

from __future__ import annotations

from scripts.convert_fflogs import build_training_samples
from tests.helpers import (
    build_test_scene_context,
    candidate_state_vector_value,
    forced_movement_window_token,
    raid_buff_window_token,
    targetable_window_token,
)


def test_build_training_samples_keeps_absolute_scene_context_and_syncs_runtime_flags(cs_backend, cs_skill_book):
    fight_payload = {
        "fight_id": "scene_demo",
        "job_tag": "black_mage",
        "player": "Tester",
        "encounter": "Demo",
        "duration": 11.0,
        "scene_context": build_test_scene_context(
            targetable_tokens=[
                targetable_window_token(0.0, 2.0, targetable=True, segment_kind="combat"),
                targetable_window_token(2.0, 5.0, targetable=False, segment_kind="downtime"),
                targetable_window_token(5.0, 11.0, targetable=True, segment_kind="combat_final"),
            ],
            forced_movement_tokens=[
                forced_movement_window_token(0.0, 1.5),
            ],
            raid_buff_tokens=[
                raid_buff_window_token(0.0, 4.0),
            ],
        ),
        "actions": [
            {
                "time_offset": 0.0,
                "request_time_offset": 0.0,
                "time_gap": 0.0,
                "action_key": "lucid_dreaming",
                "fight_remaining": 11.0,
                "anchor": "combat",
            },
            {
                "time_offset": 3.0,
                "request_time_offset": 3.0,
                "time_gap": 3.0,
                "action_key": "potion",
                "fight_remaining": 8.0,
                "anchor": "downtime_1",
            },
        ],
    }

    payload = build_training_samples(cs_backend, cs_skill_book, fight_payload)

    first_sample = payload["samples"][0]
    second_sample = payload["samples"][1]

    fire_iii_at_movement = _candidate_skill_token(first_sample["context"], "fire_iii")
    fire_iii_at_downtime = _candidate_skill_token(second_sample["context"], "fire_iii")

    assert fire_iii_at_movement["is_legal"] is True
    assert fire_iii_at_movement["invalid_reason"] == ""
    assert candidate_state_vector_value(
        first_sample["context"],
        "fire_iii",
        "player_state",
        "before.is_moving",
    ) == 1.0
    assert first_sample["context"]["scene_context"]["forced_movement_context"]["tokens"] == [
        forced_movement_window_token(0.0, 1.5)
    ]
    assert first_sample["context"]["scene_context"]["targetable_window_context"]["tokens"][0] == (
        targetable_window_token(0.0, 2.0, targetable=True, segment_kind="combat")
    )
    assert candidate_state_vector_value(
        first_sample["context"],
        "fire_iii",
        "buff_state",
        "before.system.raid_buff_window.active",
    ) == 1.0
    assert candidate_state_vector_value(
        first_sample["context"],
        "fire_iii",
        "buff_state",
        "before.system.raid_buff_window.remaining_seconds",
    ) == 4.0

    assert fire_iii_at_downtime["is_legal"] is False
    assert fire_iii_at_downtime["invalid_reason"] == "boss_untargetable"
    assert second_sample["context"]["scene_context"] == first_sample["context"]["scene_context"]
    assert second_sample["context"]["scene_context"]["targetable_window_context"]["tokens"][1] == (
        targetable_window_token(2.0, 5.0, targetable=False, segment_kind="downtime")
    )
    assert second_sample["context"]["scene_context"]["raid_buff_window_context"]["tokens"][0] == (
        raid_buff_window_token(0.0, 4.0)
    )
    assert candidate_state_vector_value(
        second_sample["context"],
        "potion",
        "buff_state",
        "before.system.raid_buff_window.active",
    ) == 1.0
    assert candidate_state_vector_value(
        second_sample["context"],
        "potion",
        "buff_state",
        "before.system.raid_buff_window.remaining_seconds",
    ) == 1.0


def test_build_training_samples_keeps_boundary_action_before_inset_downtime(cs_backend, cs_skill_book):
    fight_payload = {
        "fight_id": "downtime_boundary_demo",
        "job_tag": "black_mage",
        "player": "Tester",
        "encounter": "Demo",
        "duration": 10.0,
        "scene_context": build_test_scene_context(
            targetable_tokens=[
                targetable_window_token(0.0, 3.0001, targetable=True, segment_kind="combat"),
                targetable_window_token(3.0001, 5.0, targetable=False, segment_kind="downtime"),
                targetable_window_token(5.0, 10.0, targetable=True, segment_kind="combat_final"),
            ],
        ),
        "actions": [
            {
                "time_offset": 0.0,
                "request_time_offset": 0.0,
                "time_gap": 0.0,
                "action_key": "lucid_dreaming",
                "fight_remaining": 10.0,
                "anchor": "combat",
            },
            {
                "time_offset": 3.0,
                "request_time_offset": 3.0,
                "time_gap": 3.0,
                "action_key": "fire_iii",
                "fight_remaining": 7.0,
                "anchor": "downtime_1",
                "boss_untargetable": True,
                "next_downtime_eta": 0.0,
            }
        ],
    }

    payload = build_training_samples(cs_backend, cs_skill_book, fight_payload)

    assert payload["samples"][-1]["label"]["action_key"] == "fire_iii"
    assert payload["samples"][-1]["label"]["is_legal"] is True


def _candidate_skill_token(context_payload: dict[str, object], skill_key: str) -> dict[str, object]:
    for token in context_payload["candidate_skill_context"]:
        if token["skill_key"] == skill_key:
            return token
    raise AssertionError(f"missing candidate skill token: {skill_key}")
