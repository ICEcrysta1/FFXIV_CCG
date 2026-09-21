"""训练样本构建的绝对时间回放测试。"""

from __future__ import annotations

import pytest

from scripts.convert_fflogs import build_training_samples
from tests.helpers import build_test_scene_context, targetable_window_token


def _payload(actions: list[dict[str, object]], *, duration: float = 12.0) -> dict[str, object]:
    return {
        "fight_id": "absolute_timeline_demo",
        "job_tag": "black_mage",
        "player": "Tester",
        "encounter": "Demo",
        "duration": duration,
        "gcd_time": 2.46,
        "scene_context": build_test_scene_context(
            targetable_tokens=[
                targetable_window_token(0.0, duration, targetable=True, segment_kind="combat"),
            ],
        ),
        "actions": actions,
    }


def test_build_training_samples_uses_queue_for_sub_window_network_residual(
    cs_backend,
    cs_skill_book,
):
    """请求落在前一硬读条结束前 0.5 秒内时，由状态机容量一队列接住。"""
    cs_backend.init(actual_base_gcd=2.46, fight_remaining=12.0)
    fight_payload = _payload(
        [
            {
                "time_offset": 3.0,
                "request_time_offset": 0.0,
                "time_gap": 3.0,
                "action_key": "fire_iii",
                "fight_remaining": 9.0,
                "anchor": "combat",
                "cast_timing_source": "prepull_estimated",
            },
            {
                "time_offset": 5.96,
                "request_time_offset": 2.96,
                "time_gap": 2.96,
                "action_key": "blizzard_iii",
                "fight_remaining": 6.04,
                "anchor": "combat",
                "cast_timing_source": "begincast_duration",
            },
        ]
    )

    payload = build_training_samples(cs_backend, cs_skill_book, fight_payload)

    assert payload["sample_schema_version"] == 7
    assert payload["resolved_sequence"] == ["fire_iii", "blizzard_iii"]
    assert payload["samples"][0]["time_offset"] == 0.0
    assert payload["samples"][1]["time_offset"] == pytest.approx(2.96)
    assert payload["samples"][1]["label"]["queued"] is True
    assert payload["samples"][1]["label"]["timing_retry_count"] == 0
    assert payload["samples"][1]["label"]["request_time_clamped"] == 0.0


def test_build_training_samples_records_ogcd_wait_as_policy_action(
    cs_backend,
    cs_skill_book,
):
    cs_backend.init(actual_base_gcd=2.46, fight_remaining=12.0)
    fight_payload = _payload(
        [
            {
                "time_offset": 3.444,
                "request_time_offset": 0.0,
                "time_gap": 3.444,
                "action_key": "fire_iii",
                "fight_remaining": 8.556,
                "anchor": "combat",
            },
            {
                "time_offset": 3.444,
                "request_time_offset": 3.444,
                "time_gap": 0.0,
                "action_key": "high_thunder",
                "fight_remaining": 8.556,
                "anchor": "combat",
            },
            {
                "time_offset": 6.0,
                "request_time_offset": 6.0,
                "time_gap": 2.556,
                "action_key": "blizzard_iii",
                "fight_remaining": 6.0,
                "anchor": "combat",
            },
            {
                "time_offset": 9.5,
                "request_time_offset": 9.5,
                "time_gap": 3.5,
                "action_key": "fire_iii",
                "fight_remaining": 2.5,
                "anchor": "combat",
            },
        ]
    )

    payload = build_training_samples(cs_backend, cs_skill_book, fight_payload)

    assert payload["resolved_sequence"][:4] == [
        "fire_iii",
        "high_thunder",
        "ogcd_wait",
        "blizzard_iii",
    ]
    wait_samples = [
        sample for sample in payload["samples"]
        if sample["label"]["action_key"] == "ogcd_wait"
    ]
    assert wait_samples
    assert all(sample["label"]["skill_id"] == 0 for sample in wait_samples)


def test_build_training_samples_rejects_non_normalized_request_origin(
    cs_backend,
    cs_skill_book,
):
    fight_payload = _payload(
        [
            {
                "time_offset": 1.0,
                "request_time_offset": 1.0,
                "action_key": "lucid_dreaming",
            }
        ]
    )

    with pytest.raises(ValueError, match="must start at 0"):
        build_training_samples(cs_backend, cs_skill_book, fight_payload)
