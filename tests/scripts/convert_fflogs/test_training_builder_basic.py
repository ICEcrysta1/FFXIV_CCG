"""训练样本构建的绝对时间回放测试。"""

from __future__ import annotations

import pytest

from scripts.convert_fflogs import build_training_samples
from common.policy.data.schema import TRAINING_SAMPLE_SCHEMA_VERSION
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

    assert payload["sample_schema_version"] == TRAINING_SAMPLE_SCHEMA_VERSION
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
    assert [sample["step"] for sample in payload["samples"]] == list(
        range(1, len(payload["samples"]) + 1)
    )
    source_step = 0
    for sample in payload["samples"]:
        if sample["label"]["action_key"] != "ogcd_wait":
            source_step += 1
        assert sample["source_step"] == source_step + (
            sample["label"]["action_key"] == "ogcd_wait"
        )


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


@pytest.mark.parametrize(
    "death_time", [0.0, 5.0, 5.123456789, 6.0],
    ids=["initial", "cancel_pending_effect", "fractional_timestamp", "keep_completed_history"],
)
@pytest.mark.parametrize("revival_kind", ["player_revived", "player_revived_lb3"])
def test_death_markers_and_reset_match_between_public_and_native_replay(
    cs_backend, cs_skill_book, death_time, revival_kind,
):
    """统一死亡输出保留原始时间和历史位置；两条回放路径的完整载荷保持一致。"""
    actions = []
    if death_time:
        actions.append({
            "request_time_offset": 0.0, "action_key": "fire_iii",
            "actual_cast_seconds": 0.0 if death_time == 6.0 else 10.0,
        })
    actions.extend([
        {"request_time_offset": 6.0 if death_time else 0.0, "action_key": "lucid_dreaming"},
        {"request_time_offset": 12.0, "action_key": "blizzard_iii", "actual_cast_seconds": 0.0},
    ])
    fight = _payload(actions, duration=20.0)
    fight["external_events"] = [
        {"timestamp": death_time, "event_kind": "player_death"},
        {"timestamp": 6.0 if death_time else 0.0, "event_kind": revival_kind},
    ]
    fight["scene_context"] = build_test_scene_context(targetable_tokens=[
        targetable_window_token(0.0, 5.0, targetable=True, segment_kind="combat"),
        targetable_window_token(5.0, 6.0, targetable=False, segment_kind="downtime"),
        targetable_window_token(6.0, 20.0, targetable=True, segment_kind="combat"),
    ])
    results = []
    for native in (False, True):
        cs_backend.init(actual_base_gcd=2.46, fight_remaining=20.0)
        results.append(build_training_samples(cs_backend, cs_skill_book, fight, deduplicate_history=native))
    assert results[0] == results[1]
    payload = results[0]
    assert payload["history_markers"] == [{
        "token": "<dead/>", "timestamp": death_time, "after_history_length": 2 if death_time == 6.0 else 0,
        "step": 3 if death_time == 6.0 else 2 if death_time else 1,
    }]
    assert "<dead/>" not in payload["resolved_sequence"]
    final_context = payload["samples"][-1]["context"]
    expected_history = ["fire_iii", "ogcd_wait", "lucid_dreaming"] if death_time == 6.0 else ["lucid_dreaming"]
    assert [row["skill_key"] for row in final_context["skill_history_context"]] == expected_history
    for sample in payload["samples"]:
        assert "history_markers" not in sample["context"]
        assert "<dead/>" not in sample["context"]["action_keys"]


@pytest.mark.parametrize("event", [
    {"timestamp": 0.0, "event_kind": "player_death", "mp": 2000},
    {"timestamp": 0.0, "event_kind": "player_revived", "mp": -1},
    {"timestamp": 0.0, "event_kind": "player_revived", "mp": True},
    {"timestamp": 0.0, "event_kind": "player_revived", "mp": 2.5},
    {"timestamp": float("nan"), "event_kind": "player_revived"},
    {"timestamp": 0.0, "event_kind": "player_revived_lb3", "mp": 0},
    {"timestamp": 0.0, "event_kind": "player_revived_lb3", "mp": 2000},
])
def test_player_lifecycle_facts_reject_invalid_payload(cs_backend, cs_skill_book, event):
    fight = _payload([{"request_time_offset": 0.0, "action_key": "lucid_dreaming"}])
    fight["external_events"] = [event]
    cs_backend.init(actual_base_gcd=2.46, fight_remaining=12.0)
    with pytest.raises(ValueError, match="external player|revival events accept"):
        build_training_samples(cs_backend, cs_skill_book, fight)


@pytest.mark.parametrize("history", [["before", "after"], ["before", "skipped"]])
def test_death_history_check_only_allows_cancellation_at_the_declared_boundary(history):
    """只允许丢弃死亡前未结算的尾部，死亡后的段内不能跳过技能。"""
    from scripts.convert_fflogs.training.training import _assert_no_label_leak

    context = {
        "skill_history_context": [{"skill_key": key, "skill_id": index + 1} for index, key in enumerate(history)],
        "state_history_context": {"tokens": [{}, {}]},
        "current_state_context": {"tokens": [{}]},
    }
    kwargs = {
        "expected_history_sequence": ["before", "cancelled", "after", "skipped"],
        "policy_action_key": "ogcd_wait", "current_action_key": "next", "current_step": 5,
        "history_markers": [{"token": "<dead/>", "timestamp": 5.0, "after_history_length": 1, "step": 3}],
    }
    if history[-1] == "after":
        _assert_no_label_leak(context, **kwargs)
    else:
        with pytest.raises(AssertionError, match="not a prefix"):
            _assert_no_label_leak(context, **kwargs)
