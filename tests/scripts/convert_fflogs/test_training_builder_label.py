"""训练样本 label 与 label leak 防护测试。"""

from __future__ import annotations

import pytest

from scripts.convert_fflogs import build_training_samples
from tests.helpers import build_test_scene_context, targetable_window_token


def _payload(actions: list[dict[str, object]], *, duration: float = 10.0) -> dict[str, object]:
    return {
        "fight_id": "label_demo",
        "job_tag": "black_mage",
        "player": "Tester",
        "encounter": "Demo",
        "duration": duration,
        "scene_context": build_test_scene_context(
            targetable_tokens=[
                targetable_window_token(0.0, duration, targetable=True, segment_kind="combat"),
            ],
        ),
        "actions": actions,
    }


def test_build_training_samples_uses_pre_action_context_as_label_input(
    cs_backend,
    cs_skill_book,
):
    fight_payload = _payload(
        [
            {
                "time_offset": 3.5,
                "request_time_offset": 0.0,
                "time_gap": 3.5,
                "action_key": "fire_iii",
                "fight_remaining": 6.5,
                "anchor": "combat",
            },
            {
                "time_offset": 3.5,
                "request_time_offset": 3.5,
                "time_gap": 0.0,
                "action_key": "amplifier",
                "fight_remaining": 6.5,
                "anchor": "combat",
            },
        ]
    )

    payload = build_training_samples(cs_backend, cs_skill_book, fight_payload)
    first_sample = payload["samples"][0]
    second_sample = payload["samples"][1]

    assert first_sample["context"]["skill_history_context"] == []
    assert first_sample["context"]["state_history_context"]["tokens"] == []
    assert second_sample["context"]["skill_history_context"][-1]["skill_key"] == "fire_iii"
    assert second_sample["context"]["state_history_context"]["tokens"]


def test_queued_label_matches_a_legal_output_action(cs_backend, cs_skill_book):
    cs_backend.init(actual_base_gcd=2.46, fight_remaining=10.0)
    fight_payload = _payload(
        [
            {
                "time_offset": 3.444,
                "request_time_offset": 0.0,
                "action_key": "fire_iii",
            },
            {
                "time_offset": 6.404,
                "request_time_offset": 2.96,
                "action_key": "blizzard_iii",
            },
        ]
    )

    payload = build_training_samples(cs_backend, cs_skill_book, fight_payload)
    sample = payload["samples"][1]
    index = sample["context"]["action_keys"].index("blizzard_iii")

    assert sample["label"]["queued"] is True
    assert sample["label"]["is_legal"] is True
    assert sample["context"]["action_legal_mask"][index] is True
    assert sample["label"]["action_index"] == index


def test_build_training_samples_raises_on_explicit_label_leak(cs_backend, cs_skill_book):
    seeded = cs_backend.submit_action(0.0, "lucid_dreaming")
    assert seeded.accepted
    fight_payload = _payload(
        [
            {
                "time_offset": 3.5,
                "request_time_offset": 0.0,
                "action_key": "fire_iii",
            }
        ]
    )

    with pytest.raises(AssertionError, match="label leak"):
        build_training_samples(cs_backend, cs_skill_book, fight_payload)


def test_find_action_index_raises_when_action_missing():
    """输出动作缺失时必须抛错，禁止静默返回 0 错标训练标签。"""
    from scripts.convert_fflogs.training.training import _find_action_index

    with pytest.raises(ValueError, match="not present in action_keys"):
        _find_action_index(["fire_iii"], "ogcd_wait")
