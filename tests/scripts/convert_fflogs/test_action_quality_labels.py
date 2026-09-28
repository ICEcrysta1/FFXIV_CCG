"""动作质量标签从精确原始事件到训练目标的关联测试。"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from common.policy.data.normalizer import Normalizer
from scripts.convert_fflogs import build_training_samples
from scripts.convert_fflogs.cache import cache_compile, precompile_raw_training_caches
from scripts.convert_fflogs.cache.cache_load import load_raw_compiled_cache
from scripts.convert_fflogs.config.config import load_action_quality_skill_policy
from scripts.convert_fflogs.extraction.action_quality import (
    attach_action_quality_labels,
)
from scripts.convert_fflogs.extraction.fight_payload import build_fight_payload
from training import TrainingDataset
from training.data.collator import TrainingCollator

REASON = "blm.rotation-watchdog.suggestions.coldf3.content"


def _annotated_report() -> dict[str, object]:
    events = [
        {"type": "cast", "timestamp": 1010, "sourceID": 7, "ability": {"guid": 152}},
        {"type": "cast", "timestamp": 1020, "sourceID": 7, "ability": {"guid": 152}},
    ]
    return {
        "fight_id": 3,
        "events": events,
        "analysis": {
            "schema_version": 2,
            "actor": {"id": "7"},
            "job_tag": "black_mage",
            "source": {"fight_id": 3},
            "time_basis": {"unit": "ms", "origin": "pull_start", "report_offset_ms": 1000},
            "fight_labels": [{
                "id": "suggestion-1",
                "content": {"props": {"id": REASON}},
                "severity_kind": "standard", "severity": "medium",
                "actions": [{
                    "event_type": "cast", "match_status": "exact",
                    "raw_event_indices": [1], "action_id": 152, "time_ms": 20,
                }],
            }],
        },
    }


def test_same_skill_repeated_only_exact_raw_event_is_labelled():
    assert REASON in load_action_quality_skill_policy("black_mage")
    report = _annotated_report()
    actions = [
        {"raw_event_index": 0, "skill_id": 152},
        {"raw_event_index": 1, "skill_id": 152},
    ]
    attach_action_quality_labels(report, actions, job_tag="black_mage", source_id=7)
    assert "quality_labels" not in actions[0]
    assert actions[1]["quality_labels"] == [{
        "reason_id": REASON, "severity": "medium", "raw_event_index": 1,
    }]


@pytest.mark.parametrize("change", [
    lambda report: report["analysis"]["fight_labels"][0]["actions"][0].update(time_ms=21),
    lambda report: report["analysis"]["fight_labels"][0]["actions"][0].update(action_id=153),
    lambda report: report["analysis"]["fight_labels"][0]["actions"][0].update(raw_event_indices=[2]),
    lambda report: report["analysis"]["actor"].update(id="8"),
])
def test_inconsistent_attribution_fails_closed(change):
    report = _annotated_report()
    change(report)
    with pytest.raises(ValueError):
        attach_action_quality_labels(report, [{"raw_event_index": 1, "skill_id": 152}],
                                     job_tag="black_mage", source_id=7)


def test_unconverted_labelled_cast_fails_and_unmatched_is_not_guessed():
    report = _annotated_report()
    with pytest.raises(ValueError, match="not converted"):
        attach_action_quality_labels(report, [{"raw_event_index": 0, "skill_id": 152}],
                                     job_tag="black_mage", source_id=7)
    report = copy.deepcopy(report)
    report["analysis"]["fight_labels"][0]["actions"][0]["match_status"] = "unmatched"
    actions = [{"raw_event_index": 0, "skill_id": 152}]
    attach_action_quality_labels(report, actions, job_tag="black_mage", source_id=7)
    assert "quality_labels" not in actions[0]


def test_older_annotation_without_event_type_must_be_regenerated():
    report = _annotated_report()
    del report["analysis"]["fight_labels"][0]["actions"][0]["event_type"]
    with pytest.raises(ValueError, match="regenerate annotation"):
        attach_action_quality_labels(report, [{"raw_event_index": 1, "skill_id": 152}],
                                     job_tag="black_mage", source_id=7)


def test_label_reaches_real_decision_not_policy_wait(cs_backend, cs_skill_book):
    action = {
        "timestamp": 1.0, "request_timestamp": 1.0, "action_key": "fire_iii",
        "skill_id": 152, "skill_name": "爆炎", "raw_event_index": 1,
        "quality_labels": [{"reason_id": REASON, "severity": "medium", "raw_event_index": 1}],
        "cast_time": 0.0, "is_damaging": True, "x": 0.0, "y": 0.0,
    }
    fight = build_fight_payload(
        [action], fight_id="quality_demo", player_name="Tester", encounter_name="Demo",
        report_code="quality_demo", source_id=7, job_tag="black_mage", gcd_time=2.5,
        generated_at="2026-09-28T00:00:00Z", downtime_gap_seconds=20.0,
        raid_buff_marker_keys=("amplifier",), raid_buff_window_duration=15.0,
    )
    result = build_training_samples(cs_backend, cs_skill_book, fight)
    real = [sample for sample in result["samples"] if sample["label"]["action_key"] == "fire_iii"]
    assert len(real) == 1
    assert real[0]["label"]["raw_event_index"] == 1
    assert real[0]["label"]["quality_labels"] == action["quality_labels"]
    assert all(not sample["label"].get("quality_labels")
               for sample in result["samples"] if sample["label"]["action_key"] == "ogcd_wait")


def test_compiled_pt_and_batch_keep_label_outside_model_inputs(
    cs_backend, cs_skill_book, tmp_path: Path, monkeypatch,
):
    torch = pytest.importorskip("torch")
    action = {
        "time_offset": 0.0, "request_time_offset": 0.0, "action_key": "fire_iii",
        "raw_event_index": 1, "quality_labels": [{
            "reason_id": REASON, "severity": "medium", "raw_event_index": 1,
        }],
    }
    from tests.helpers import build_test_scene_context, targetable_window_token

    fight = {
        "fight_id": "quality_pt", "job_tag": "black_mage", "duration": 10.0,
        "scene_context": build_test_scene_context(targetable_tokens=[
            targetable_window_token(0.0, 10.0, targetable=True, segment_kind="combat"),
        ]),
        "actions": [action],
    }
    training_payload = build_training_samples(cs_backend, cs_skill_book, fight)
    source = tmp_path / "annotated" / "FRU" / "00-10" / "quality.json"
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cache_compile, "convert_raw_file", lambda *_args, **_kwargs: (training_payload, {}))
    cache_root = tmp_path / ".cache"
    normalizer = Normalizer()
    normalizer.ensure_job_resources("black_mage")
    paths = precompile_raw_training_caches(
        [source], job_tag="black_mage", normalizer=normalizer,
        int_dtype=torch.int32, float_dtype=torch.float32,
        cache_dir=cache_root, shard_size=1, max_workers=1,
    )
    assert paths == [source]
    reader = load_raw_compiled_cache(
        source, cache_dir=cache_root, normalizer=normalizer,
        int_dtype=torch.int32, float_dtype=torch.float32, shard_size=1,
    )
    assert reader is not None
    saved = reader.sample(0)
    assert saved["metadata"]["step"] == 1
    assert saved["metadata"]["source_step"] == 1
    assert saved["raw_event_index"] == 1
    assert saved["quality_labels"] == action["quality_labels"]
    dataset = TrainingDataset(
        [source], job_tag="black_mage", normalizer=normalizer,
        int_dtype=torch.int32, float_dtype=torch.float32,
        cache_dir=cache_root, compiled_cache_shard_size=1,
    )
    batch = TrainingCollator()([dataset[0]])
    assert batch["quality_labels"] == [action["quality_labels"]]
    assert isinstance(batch["candidate_skill_features"], torch.Tensor)
