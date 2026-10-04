"""动作质量标签从精确原始事件到训练目标的关联测试。"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from common.policy.data.normalizer import Normalizer
from common.policy.data.action_space import ActionSpace
from scripts.convert_fflogs import (
    build_skill_book,
    build_training_samples,
    convert_report_payload,
    load_job_project_config,
)
from scripts.convert_fflogs.cache import cache_compile, precompile_raw_training_caches
from scripts.convert_fflogs.cache.cache_load import load_raw_compiled_cache
from scripts.convert_fflogs.config.config import load_action_quality_skill_policy
from scripts.convert_fflogs.extraction.action_quality import (
    attach_action_quality_labels,
)
from scripts.convert_fflogs.extraction.fight_payload import build_fight_payload
from scripts.convert_fflogs.training.quality_supervision import source_ranking
from training import TrainingDataset
from training.config import load_run_config
from training.data.collator import TrainingCollator
from training.loop.losses.action_quality import action_quality_sample_weights

REASON = "blm.rotation-watchdog.suggestions.coldf3.content"


def test_machinist_annotated_report_converts_with_skill_labels_disabled():
    assert load_action_quality_skill_policy("machinist") == frozenset()
    events = [
        {
            "type": "cast", "timestamp": timestamp, "sourceID": 7,
            "abilityGameID": ability_id,
            "ability": {"guid": ability_id, "name": name},
            "sourceResources": {"x": 0.0, "y": 0.0},
        }
        for timestamp, ability_id, name in (
            (1010, 7411, "热分裂弹"),
            (3510, 7412, "热独头弹"),
        )
    ]
    report = {
        "fight_id": 3, "events": events,
        "analysis": {
            "schema_version": 2, "bridge_version": 4,
            "actor": {"id": "7"},
            "job_tag": "machinist",
            "source": {"fight_id": 3},
            "time_basis": {"unit": "ms", "origin": "pull_start", "report_offset_ms": 1000},
            "fight_labels": [{
                "content": {"props": {"id": "core.aoeusages.suggestion.content"}},
                "severity_kind": "standard", "severity": "major",
                "actions": [{
                    "event_type": "cast", "match_status": "exact",
                    "raw_event_indices": [0], "action_id": 7411, "time_ms": 10,
                }],
            }],
        },
    }
    project_config = load_job_project_config("machinist")
    fight, ignored = convert_report_payload(
        report,
        job_tag="machinist",
        project_config=project_config,
        skill_book=build_skill_book(project_config),
        source_id=7,
        encounter_name="Demo",
        report_code="demo",
        player_name="Tester",
        generated_at="2026-09-28T00:00:00Z",
    )
    assert ignored == {}
    assert fight is not None
    assert [action["action_key"] for action in fight["actions"]] == [
        "heated_split_shot", "heated_slug_shot",
    ]
    assert all(not action["quality_labels"] for action in fight["actions"])


def _annotated_report() -> dict[str, object]:
    events = [
        {"type": "cast", "timestamp": 1010, "sourceID": 7, "ability": {"guid": 152}},
        {"type": "cast", "timestamp": 1020, "sourceID": 7, "ability": {"guid": 152}},
    ]
    return {
        "fight_id": 3,
        "events": events,
        "analysis": {
            "schema_version": 2, "bridge_version": 4,
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


def test_flat_ability_game_id_cast_receives_exact_label():
    report = _annotated_report()
    report["events"][1] = {
        "type": "cast", "timestamp": 1020, "sourceID": 7, "abilityGameID": 152,
    }
    actions = [{"raw_event_index": 1, "skill_id": 152}]

    attach_action_quality_labels(report, actions, job_tag="black_mage", source_id=7)

    assert actions[0]["quality_labels"] == [{
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


def test_legacy_weighted_annotation_must_be_regenerated():
    report = _annotated_report()
    report["analysis"]["bridge_version"] = 2
    report["analysis"]["severity_weights"] = {"medium": 0.5}
    report["analysis"]["fight_labels"][0]["severity_weight"] = 0.5
    with pytest.raises(ValueError, match="bridge version 4"):
        attach_action_quality_labels(report, [{"raw_event_index": 1, "skill_id": 152}],
                                     job_tag="black_mage", source_id=7)
    report["analysis"]["bridge_version"] = 4
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


@pytest.mark.parametrize("annotation_status, with_labels, expected_status", [
    ("partial", True, "attributed_label"),
    ("partial", False, "no_attributed_label"),
    ("unannotated", False, "unannotated"),
])
@pytest.mark.parametrize("percentile, bucket", [
    (None, "00-10"),
    (7.5, "00-10"),
    (97.77, "90-100"),
])
def test_compiled_pt_and_batch_keep_label_outside_model_inputs(
    cs_backend, cs_skill_book, tmp_path: Path, monkeypatch,
    annotation_status, with_labels, expected_status, percentile, bucket,
):
    torch = pytest.importorskip("torch")
    labels = ([
        {"reason_id": REASON, "severity": "minor", "raw_event_index": 1},
        {"reason_id": "medium.error", "severity": "medium", "raw_event_index": 1},
        {"reason_id": "second.error", "severity": "major", "raw_event_index": 1},
    ] if with_labels else [])
    action = {
        "time_offset": 0.0, "request_time_offset": 0.0, "action_key": "fire_iii",
        "raw_event_index": 1, "quality_labels": labels,
    }
    from tests.helpers import build_test_scene_context, targetable_window_token

    fight = {
        "fight_id": "quality_pt", "job_tag": "black_mage", "duration": 10.0,
        "annotation_status": annotation_status,
        "ranking": {"percentile": percentile, "percentile_bucket": bucket},
        "scene_context": build_test_scene_context(targetable_tokens=[
            targetable_window_token(0.0, 10.0, targetable=True, segment_kind="combat"),
        ]),
        "actions": [action],
    }
    if percentile is None:
        fight.pop("ranking")
    training_payload = build_training_samples(cs_backend, cs_skill_book, fight)
    source = tmp_path / "annotated" / "FRU" / bucket / "quality.json"
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
        expected_action_space=ActionSpace.from_job_tag("black_mage"),
        int_dtype=torch.int32, float_dtype=torch.float32, shard_size=1,
    )
    assert reader is not None
    saved = reader.sample(0)
    assert saved["metadata"]["step"] == 1
    assert saved["metadata"]["source_step"] == 1
    assert saved["raw_event_index"] == 1
    assert saved["quality_labels"] == action["quality_labels"]
    assert saved["metadata"]["percentile"] == percentile
    assert saved["metadata"]["percentile_bucket"] == bucket
    assert saved["metadata"]["quality_label_status"] == expected_status
    assert saved["quality_label_levels"].dtype == torch.int32
    assert saved["quality_label_levels"].tolist() == ([1, 2, 3] if with_labels else [])
    assert "quality_label_weights" not in saved
    dataset = TrainingDataset(
        [source], job_tag="black_mage", normalizer=normalizer,
        expected_action_space=ActionSpace.from_job_tag("black_mage"),
        int_dtype=torch.int32, float_dtype=torch.float32,
        cache_dir=cache_root, compiled_cache_shard_size=1,
    )
    batch = TrainingCollator()([dataset[0]])
    assert batch["quality_label_levels"].dtype == torch.int32
    assert batch["label_index"].dtype == torch.int64
    assert batch["source_quality"].dtype == torch.float32
    assert batch["quality_label_levels"].tolist() == ([[1, 2, 3]] if with_labels else [[]])
    assert batch["quality_label_mask"].tolist() == ([[True, True, True]] if with_labels else [[]])
    assert "quality_label_weights" not in batch
    assert batch["quality_annotation_available"].tolist() == [annotation_status == "partial"]
    normalized_quality = -1.0 if percentile is None else percentile / 100.0
    assert batch["source_quality"].tolist() == pytest.approx([normalized_quality])
    assert "source_percentile" not in batch
    assert "source_percentile_bucket_lower" not in batch
    assert isinstance(batch["current_state_vectors"], torch.Tensor)
    quality_config = load_run_config("config/models/black_mage/artzip/config.yaml").action_quality_loss
    if percentile is None:
        if with_labels:
            with pytest.raises(ValueError, match="percentile"):
                TrainingCollator(require_quality_percentile=True)([dataset[0]])
        quality_config = type(quality_config)(enabled=False)
    weight = action_quality_sample_weights(batch, quality_config).item()
    expected_weight = (
        1.0 - math.exp(-((percentile / 100.0 / 0.6) ** 4))
        if with_labels and percentile is not None else 1.0
    )
    assert weight == pytest.approx(expected_weight)
    if with_labels:
        empty_label_sample = {**dataset[0], "quality_label_levels": torch.empty(0, dtype=torch.int32)}
        mixed_batch = TrainingCollator()([dataset[0], empty_label_sample])
        assert mixed_batch["quality_label_levels"].tolist() == [[1, 2, 3], [0, 0, 0]]
        assert mixed_batch["quality_label_mask"].tolist() == [[True, True, True], [False, False, False]]
    if with_labels and percentile == 7.5:
        from common.policy import config as policy_config

        monkeypatch.setenv("FFXIV_MODEL_VARIANT", "different_quality_weights")
        monkeypatch.setattr(
            policy_config, "load_action_quality_severity_weights",
            lambda _job: {"minor": 0.1, "medium": 0.2, "major": 0.3},
        )
        monkeypatch.setattr(
            cache_compile, "convert_raw_file",
            lambda *_args, **_kwargs: pytest.fail("changing model weights must reuse compiled PT"),
        )
        assert precompile_raw_training_caches(
            [source], job_tag="black_mage", normalizer=normalizer,
            int_dtype=torch.int32, float_dtype=torch.float32,
            cache_dir=cache_root, shard_size=1, max_workers=1,
        ) == [source]
        assert reader.sample(0)["quality_label_levels"].tolist() == [1, 2, 3]


def test_source_ranking_checks_directory_and_preserves_unknown_percentile(tmp_path):
    source = tmp_path / "annotated" / "FRU" / "00-10" / "fight.json"
    assert source_ranking(source, None) == {
        "percentile": None, "percentile_bucket": "00-10",
    }
    with pytest.raises(ValueError, match="directory disagree"):
        source_ranking(source, {"percentile": 95.0, "percentile_bucket": "90-100"})
