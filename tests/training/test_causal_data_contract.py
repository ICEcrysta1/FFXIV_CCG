"""阶段 1 的当前状态、固定输出监督和完整历史 bank 契约。"""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from common.output_context_schema import CANONICAL_CONTEXT_SCHEMA_VERSION
from common.policy.data import ActionSpace, SkillVocab
from common.policy.data.schema import TRAINING_SAMPLE_SCHEMA_VERSION
from scripts.convert_fflogs.source.source_reader import TrainingSourceReader
from scripts.convert_fflogs.training.history_bank import build_history_bank
from scripts.convert_fflogs.training.sample_builder import TrainingSampleBuilder
from tests.helpers import build_test_scene_context, targetable_window_token


def _source(*, history_length=0):
    space = ActionSpace.from_job_tag("black_mage")
    keys = {
        "player_state_feature_keys": ["previous_action_after.mp", "request_state.mp"],
        "buff_state_feature_keys": [],
        "target_buff_state_feature_keys": ["previous_action_after.target.cumulative_dot_potency", "request_state.target.cumulative_dot_potency"],
        "resource_state_feature_keys": [],
    }
    token = {"player_state": [10000.0, 10000.0], "buff_state": [], "target_buff_state": [0.0, 0.0], "resource_state": []}
    samples = []
    for step in range(history_length + 1):
        history = [{"skill_id": 0, "skill_key": "ogcd_wait", "kind": 0, "potency": 0.0} for _ in range(step)]
        samples.append({
            "step": step + 1,
            "context": {
                "job_tag": "black_mage", "schema_version": CANONICAL_CONTEXT_SCHEMA_VERSION,
                "history_cursor": len(history),
                "scene_context": build_test_scene_context(), "skill_history_context": history,
                "state_history_context": {**keys, "tokens": [deepcopy(token) for _ in history],
                                          "execution_metrics": [{"cumulative_dot_potency": 0.0} for _ in history]},
                "current_state_context": {**keys, "tokens": [deepcopy(token)]},
                "action_keys": list(space.action_keys), "action_legal_mask": [True] * len(space.action_keys),
                "action_values": [1.0] * len(space.action_keys),
            },
            "label": {"action_key": "ogcd_wait", "action_index": space.action_keys.index("ogcd_wait"), "cast_time_seconds": 0.0},
        })
    return {"sample_schema_version": TRAINING_SAMPLE_SCHEMA_VERSION, "job_tag": "black_mage", "fight_id": "synthetic", "fight_scene_context": build_test_scene_context(), "samples": samples}


def _builder(reader, **kwargs):
    kwargs.setdefault("float_dtype", torch.float32)
    if "history_bank" not in kwargs:
        kwargs["history_bank"] = build_history_bank(
            reader, torch=torch, normalizer=None,
            skill_vocab=SkillVocab.build_from_job_tag(reader.job_tag),
            skill_feature_names=reader.skill_feature_names,
            int_dtype=torch.int32, float_dtype=kwargs["float_dtype"],
        )
    return TrainingSampleBuilder(torch=torch, int_dtype=torch.int32,
                                 num_actions=reader.num_actions, **kwargs)


def test_empty_history_and_wait_only_history_share_stable_skill_schema():
    empty = TrainingSourceReader(_source())
    wait = TrainingSourceReader(_source(history_length=1))
    assert empty.schema == wait.schema
    assert empty.skill_feature_names == wait.skill_feature_names
    assert len(empty.skill_feature_names) == 18
    assert "time_seconds" not in empty.skill_feature_names
    assert "time_seconds" not in empty.schema.skill_history_fields
    empty_sample = _builder(empty).build(empty, 0)
    wait_sample = _builder(wait).build(wait, 1)
    assert empty_sample["history_length"] == 0
    assert wait_sample["history_length"] == 1
    assert empty_sample["current_state_abs_values"].shape == (empty.schema.state_vector_dim(),)
    assert wait_sample["current_state_delta_values"].tolist() == [0.0] * empty.schema.state_vector_dim()
    assert empty_sample["label_index"] == wait_sample["label_index"] == empty.action_keys.index("ogcd_wait")
    assert all("candidate" not in key for key in empty_sample)
    assert not any(key.startswith("history_skill") for key in empty_sample)


def test_current_state_preserves_explicit_missing_values_without_action_dimension():
    payload = _source()
    payload["samples"][0]["context"]["current_state_context"]["tokens"][0]["player_state"] = [None, None]
    reader = TrainingSourceReader(payload)
    raw = reader.current_state_matrix(0, dtype=torch.float32)
    assert raw.values.tolist() == [[0.0, 0.0, 0.0, 0.0]]
    assert raw.null_mask.tolist() == [[True, True, False, False]]
    sample = _builder(reader).build(reader, 0)
    assert sample["current_state_null_mask"].tolist() == [True, True, False, False]
    assert sample["current_state_abs_values"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert sample["current_state_delta_values"].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert sample["current_state_delta_reset_mask"].tolist() == [False, False, True, True]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float64])
def test_raw_reader_rejects_precision_changes_before_state_difference(dtype):
    """所有 raw 状态入口只输出 FP32，防止时间先降精度再差分。"""
    payload = _source(history_length=1)
    reader = TrainingSourceReader(payload)
    with pytest.raises(ValueError, match="require FP32"):
        reader.current_state_matrix(1, dtype=dtype)
    with pytest.raises(ValueError, match="require FP32"):
        reader.history_state_matrix(1, dtype=dtype)
    with pytest.raises(ValueError, match="require FP32"):
        reader.state_matrix_from_tokens(payload["samples"][1]["context"]["state_history_context"]["tokens"], dtype=dtype)


@pytest.mark.parametrize("change,error", [
    ("state_width", "current request state group width mismatch"),
    ("mask_width", "action_legal_mask must match"),
    ("action_order", "output action space must match"),
    ("nonfinite_value", "numeric request-time values"),
])
def test_source_rejects_invalid_current_request_contract(change, error):
    payload = _source()
    context = payload["samples"][0]["context"]
    if change == "state_width":
        context["current_state_context"]["tokens"][0]["player_state"].pop()
    elif change == "mask_width":
        context["action_legal_mask"].pop()
    elif change == "action_order":
        context["action_keys"].reverse()
    else:
        context["action_values"][0] = float("nan")
    with pytest.raises(ValueError, match=error):
        TrainingSourceReader(payload)


def test_sample_rejects_misaligned_fixed_output_label():
    payload = _source()
    payload["samples"][0]["label"]["action_index"] = 0
    reader = TrainingSourceReader(payload)
    with pytest.raises(ValueError, match="does not match output index"):
        _builder(reader).build(reader, 0)


def test_compile_builds_full_history_bank_without_read_window():
    reader = TrainingSourceReader(_source(history_length=5))
    vocab = SkillVocab.build_from_job_tag(reader.job_tag)
    bank = build_history_bank(reader, torch=torch, normalizer=None, skill_vocab=vocab,
                              skill_feature_names=reader.skill_feature_names,
                              int_dtype=torch.int32, float_dtype=torch.float32)
    assert bank["skill_ids"].shape[0] == 6
    sample = _builder(reader, history_bank=bank).build(reader, 5)
    assert sample["history_length"] == 5
    assert sample["history_end"] == 6
    assert sample["current_state_abs_values"].ndim == 1


def test_request_time_values_are_supervision_and_do_not_create_skill_inputs():
    payload = _source()
    reader = TrainingSourceReader(payload)
    baseline = _builder(reader).build(reader, 0)
    context = payload["samples"][0]["context"]
    context["action_values"][0] = 3.0
    reader = TrainingSourceReader(payload)
    changed = _builder(reader).build(reader, 0)
    assert changed["action_values"][0].item() == 3.0
    assert torch.equal(changed["current_state_abs_values"], baseline["current_state_abs_values"])
    assert changed["history_length"] == 0


@pytest.mark.parametrize("contract", ["sample", "canonical"])
def test_previous_internal_training_packet_versions_are_rejected(contract):
    payload = _source()
    if contract == "sample":
        payload["sample_schema_version"] -= 1
    else:
        payload["samples"][0]["context"]["schema_version"] -= 1
    with pytest.raises(ValueError, match="unsupported.*schema version"):
        TrainingSourceReader(payload)


def test_current_state_accepts_different_previous_and_request_snapshots():
    payload = _source()
    payload["samples"][0]["context"]["current_state_context"]["tokens"][0]["player_state"] = [8000.0, 9000.0]
    reader = TrainingSourceReader(payload)
    assert _builder(reader).build(reader, 0)["current_state_abs_values"][:2].tolist() == [8000.0, 9000.0]


def test_execution_metrics_are_independent_of_model_state_and_not_model_features():
    payload = _source(history_length=1)
    context = payload["samples"][1]["context"]
    context["state_history_context"]["execution_metrics"][0]["cumulative_dot_potency"] = 123.0
    context["state_history_context"]["tokens"][0]["target_buff_state"] = [1.0, 2.0]
    reader = TrainingSourceReader(payload)
    sample = _builder(reader).build(reader, 1)
    bank = build_history_bank(reader, torch=torch, normalizer=None,
                              skill_vocab=SkillVocab.build_from_job_tag(reader.job_tag),
                              skill_feature_names=reader.skill_feature_names,
                              int_dtype=torch.int32, float_dtype=torch.float32)
    assert bank["cumulative_dot_potencies"].tolist() == [0.0, 123.0]
    assert bank["state_abs_values"][1, -2:].tolist() == [1.0, 2.0]
    assert sample["history_end"] == 2


def _bank_for_payload(payload):
    reader = TrainingSourceReader(payload)
    return build_history_bank(
        reader, torch=torch, normalizer=None,
        skill_vocab=SkillVocab.build_from_job_tag(reader.job_tag),
        skill_feature_names=reader.skill_feature_names,
        int_dtype=torch.int32, float_dtype=torch.float32,
    )


@pytest.mark.parametrize("field", ["skill", "state", "execution_metrics"])
@pytest.mark.parametrize("same_length", [False, True])
def test_history_bank_rejects_changed_prefix_before_appending(field, same_length):
    payload = _source(history_length=2)
    if same_length:
        payload["samples"].insert(2, deepcopy(payload["samples"][1]))
    context = payload["samples"][2]["context"]
    if field == "skill":
        context["skill_history_context"][0]["potency"] = 99.0
    elif field == "state":
        context["state_history_context"]["tokens"][0]["player_state"][1] = 9000.0
    else:
        context["state_history_context"]["execution_metrics"][0]["cumulative_dot_potency"] = 60.0
    with pytest.raises(ValueError, match=rf"history prefix changed: sample=2 row=0 field={field}"):
        _bank_for_payload(payload)


def test_history_bank_rejects_reordered_prefix_when_history_grows():
    payload = _source(history_length=3)
    for sample in payload["samples"][2:]:
        sample["context"]["skill_history_context"][1]["potency"] = 10.0
    rows = payload["samples"][3]["context"]["skill_history_context"]
    rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError, match="history prefix changed: sample=3 row=0 field=skill"):
        _bank_for_payload(payload)


def test_history_bank_accepts_unchanged_history_between_multiple_settlements():
    payload = _source(history_length=3)
    # 相邻请求历史长度可以相同；之后一次结算两行仍逐行追加，不按请求数截取尾部。
    payload["samples"].insert(2, deepcopy(payload["samples"][1]))
    del payload["samples"][3]
    bank = _bank_for_payload(payload)
    assert bank["action_keys"] == ("", "ogcd_wait", "ogcd_wait", "ogcd_wait")
    assert bank["state_abs_values"].shape[0] == 4
    assert bank["cumulative_dot_potencies"].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_raw_state_bank_preserves_signed_mp_bool_and_damage_deltas_without_normalizing():
    from common.policy.data.normalizer import Normalizer

    payload = _source(history_length=3)
    keys = payload["samples"][0]["context"]["current_state_context"]["player_state_feature_keys"]
    keys[:] = ["previous_action_after.is_moving", "request_state.mp"]
    states = [(1.0, 10000.0, 50000.0), (0.0, 9200.0, 50700.0), (1.0, 10000.0, 51350.0)]
    for sample in payload["samples"]:
        context = sample["context"]
        context["state_history_context"]["player_state_feature_keys"] = list(keys)
        for token, (moving, mp, damage) in zip(context["state_history_context"]["tokens"], states):
            token["player_state"] = [moving, mp]
            token["target_buff_state"] = [damage, damage + 20.0]
    reader = TrainingSourceReader(payload)
    normalizer = Normalizer()
    normalizer.ensure_job_resources(reader.job_tag)
    normalizer.register_schema(reader.schema)
    bank = build_history_bank(
        reader, torch=torch, normalizer=normalizer,
        skill_vocab=SkillVocab.build_from_job_tag(reader.job_tag),
        skill_feature_names=reader.skill_feature_names,
        int_dtype=torch.int32, float_dtype=torch.bfloat16,
    )
    assert bank["state_abs_values"].dtype == bank["state_delta_values"].dtype == torch.float32
    assert bank["state_abs_values"][1].tolist() == [1.0, 10000.0, 50000.0, 50020.0]
    assert bank["state_delta_values"][2].tolist() == [-1.0, -800.0, 700.0, 700.0]
    assert bank["state_delta_values"][3].tolist() == [1.0, 800.0, 650.0, 650.0]
    assert bank["state_delta_reset_mask"][1].all()
    assert not bank["state_delta_reset_mask"][2:].any()
    torch.testing.assert_close(bank["state_abs_values"][1:].diff(dim=0), bank["state_delta_values"][2:])


def test_raw_history_and_current_state_reset_fields_when_null_becomes_known():
    payload = _source(history_length=2)
    for sample in payload["samples"][1:]:
        rows = sample["context"]["state_history_context"]["tokens"]
        rows[0]["player_state"] = [None, 10000.0]
        if len(rows) > 1:
            rows[1]["player_state"] = [7600.0, None]
    current = payload["samples"][2]["context"]["current_state_context"]["tokens"][0]
    current["player_state"] = [8000.0, 8200.0]
    reader = TrainingSourceReader(payload)
    builder = _builder(reader)
    bank = builder._history_bank
    assert bank["state_abs_values"][1, 0].item() == bank["state_delta_values"][1, 0].item() == 0.0
    assert bank["state_delta_values"][2, :2].tolist() == [7600.0, 0.0]
    assert bank["state_delta_reset_mask"][2, :2].tolist() == [True, False]
    sample = builder.build(reader, 2)
    assert sample["current_state_delta_values"][:2].tolist() == [400.0, 8200.0]
    assert sample["current_state_delta_reset_mask"][:2].tolist() == [False, True]


def test_raw_current_state_deltas_use_last_actual_history_not_previous_sample_request():
    payload = _source(history_length=1)
    payload["samples"][0]["context"]["current_state_context"]["tokens"][0]["player_state"] = [111.0, 222.0]
    context = payload["samples"][1]["context"]
    context["state_history_context"]["tokens"][0]["player_state"] = [7600.0, 7800.0]
    context["current_state_context"]["tokens"][0]["player_state"] = [8000.0, 8200.0]
    reader = TrainingSourceReader(payload)
    sample = _builder(reader).build(reader, 1)
    assert sample["current_state_abs_values"][:2].tolist() == [8000.0, 8200.0]
    assert sample["current_state_delta_values"][:2].tolist() == [400.0, 400.0]
    assert not sample["current_state_delta_reset_mask"].any()


def test_raw_scene_cache_keeps_windows_beyond_old_normalization_time_limit():
    payload = _source()
    payload["fight_scene_context"] = build_test_scene_context(
        targetable_tokens=[targetable_window_token(1800.0, 2400.0, targetable=True, segment_kind="combat")],
    )
    reader = TrainingSourceReader(payload)
    sample = _builder(reader, float_dtype=torch.bfloat16).build(reader, 0)
    assert sample["scene_vectors"].dtype == torch.float32
    assert sample["scene_vectors"][0, :3].tolist() == [1800.0, 2400.0, 600.0]
