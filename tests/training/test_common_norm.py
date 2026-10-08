"""正式状态编码、技能归一化与场景 schema 测试。"""

from __future__ import annotations

import math

import pytest
import torch

from common.policy.config import ModelConfig
from common.policy.data import Normalizer
from common.policy.data.context_encoding import ContextEncoder, raw_state_delta
from common.policy.data.schema import SceneWindowSchema, TrainingSchema
from common.policy.data.normalization import NormalizerConfig
from common.policy.data.normalizer import NORMALIZER_CONTRACT_VERSION


def _encode_states(normalizer, feature_keys, rows):
    """从 raw ABS/DELTA 经正式入口编码，覆盖首条绝对锚点和后续有符号差分。"""
    keys = tuple(feature_keys)
    values = torch.tensor(rows, dtype=torch.float32)
    if "request_state.time_seconds" not in keys:
        keys += ("request_state.time_seconds",)
        times = 1200 + 3 * torch.arange(len(rows), dtype=torch.float32)
        values = torch.cat((values, times[:, None]), dim=1)
    schema = TrainingSchema(
        serialization_format="test", sample_schema_version=10, context_schema_version=14,
        scene_context_mode="absolute", skill_history_fields=("kind",),
        state_group_feature_keys={"state": keys},
        scene_windows=(SceneWindowSchema.from_feature_keys(
            context_key="combat", scene_type_id=0,
            feature_keys=("start_offset_seconds", "end_offset_seconds", "duration_seconds"),
        ),),
    )
    nulls = torch.zeros_like(values, dtype=torch.bool)
    delta, reset = raw_state_delta(
        values, nulls, torch.cat((torch.zeros_like(values[:1]), values[:-1])),
        torch.cat((torch.ones_like(nulls[:1]), nulls[:-1])),
    )
    history_size = len(rows) - 1
    output = ContextEncoder(normalizer, schema, ModelConfig()).encode({
        "history_skill_ids": torch.ones((1, history_size), dtype=torch.long),
        "history_skill_features": torch.zeros((1, history_size, 1)),
        "history_state_abs_values": values[:-1][None],
        "history_state_delta_values": delta[:-1][None],
        "history_state_null_mask": nulls[:-1][None],
        "history_state_delta_reset_mask": reset[:-1][None],
        "history_mask": torch.ones((1, history_size), dtype=torch.bool),
        "current_state_abs_values": values[-1:],
        "current_state_delta_values": delta[-1:],
        "current_state_null_mask": nulls[-1:],
        "current_state_delta_reset_mask": reset[-1:],
        "scene_abs_values": torch.zeros((1, 0, 3)),
        "scene_types": torch.zeros((1, 0), dtype=torch.long),
        "scene_mask": torch.zeros((1, 0), dtype=torch.bool),
    })
    return torch.cat((output["history_state_vectors"][0], output["current_state_vectors"]))[:, :len(feature_keys)]


def test_normalizer_contract_rejects_previous_gcd_time_unit_config():
    """不把旧 GCD 单位上限静默带进新的模型输入契约。"""
    contract = Normalizer().normalization_contract
    assert "remaining_gcds_max" not in contract["config"]
    contract["version"] = NORMALIZER_CONTRACT_VERSION - 1
    contract["config"]["remaining_gcds_max"] = 48.0
    with pytest.raises(ValueError, match="unsupported normalizer contract version"):
        Normalizer.from_contract(contract)


def test_state_encoder_does_not_treat_midfield_seconds_name_as_time_rule():
    encoded = _encode_states(Normalizer(), (
        "previous_action_after.some_seconds_counter", "previous_action_after.real_remaining_seconds",
    ), [[7, 60], [8, 0]])
    torch.testing.assert_close(encoded, torch.tensor([[7, .5], [1, -.5]]))


def test_state_encoder_reanchors_player_time_and_preserves_small_deltas():
    encoded = _encode_states(Normalizer(), (
        "previous_action_after.time_seconds", "request_state.time_seconds",
    ), [[898, 900], [901, 903], [903, 907]])
    torch.testing.assert_close(encoded, torch.tensor([[-2 / 120, 0], [3 / 120, 3 / 120], [2 / 120, 4 / 120]]))


def test_normalizer_caches_job_max_cooldown_for_remaining_seconds(monkeypatch):
    from dataclasses import replace

    import common.config as config_module

    project_config = config_module.load_project_config(job_tag="black_mage")
    custom_skills = (replace(project_config.job.skills[0], cooldown=135.0), *project_config.job.skills[1:])
    custom_system_skills = (replace(project_config.system.skills[0], cooldown=270.0), *project_config.system.skills[1:])
    project_config = replace(
        project_config,
        system=replace(project_config.system, skills=custom_system_skills),
        job=replace(project_config.job, skills=custom_skills),
    )
    calls = []

    def fake_load_project_config(*, job_tag=None):
        calls.append(str(job_tag))
        return project_config

    monkeypatch.setattr(config_module, "load_project_config", fake_load_project_config)
    normalizer = Normalizer()
    normalizer.configure_job_resources("black_mage")
    encoded = _encode_states(normalizer, ("previous_action_after.next_cooldown_seconds",), [[270], [135]])
    torch.testing.assert_close(encoded, torch.tensor([[1], [-.5]]))
    normalizer.configure_job_resources("black_mage")
    assert calls == ["black_mage"]


def test_normalizer_rejects_removed_skill_time_feature():
    with pytest.raises(ValueError, match="skill features must not include time_seconds"):
        Normalizer().normalize_skill_features(torch.tensor([[900.0]]), ("time_seconds",))


@pytest.mark.parametrize("rows,expected", [
    ([[1250], [2500]], [[.5], [.5]]),
    ([[3000], [1250]], [[1], [-.7]]),
])
def test_state_encoder_clips_absolute_potency_but_keeps_signed_delta(rows, expected):
    encoded = _encode_states(Normalizer(), ("previous_action_after.target.current_potency",), rows)
    torch.testing.assert_close(encoded, torch.tensor(expected, dtype=torch.float32))


@pytest.mark.parametrize("mode,initial,change", [
    ("log1p", math.log1p(900), -math.log1p(50)),
    ("divide", .5, -50 / 1800),
])
def test_state_encoder_honors_cumulative_potency_mode_for_signed_deltas(mode, initial, change):
    normalizer = Normalizer(config=NormalizerConfig(cumulative_potency_mode=mode))
    encoded = _encode_states(normalizer, (
        "previous_action_after.target.cumulative_potency",
        "previous_action_after.target.current_gcd_dot_potency",
    ), [[900, 1250], [850, 1000]])
    torch.testing.assert_close(encoded, torch.tensor([[initial, .5], [change, -.1]]))


def test_scene_window_schema_resolves_indices_once_and_reports_missing_fields():
    schema = SceneWindowSchema.from_feature_keys(
        context_key="test_window", scene_type_id=7,
        feature_keys=("targetable", "duration_seconds", "start_offset_seconds", "end_offset_seconds"),
    )
    assert schema.start_offset_index == 2
    assert schema.end_offset_index == 3
    assert schema.duration_index == 1
    with pytest.raises(ValueError, match="missing required feature keys: duration_seconds"):
        SceneWindowSchema.from_feature_keys(
            context_key="broken_window", scene_type_id=7,
            feature_keys=("start_offset_seconds", "end_offset_seconds"),
        )


def test_state_encoder_infers_job_agnostic_ready_and_timer_fields():
    encoded = _encode_states(Normalizer(), (
        "previous_action_after.some_ready", "previous_action_after.some_timer",
    ), [[1, 60], [0, 0]])
    torch.testing.assert_close(encoded, torch.tensor([[1, .5], [-1, -.5]]))


def test_normalizer_uses_job_resource_limits_for_state_and_skill_features():
    normalizer = Normalizer()
    normalizer.configure_job_resources("black_mage")
    encoded = _encode_states(normalizer, (
        "previous_action_after.astral_soul", "previous_action_after.polyglot_timer",
    ), [[99, 99], [0, 0]])
    torch.testing.assert_close(encoded, torch.tensor([[1, 1], [-99 / 6, -99 / 30]]))
    normalized = normalizer.normalize_skill_features(
        torch.tensor([[6.0, 30.0]]), ("job_resources_after.astral_soul", "job_resources_after.polyglot_timer"),
    )
    torch.testing.assert_close(normalized, torch.ones((1, 2)))


def test_state_encoder_uses_system_and_job_status_max_stacks():
    normalizer = Normalizer()
    normalizer.configure_job_resources("black_mage")
    encoded = _encode_states(normalizer, (
        "previous_action_after.system.burst_potion.stacks",
        "previous_action_after.job.triplecast.stacks",
        "previous_action_after.target.high_thunder.stacks",
    ), [[1, 3, 1], [0, 1, 1]])
    torch.testing.assert_close(encoded, torch.tensor([[1, 1, 1], [-1, -2 / 3, 0]]))
    assert normalizer.cache_signature["status_limits"] == {
        "job.ley_lines": 1.0, "job.lucid_dreaming": 1.0, "job.swiftcast": 1.0, "job.triplecast": 3.0,
        "system.burst_potion": 1.0, "system.raid_buff_window": 1.0,
    }
