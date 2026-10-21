"""正式状态编码、技能归一化与场景 schema 测试。"""

from __future__ import annotations

from tests.training._causal_fixtures import make_state_groups

import pytest
import torch

from common.policy.data import Normalizer
from common.policy.data.context_encoding import ContextEncoder, raw_state_delta
from common.policy.data.schema import SceneWindowSchema, TrainingSchema
from common.policy.data.normalization import NormalizerConfig, load_normalizer_config
from common.policy.data.normalizer import NORMALIZER_CONTRACT_VERSION


def _encode_states(normalizer, feature_keys, rows):
    """从 raw ABS/DELTA 经正式入口编码，只有时间坐标使用有符号差分。"""
    contract = normalizer.normalization_contract
    normalizer = Normalizer.from_contract({**contract, "skill_charge_limits": {**contract["skill_charge_limits"], "first": 2, "second": 2}})
    keys = tuple(feature_keys)
    values = torch.tensor(rows, dtype=torch.float32)
    if "request_state.time_seconds" not in keys:
        keys += ("request_state.time_seconds",)
        times = 1200 + 3 * torch.arange(len(rows), dtype=torch.float32)
        values = torch.cat((values, times[:, None]), dim=1)
    schema = TrainingSchema(
        serialization_format="test", sample_schema_version=10, context_schema_version=14,
        scene_context_mode="absolute", skill_history_fields=("kind",),
        state_groups=make_state_groups({"state": keys}, ("first", "second")), state_snapshots=("previous_action_after", "request_state"),
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
    output = ContextEncoder(normalizer, schema, layout=schema.state_layout(("first", "second"))).encode({
        "history_state_skill_state": torch.zeros((*(values[:-1][None]).shape[:-1], 12), dtype=torch.float32),
        "current_state_skill_state": torch.zeros((*(values[-1:]).shape[:-1], 12), dtype=torch.float32),
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


@pytest.mark.parametrize("missing, extra, message", [
    (("time_scale", "mp_max"), {"unknown": 1},
     "normalizer contract config is missing: mp_max, time_scale"),
    ((), {"unknown": 1, "removed": 2},
     "normalizer contract config has removed or unknown fields: removed, unknown"),
])
def test_normalizer_contract_preserves_config_field_validation(missing, extra, message):
    """缺失字段按声明顺序报告，且优先于多余字段检查。"""
    contract = Normalizer().normalization_contract
    for key in missing:
        contract["config"].pop(key)
    contract["config"].update(extra)
    with pytest.raises(ValueError) as exc:
        Normalizer.from_contract(contract)
    assert str(exc.value) == message


def test_state_encoder_only_scales_declared_time_fields():
    encoded = _encode_states(Normalizer(), (
        "previous_action_after.some_seconds_counter", "previous_action_after.remaining_seconds",
    ), [[7, 60], [8, 0]])
    torch.testing.assert_close(encoded, torch.tensor([[7, 60 / 270], [8, 0]]))


def test_state_encoder_reanchors_player_time_and_preserves_small_deltas():
    encoded = _encode_states(Normalizer(), (
        "previous_action_after.time_seconds", "request_state.time_seconds",
    ), [[898, 900], [901, 903], [903, 907]])
    torch.testing.assert_close(encoded, torch.tensor([[-2 / 270, 0], [3 / 270, 3 / 270], [2 / 270, 4 / 270]]))


def test_normalizer_keeps_configured_duration_scale_when_job_cooldowns_change(monkeypatch):
    from dataclasses import replace

    import common.game.config as config_module

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
    normalizer = Normalizer(config=NormalizerConfig(time_scale=30, duration_scale=180))
    normalizer.configure_job_resources("black_mage")
    encoded = _encode_states(normalizer, ("previous_action_after.next_cooldown_seconds",), [[270], [135]])
    torch.testing.assert_close(encoded, torch.tensor([[1], [135 / 180]]))
    normalizer.configure_job_resources("black_mage")
    assert calls == ["black_mage"]


def test_normalizer_rejects_removed_skill_time_feature():
    with pytest.raises(ValueError, match="removed skill time/charge features"):
        Normalizer().normalize_skill_features(torch.tensor([[900.0]]), ("time_seconds",))


@pytest.mark.parametrize("rows,expected", [
    ([[1000], [2000]], [[.5], [1]]),
    ([[3000], [1000]], [[1], [.5]]),
])
def test_state_encoder_clips_absolute_potency_on_every_row(rows, expected):
    encoded = _encode_states(Normalizer(), ("previous_action_after.target.current_potency",), rows)
    torch.testing.assert_close(encoded, torch.tensor(expected, dtype=torch.float32))


@pytest.mark.parametrize("name", ["cumulative_potency", "cumulative_dot_potency"])
def test_normalizer_rejects_cumulative_potency_model_features(name):
    normalizer = Normalizer()
    with pytest.raises(ValueError, match="cumulative potency must not enter"):
        normalizer.register_feature_keys("state", (f"request_state.target.{name}",))
    with pytest.raises(ValueError, match="cumulative potency must not enter"):
        normalizer.normalize_skill_features(torch.tensor([[1000.0]]), (name,))


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


def test_state_encoder_does_not_infer_units_from_ready_or_timer_suffix():
    encoded = _encode_states(Normalizer(), (
        "previous_action_after.some_ready", "previous_action_after.some_timer",
    ), [[1, 60], [0, 0]])
    torch.testing.assert_close(encoded, torch.tensor([[1, 60], [0, 0]], dtype=torch.float32))


def test_normalizer_uses_job_resource_limits_for_state_and_skill_features():
    normalizer = Normalizer()
    normalizer.configure_job_resources("black_mage")
    encoded = _encode_states(normalizer, (
        "previous_action_after.astral_soul", "previous_action_after.polyglot_timer",
    ), [[99, 99], [0, 0]])
    torch.testing.assert_close(encoded, torch.tensor([[1, 99 / 270], [0, 0]]))
    normalized = normalizer.normalize_skill_features(
        torch.tensor([[6.0, 30.0]]), ("job_resources_after.astral_soul", "job_resources_after.polyglot_timer"),
    )
    torch.testing.assert_close(normalized, torch.tensor([[1, 30 / 270]]))


def test_state_encoder_uses_system_and_job_status_max_stacks():
    normalizer = Normalizer()
    normalizer.configure_job_resources("black_mage")
    encoded = _encode_states(normalizer, (
        "previous_action_after.system.burst_potion.stacks",
        "previous_action_after.job.triplecast.stacks",
        "previous_action_after.target.high_thunder.stacks",
    ), [[1, 3, 1], [0, 1, 1]])
    torch.testing.assert_close(encoded, torch.tensor([[1, 1, 1], [0, 1 / 3, 1]]))
    assert normalizer.cache_signature["status_limits"] == {
        "job.ley_lines": 1.0, "job.lucid_dreaming": 1.0, "job.swiftcast": 1.0, "job.triplecast": 3.0,
        "system.burst_potion": 1.0, "system.raid_buff_window": 1.0,
    }


@pytest.mark.parametrize("names", [
    ("max_charges", "potency", "kind", "is_legal"),
    ("potency", "kind", "is_legal", "max_charges"),
])
def test_skill_maximum_charge_ratios_use_frozen_limits_without_mutating_raw(names):
    """不同技能的充能上限不同，字段顺序不能改变每行的分母。"""
    rows = [dict(max_charges=2, potency=1000, kind=0, is_legal=1),
            dict(max_charges=3, potency=2000, kind=1, is_legal=0),
            dict(max_charges=1, potency=0, kind=0, is_legal=1)]
    raw = torch.tensor([[row[key] for key in names] for row in rows], dtype=torch.float32)
    saved = raw.clone()
    encoded = Normalizer().normalize_skill_features(raw, names)
    for key, values in {"max_charges": [1, 1, 1],
                        "potency": [.5, 1, 0], "kind": [0, 1, 0], "is_legal": [1, 0, 1]}.items():
        torch.testing.assert_close(encoded[:, names.index(key)], torch.tensor(values, dtype=torch.float32))
    torch.testing.assert_close(raw, saved, rtol=0, atol=0)


@pytest.mark.parametrize("maximum", [0, -1, float("nan"), float("inf")])
def test_skill_charge_ratios_reject_invalid_raw_limits(maximum):
    with pytest.raises(ValueError, match="charges"):
        Normalizer().normalize_skill_features(
            torch.tensor([[maximum]]), ("max_charges",),
        )


def test_skill_features_reject_moved_fields_and_allow_empty_history():
    normalizer = Normalizer()
    for field in ("available_charges", "next_cooldown_seconds"):
        with pytest.raises(ValueError, match="skill.*field|skill.*state|removed"):
            normalizer.normalize_skill_features(torch.tensor([[1.0]]), (field,))
    empty = torch.empty((0, 1))
    assert normalizer.normalize_skill_features(empty, ("max_charges",)).shape == (0, 1)


@pytest.mark.parametrize("job,resource", [("black_mage", "polyglot_timer"),
                                        ("machinist", "combo_remaining"),
                                        ("machinist", "wildfire_remaining")])
def test_time_resources_share_duration_scale_in_state_skill_and_saved_contract(job, resource, monkeypatch):
    normalizer = Normalizer(NormalizerConfig(time_scale=30, duration_scale=60))
    normalizer.configure_job_resources(job)
    contract = normalizer.normalization_contract
    assert resource in contract["resource_time_fields"]
    import common.game.config as config_module
    monkeypatch.setattr(config_module, "load_project_config", lambda **_kwargs: pytest.fail("恢复不得读取本机配置"))
    restored = Normalizer.from_contract(contract)
    restored.ensure_job_resources(job)
    torch.testing.assert_close(restored.normalize_skill_features(
        torch.tensor([[10.0]]), (f"job_resources_consumed.{resource}",),
    ), torch.tensor([[10 / 60]]))
    torch.testing.assert_close(_encode_states(restored, (f"request_state.{resource}",), [[10], [5]]),
                               torch.tensor([[10 / 60], [5 / 60]]))


def test_time_scales_come_from_saved_public_normalization_config(monkeypatch):
    import common.policy.data.normalizer as normalization_module
    monkeypatch.setattr(normalization_module, "load_normalizer_config", lambda _path=None: NormalizerConfig(time_scale=180, duration_scale=60))
    normalizer = Normalizer()
    assert normalizer.time_scale == 180
    assert normalizer.duration_scale == 60
    monkeypatch.setattr(normalization_module, "load_normalizer_config", lambda _path=None: pytest.fail("恢复不能重新读取配置"))
    restored = Normalizer.from_contract(normalizer.normalization_contract)
    assert restored.time_scale == 180
    assert restored.duration_scale == 60


@pytest.mark.parametrize("field", ["mp_max", "time_scale", "duration_scale", "target_count_max", "current_potency_max"])
@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf"),
                                 1e-46, 1e-300, 1e-38, 1e39, 1e300])
def test_public_normalization_scales_are_finite_positive_numbers(field, value):
    with pytest.raises(TypeError if value is True else ValueError, match=f"normalizer.{field}"):
        NormalizerConfig(**{field: value})


@pytest.mark.parametrize("key", ["resource_limits", "status_limits", "skill_charge_limits"])
@pytest.mark.parametrize("value", [1e-46, 1e39])
def test_saved_normalization_limits_must_be_representable_in_fp32(key, value):
    contract = Normalizer().normalization_contract
    contract[key] = {"test": value}
    with pytest.raises(ValueError, match="FP32"):
        Normalizer.from_contract(contract)
    if key == "resource_limits":
        with pytest.raises(ValueError, match="FP32"):
            Normalizer().register_resource_limits({"test": value})


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_skill_normalization_requires_fp32_before_calculation(dtype):
    with pytest.raises(ValueError, match="FP32"):
        Normalizer().normalize_skill_features(torch.tensor([[1000.0]], dtype=dtype), ("potency",))


def test_public_normalizer_yaml_preserves_only_active_scales():
    assert load_normalizer_config() == NormalizerConfig(mp_max=10000, time_scale=270, duration_scale=270,
                                                       target_count_max=5, current_potency_max=2000)
