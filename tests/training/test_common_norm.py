"""训练公共层归一化与 scene schema 测试。"""

from __future__ import annotations

import math

import pytest

from common.policy.data import Normalizer
from common.policy.data.schema import SceneWindowSchema, TrainingSchema
from common.policy.data.normalization import NormalizerConfig


def test_normalizer_does_not_treat_midfield_seconds_name_as_time_rule():
    normalizer = Normalizer()
    normalizer.register_feature_keys(
        "player_state",
        ["before.some_seconds_counter", "before.real_remaining_seconds"],
    )

    assert normalizer.normalize_value("player_state", "before.some_seconds_counter", 7.0) == 7.0
    assert normalizer.normalize_value("player_state", "before.real_remaining_seconds", 60.0) == pytest.approx(0.5)


def test_normalizer_uses_fight_time_max_for_player_time_seconds():
    normalizer = Normalizer()
    normalizer.register_feature_keys(
        "player_state",
        ["before.time_seconds", "before.fight_remaining_seconds"],
    )

    assert normalizer.normalize_value("player_state", "before.time_seconds", 900.0) == pytest.approx(0.5)
    assert normalizer.normalize_value(
        "player_state",
        "before.fight_remaining_seconds",
        900.0,
    ) == pytest.approx(0.5)
    assert normalizer.normalize_value(
        "player_state",
        "before.fight_remaining_seconds",
        1800.0,
    ) == pytest.approx(1.0)


def test_normalizer_caches_job_max_cooldown_for_remaining_seconds(monkeypatch):
    from dataclasses import replace

    import common.config as config_module

    project_config = config_module.load_project_config(job_tag="black_mage")
    custom_skills = (
        replace(project_config.job.skills[0], cooldown=135.0),
        *project_config.job.skills[1:],
    )
    custom_system_skills = (
        replace(project_config.system.skills[0], cooldown=270.0),
        *project_config.system.skills[1:],
    )
    project_config = replace(
        project_config,
        system=replace(project_config.system, skills=custom_system_skills),
        job=replace(project_config.job, skills=custom_skills),
    )
    calls: list[str] = []

    def fake_load_project_config(*, job_tag=None):
        calls.append(str(job_tag))
        return project_config

    monkeypatch.setattr(config_module, "load_project_config", fake_load_project_config)

    normalizer = Normalizer()
    normalizer.configure_job_resources("black_mage")
    normalizer.register_feature_keys("player_state", ["before.next_cooldown_seconds"])

    assert normalizer.normalize_value(
        "player_state",
        "before.next_cooldown_seconds",
        270.0,
    ) == pytest.approx(1.0)
    assert normalizer.normalize_value(
        "player_state",
        "before.next_cooldown_seconds",
        135.0,
    ) == pytest.approx(0.5)

    normalizer.configure_job_resources("black_mage")
    assert calls == ["black_mage"]


def test_normalizer_uses_fight_time_max_for_skill_time_seconds():
    torch = pytest.importorskip("torch")
    normalizer = Normalizer()
    values = torch.tensor([[900.0], [1800.0], [1900.0]])

    normalized = normalizer.normalize_skill_features(values, ("time_seconds",))

    assert normalized[:, 0].tolist() == pytest.approx([0.5, 1.0, 1.0])


def test_normalizer_uses_2500_current_potency_max():
    normalizer = Normalizer()
    normalizer.register_feature_keys(
        "target_buff_state",
        ["before.target.current_potency"],
    )

    assert normalizer.normalize_value(
        "target_buff_state",
        "before.target.current_potency",
        1250.0,
    ) == pytest.approx(0.5)
    assert normalizer.normalize_value(
        "target_buff_state",
        "before.target.current_potency",
        2500.0,
    ) == pytest.approx(1.0)
    assert normalizer.normalize_value(
        "target_buff_state",
        "before.target.current_potency",
        3000.0,
    ) == pytest.approx(1.0)


def test_normalizer_honors_cumulative_potency_mode():
    log1p_normalizer = Normalizer()
    log1p_normalizer.register_feature_keys(
        "target_buff_state",
        [
            "before.target.cumulative_potency",
            "before.target.current_gcd_dot_potency",
        ],
    )
    assert log1p_normalizer.normalize_value(
        "target_buff_state",
        "before.target.cumulative_potency",
        900.0,
    ) == pytest.approx(math.log1p(900.0))
    assert log1p_normalizer.normalize_value(
        "target_buff_state",
        "before.target.current_gcd_dot_potency",
        1250.0,
    ) == pytest.approx(0.5)

    divide_normalizer = Normalizer(config=NormalizerConfig(cumulative_potency_mode="divide"))
    divide_normalizer.register_feature_keys("target_buff_state", ["before.target.cumulative_potency"])
    assert divide_normalizer.normalize_value(
        "target_buff_state",
        "before.target.cumulative_potency",
        900.0,
    ) == pytest.approx(0.5)


def test_normalizer_scales_scene_time_to_1800_seconds_and_preserves_other_fields():
    torch = pytest.importorskip("torch")
    targetable_schema = SceneWindowSchema.from_feature_keys(
        context_key="targetable_window_context",
        feature_keys=("start_offset_seconds", "end_offset_seconds", "duration_seconds", "targetable"),
        scene_type_id=0,
    )
    target_count_schema = SceneWindowSchema.from_feature_keys(
        context_key="target_count_window_context",
        feature_keys=("start_offset_seconds", "end_offset_seconds", "duration_seconds", "target_count"),
        scene_type_id=3,
    )
    schema = TrainingSchema(
        serialization_format="test",
        sample_schema_version=1,
        context_schema_version=1,
        scene_context_mode="test",
        scene_windows=(targetable_schema, target_count_schema),
        state_group_feature_keys={},
        candidate_skill_fields=(),
        skill_history_fields=(),
    )
    values = torch.tensor(
        [
            [0.0, 120.0, 120.0, 1.0],
            [1700.0, 1900.0, 200.0, 0.0],
            [2000.0, 2200.0, 200.0, 8.0],
        ]
    )
    scene_types = torch.tensor([0, 0, 3])

    normalized = Normalizer().normalize_scene_tokens(values, scene_types, schema)

    assert normalized[:, 0].tolist() == pytest.approx([0.0, 1700.0 / 1800.0, 1.0])
    assert normalized[:, 1].tolist() == pytest.approx([120.0 / 1800.0, 1.0, 1.0])
    assert normalized[:, 2].tolist() == pytest.approx([120.0 / 1800.0, 100.0 / 1800.0, 0.0])
    assert normalized[:, 3].tolist() == pytest.approx([1.0, 0.0, 1.0])
    assert values.tolist() == [
        [0.0, 120.0, 120.0, 1.0],
        [1700.0, 1900.0, 200.0, 0.0],
        [2000.0, 2200.0, 200.0, 8.0],
    ]


def test_scene_window_schema_resolves_indices_once_and_reports_missing_fields():
    schema = SceneWindowSchema.from_feature_keys(
        context_key="test_window",
        feature_keys=("targetable", "duration_seconds", "start_offset_seconds", "end_offset_seconds"),
        scene_type_id=7,
    )

    assert schema.start_offset_index == 2
    assert schema.end_offset_index == 3
    assert schema.duration_index == 1

    with pytest.raises(ValueError, match="missing required feature keys: duration_seconds"):
        SceneWindowSchema.from_feature_keys(
            context_key="broken_window",
            feature_keys=("start_offset_seconds", "end_offset_seconds"),
            scene_type_id=7,
        )


def test_normalizer_infers_job_agnostic_ready_and_timer_fields():
    normalizer = Normalizer()
    normalizer.register_feature_keys("resource_state", ["before.some_ready", "before.some_timer"])

    assert normalizer.normalize_value("resource_state", "before.some_ready", 1.0) == 1.0
    assert normalizer.normalize_value("resource_state", "before.some_timer", 60.0) == pytest.approx(0.5)


def test_normalizer_uses_job_resource_limits_for_state_and_skill_features():
    torch = pytest.importorskip("torch")
    normalizer = Normalizer()
    normalizer.configure_job_resources("black_mage")
    normalizer.register_feature_keys("resource_state", ["before.astral_soul", "before.polyglot_timer"])

    assert normalizer.normalize_value("resource_state", "before.astral_soul", 99.0) == pytest.approx(1.0)
    assert normalizer.normalize_value("resource_state", "before.polyglot_timer", 99.0) == pytest.approx(1.0)

    values = torch.tensor([[6.0, 30.0]])
    normalized = normalizer.normalize_skill_features(
        values,
        ("job_resources_after.astral_soul", "job_resources_after.polyglot_timer"),
    )
    assert normalized[0].tolist() == pytest.approx([1.0, 1.0])


def test_normalizer_uses_system_and_job_status_max_stacks():
    torch = pytest.importorskip("torch")
    normalizer = Normalizer()
    normalizer.configure_job_resources("black_mage")
    normalizer.register_feature_keys(
        "buff_state",
        [
            "before.system.burst_potion.stacks",
            "before.job.triplecast.stacks",
            "before.target.high_thunder.stacks",
        ],
    )

    values = torch.tensor([[1.0, 3.0, 1.0], [0.0, 1.0, 1.0]])
    normalized = normalizer.normalize(values, "buff_state")

    normalized_rows = normalized.tolist()
    assert normalized_rows[0] == pytest.approx([1.0, 1.0, 1.0])
    assert normalized_rows[1] == pytest.approx([0.0, 1.0 / 3.0, 1.0])
    assert normalizer.cache_signature["status_limits"] == {
        "job.ley_lines": 1.0,
        "job.lucid_dreaming": 1.0,
        "job.manaward": 1.0,
        "job.surecast": 1.0,
        "job.swiftcast": 1.0,
        "job.triplecast": 3.0,
        "system.burst_potion": 1.0,
        "system.raid_buff_window": 1.0,
    }
