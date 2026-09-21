"""模型输入契约和归一化契约测试。"""

from __future__ import annotations

import pytest

from common.policy.data import ModelInputContract, Normalizer
from common.policy.data.schema import SceneWindowSchema, TrainingSchema
from common.policy.data.spec import DataSpec


def _build_contract() -> ModelInputContract:
    data_spec = DataSpec(
        job_tag="black_mage",
        num_candidates=1,
        state_dim=1,
        scene_dim=3,
        skill_feature_dim=1,
        num_scene_types=1,
        candidate_action_keys=("fire_iii",),
        skill_feature_names=("potency",),
    )
    schema = TrainingSchema(
        serialization_format="test",
        sample_schema_version=1,
        context_schema_version=1,
        scene_context_mode="absolute",
        scene_windows=(
            SceneWindowSchema.from_feature_keys(
                context_key="targetable_window_context",
                feature_keys=(
                    "start_offset_seconds",
                    "end_offset_seconds",
                    "duration_seconds",
                ),
                scene_type_id=0,
            ),
        ),
        state_group_feature_keys={"player_state": ("before.time_seconds",)},
        candidate_skill_fields=("potency",),
        skill_history_fields=("skill_key",),
    )
    normalizer = Normalizer()
    normalizer.configure_job_resources("black_mage")
    return ModelInputContract.from_training(
        data_spec=data_spec,
        schema=schema,
        normalizer=normalizer,
    )


def test_model_input_contract_round_trips_without_project_yaml(monkeypatch):
    contract = _build_contract()
    import common.config as config_module

    monkeypatch.setattr(
        config_module,
        "load_project_config",
        lambda **_kwargs: pytest.fail("checkpoint contract must not load project YAML"),
    )
    restored = ModelInputContract.from_dict(contract.to_dict())

    assert restored.job_tag == "black_mage"
    assert restored.data_spec == contract.data_spec
    assert restored.schema == contract.schema
    assert restored.normalizer_contract == contract.normalizer_contract

    normalizer = restored.create_normalizer()
    assert normalizer.normalize_value(
        "player_state",
        "before.time_seconds",
        900.0,
    ) == pytest.approx(0.5)


def test_model_input_contract_rejects_checkpoint_without_contract():
    with pytest.raises(ValueError, match="missing input_contract"):
        ModelInputContract.from_checkpoint({"data_spec": {}})


def test_model_input_contract_rejects_previous_state_semantics():
    payload = _build_contract().to_dict()
    payload["version"] = 1

    with pytest.raises(ValueError, match="unsupported input contract version"):
        ModelInputContract.from_dict(payload)
