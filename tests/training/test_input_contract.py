"""模型输入契约和归一化契约测试。"""

from __future__ import annotations

import pytest

from common.policy.data import ModelInputContract, Normalizer
from common.policy.data.schema import SceneWindowSchema, TrainingSchema, TRAINING_SAMPLE_SCHEMA_VERSION
from common.policy.data.spec import DataSpec
from common.policy.data.input_contract import INPUT_CONTRACT_VERSION, TOKEN_ENCODING_CONTRACT
from common.output_context_schema import CANONICAL_CONTEXT_SCHEMA_VERSION


def _build_contract() -> ModelInputContract:
    data_spec = DataSpec(
        job_tag="black_mage",
        num_actions=1,
        state_dim=1,
        scene_dim=3,
        skill_feature_dim=1,
        num_scene_types=1,
        action_keys=("fire_iii",),
        skill_feature_names=("potency",),
        action_to_vocab_id=(1,),
        action_is_gcd=(True,),
    )
    schema = TrainingSchema(
        serialization_format="test",
        sample_schema_version=TRAINING_SAMPLE_SCHEMA_VERSION,
        context_schema_version=CANONICAL_CONTEXT_SCHEMA_VERSION,
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
        state_group_feature_keys={"player_state": ("previous_action_after.time_seconds",)},
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
        "previous_action_after.time_seconds",
        900.0,
    ) == pytest.approx(0.5)


def test_model_input_contract_rejects_checkpoint_without_contract():
    with pytest.raises(ValueError, match="missing input_contract"):
        ModelInputContract.from_checkpoint({"data_spec": {}})


def test_model_input_contract_rejects_previous_skill_time_contract():
    payload = _build_contract().to_dict()
    payload["version"] = INPUT_CONTRACT_VERSION - 1

    with pytest.raises(ValueError, match="unsupported input contract version"):
        ModelInputContract.from_dict(payload)


@pytest.mark.parametrize("change", ["missing", "role", "token_order", "output_projection", "state_encoder", "state_snapshots", "history_state_frozen_at"])
def test_model_input_contract_requires_exact_independent_token_descriptor(change):
    payload = _build_contract().to_dict()
    assert payload["version"] == INPUT_CONTRACT_VERSION
    assert payload["token_encoding"] == TOKEN_ENCODING_CONTRACT
    if change == "missing":
        payload.pop("token_encoding")
    elif change == "role":
        payload["token_encoding"]["role_ids"]["state"] = 2
    elif change == "state_encoder":
        payload["token_encoding"]["current_state_encoder"] = "separate_current_state_encoder"
    else:
        payload["token_encoding"][change] = "legacy_fused_tokens"
    with pytest.raises(ValueError, match="token_encoding"):
        ModelInputContract.from_dict(payload)


def test_serialized_token_descriptor_does_not_mutate_contract_authority():
    contract = _build_contract()
    payload = contract.to_dict()
    payload["token_encoding"]["role_ids"]["state"] = 99
    assert contract.to_dict()["token_encoding"] == TOKEN_ENCODING_CONTRACT


@pytest.mark.parametrize("location", ["features", "fields"])
def test_current_input_contract_rejects_removed_skill_time(location):
    """仅更新版本号不能把带旧技能时间的契约变成新版输入。"""
    payload = _build_contract().to_dict()
    if location == "features":
        payload["data_spec"]["skill_feature_names"] = ("potency", "time_seconds")
        payload["data_spec"]["skill_feature_dim"] = 2
    else:
        payload["schema"]["skill_history_fields"] = ("skill_key", "time_seconds")
    with pytest.raises(ValueError, match="removed skill time_seconds"):
        ModelInputContract.from_dict(payload)


def test_data_spec_rejects_removed_column_hidden_by_feature_names():
    payload = _build_contract().to_dict()["data_spec"]
    payload["skill_feature_dim"] += 1
    with pytest.raises(ValueError, match="skill feature order length"):
        DataSpec.from_dict(payload)
