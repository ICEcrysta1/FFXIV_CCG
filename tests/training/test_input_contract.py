"""模型输入契约和归一化契约测试。"""

from __future__ import annotations

import pytest

from common.policy.data import ModelInputContract, Normalizer, SkillVocab
from common.policy.data.schema import SceneWindowSchema, TrainingSchema, TRAINING_SAMPLE_SCHEMA_VERSION
from common.policy.data.spec import DataSpec
from common.policy.data.input_contract import (
    INPUT_CONTRACT_VERSION, TOKEN_ENCODING_CONTRACT, RESIDUAL_MIX_CONFIG_FIELDS,
    residual_composition_contract,
)
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
        skill_vocab=SkillVocab.from_entries([(152, 1), (900001, 2), (0, 3)]),
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
    assert restored.create_skill_vocab().to_dict() == contract.create_skill_vocab().to_dict()
    assert restored.create_skill_vocab().require_lookup(900001, context="disabled") == 2
    assert restored.create_skill_vocab().require_lookup(0, context="wait") == 3

    normalizer = restored.create_normalizer()
    assert normalizer.normalize_value(
        "player_state",
        "previous_action_after.time_seconds",
        900.0,
    ) == pytest.approx(0.5)


def test_model_input_contract_rejects_checkpoint_without_contract():
    with pytest.raises(ValueError, match="missing input_contract"):
        ModelInputContract.from_checkpoint({"data_spec": {}})


def test_model_input_contract_describes_post_role_parameterless_rms():
    """保存精确编码位置与参数，避免将归一化放在 role 相加之前。"""
    payload = _build_contract().to_dict()
    assert payload["version"] == 21
    encoding = payload["token_encoding"]
    assert encoding["skill"] == "E[id] + Linear(skill_features)"
    assert encoding["state"] == "Linear(state_values) + Linear(null_mask, bias=False) + Linear(state_reset_mask, bias=False)"
    assert encoding["scene"] == "Linear_by_scene_type(scene_values)"
    assert encoding["token_normalization"] == {
        "type": "RMSNorm",
        "position": "after_content_plus_role",
        "eps": 1e-5,
        "elementwise_affine": False,
        "applications": 1,
    }


def test_model_input_contract_describes_parameterless_backbone_rms():
    """主干两处子层和最终输出固定使用同一无参数 RMS 规则。"""
    encoding = _build_contract().to_dict()["token_encoding"]
    assert encoding["backbone_normalization"] == {
        "type": "RMSNorm",
        "positions": [
            "encoder.layers[*].norm1",
            "encoder.layers[*].norm2",
            "encoder.norm",
        ],
        "eps": 1e-5,
        "elementwise_affine": False,
    }


def test_model_input_contract_describes_post_rope_per_head_qk_rms():
    """Q/K 共用保存的尺度，缓存仅归一化新 K，不固定 autocast 的张量 dtype。"""
    encoding = _build_contract().to_dict()["token_encoding"]
    assert encoding["attention_qk_normalization"] == {
        "type": "RMSNorm",
        "position": "after_RoPE_before_attention",
        "normalized_shape": "head_dim",
        "eps": None,
        "elementwise_affine": False,
        "scale": "model_config.qk_norm_scale",
        "applies_to": ["query", "key"],
        "key_cache": "normalize_new_keys_once_before_append",
    }


def test_model_input_contract_describes_independent_output_head_and_fp32_softcap():
    """动作头仅初始化时拷贝技能行，之后独立训练，softcap 先于宿主策略。"""
    encoding = _build_contract().to_dict()["token_encoding"]
    assert encoding["output_projection"] == {
        "type": "Linear",
        "parameter": "output_head.weight",
        "weight_shape": "[data_spec.num_actions, model_config.d_model]",
        "bias": False,
        "weight_tying": False,
        "initialization": "copy_skill_embed_rows_in_action_to_vocab_id_order",
    }
    assert encoding["logit_softcap"] == {
        "formula": "cap * tanh(output_head(hidden).float() / cap)",
        "cap": "model_config.logit_softcap",
        "compute_dtype": "float32",
        "position": "after_output_projection_before_repetition_penalty_and_legal_mask",
    }


def test_model_input_contract_rejects_previous_shared_output_version():
    payload = _build_contract().to_dict()
    payload["version"] = 19
    payload["token_encoding"]["output_projection"] = "hidden @ E[action_to_vocab_id].T"
    payload["token_encoding"].pop("logit_softcap")
    with pytest.raises(ValueError, match="unsupported input contract version"):
        ModelInputContract.from_checkpoint({"input_contract": payload})


@pytest.mark.parametrize("field,value", [
    ("type", "shared_embedding"),
    ("parameter", "input_encoder.skill_embed.weight"),
    ("weight_shape", "[vocab_size, model_config.d_model]"),
    ("bias", True),
    ("weight_tying", True),
    ("initialization", "normal_std_0.001"),
])
def test_model_input_contract_rejects_forged_output_head(field, value):
    payload = _build_contract().to_dict()
    payload["token_encoding"]["output_projection"][field] = value
    with pytest.raises(ValueError, match="token_encoding"):
        ModelInputContract.from_checkpoint({"input_contract": payload})


@pytest.mark.parametrize("field,value", [
    ("formula", "clamp(logits, -cap, cap)"),
    ("cap", 15.0),
    ("compute_dtype", "bfloat16"),
    ("position", "after_repetition_penalty_and_legal_mask"),
])
def test_model_input_contract_rejects_forged_logit_softcap(field, value):
    payload = _build_contract().to_dict()
    payload["token_encoding"]["logit_softcap"][field] = value
    with pytest.raises(ValueError, match="token_encoding"):
        ModelInputContract.from_checkpoint({"input_contract": payload})


def test_model_input_contract_rejects_previous_unnormalized_qk_version():
    payload = _build_contract().to_dict()
    payload["version"] = 18
    payload["token_encoding"].pop("attention_qk_normalization")
    with pytest.raises(ValueError, match="unsupported input contract version"):
        ModelInputContract.from_checkpoint({"input_contract": payload})


@pytest.mark.parametrize("field,value", [
    ("position", "before_RoPE"),
    ("normalized_shape", "d_model"),
    ("eps", 1e-5),
    ("elementwise_affine", True),
    ("scale", 1.2),
    ("applies_to", ["query"]),
    ("key_cache", "normalize_all_cached_keys"),
])
def test_model_input_contract_rejects_forged_qk_normalization(field, value):
    payload = _build_contract().to_dict()
    payload["token_encoding"]["attention_qk_normalization"][field] = value
    with pytest.raises(ValueError, match="token_encoding"):
        ModelInputContract.from_checkpoint({"input_contract": payload})


def test_model_input_contract_rejects_previous_backbone_layernorm_version():
    """版本 16 的主干 LayerNorm 权重不能通过新版输入字段静默复用。"""
    payload = _build_contract().to_dict()
    payload["version"] = 16
    payload["token_encoding"].pop("backbone_normalization")
    with pytest.raises(ValueError, match="unsupported input contract version"):
        ModelInputContract.from_dict(payload)


@pytest.mark.parametrize("legacy_backbone", ["missing", "layernorm", "gamma_rms"])
def test_model_input_contract_rejects_forged_parameterless_backbone_descriptor(legacy_backbone):
    """仅把旧模型的版本号改成当前版本，不能掩盖主干归一化算法或参数差异。"""
    payload = _build_contract().to_dict()
    backbone = payload["token_encoding"]["backbone_normalization"]
    if legacy_backbone == "missing":
        payload["token_encoding"].pop("backbone_normalization")
    elif legacy_backbone == "layernorm":
        backbone.update({"type": "LayerNorm", "elementwise_affine": True})
    else:
        backbone["elementwise_affine"] = True
    with pytest.raises(ValueError, match="token_encoding"):
        ModelInputContract.from_dict(payload)


def test_model_input_contract_rejects_previous_content_layernorm_version():
    """版本 15 即使使用新版描述，也不能静默恢复到新版架构。"""
    payload = _build_contract().to_dict()
    payload["version"] = 15
    with pytest.raises(ValueError, match="unsupported input contract version"):
        ModelInputContract.from_dict(payload)


def test_model_input_contract_rejects_legacy_content_layernorm_descriptor():
    """仅改版本号不能将三路 LayerNorm checkpoint 冒充统一 RMS 输入。"""
    payload = _build_contract().to_dict()
    payload["token_encoding"].update({
        "skill": "LayerNorm(E[id] + Linear(skill_features))",
        "state": "LayerNorm(Linear(state_values) + Linear(null_mask, bias=False))",
        "scene": "LayerNorm(Linear_by_scene_type(scene_values))",
    })
    payload["token_encoding"].pop("token_normalization")
    with pytest.raises(ValueError, match="token_encoding"):
        ModelInputContract.from_dict(payload)


def test_model_input_contract_rejects_previous_skill_first_contract():
    payload = _build_contract().to_dict()
    payload["version"] = 14
    payload["token_encoding"]["token_order"] = "scene, (skill_i, state_i)*H, current_state"

    with pytest.raises(ValueError, match="unsupported input contract version"):
        ModelInputContract.from_dict(payload)


@pytest.mark.parametrize("change", ["missing", "role", "token_order", "output_projection", "logit_softcap", "state_encoder", "state_snapshots", "history_state_frozen_at", "token_normalization", "backbone_normalization", "backbone_residual", "attention_qk_normalization"])
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
    elif change == "token_order":
        payload["token_encoding"][change] = "scene, (skill_i, state_i)*H, current_state"
    else:
        payload["token_encoding"][change] = "legacy_fused_tokens"
    with pytest.raises(ValueError, match="token_encoding"):
        ModelInputContract.from_dict(payload)


def test_serialized_token_descriptor_does_not_mutate_contract_authority():
    contract = _build_contract()
    payload = contract.to_dict()
    payload["token_encoding"]["role_ids"]["state"] = 99
    assert contract.to_dict()["token_encoding"] == TOKEN_ENCODING_CONTRACT


def test_model_input_contract_describes_both_residual_paths():
    """归一化后的初始 token 逐层注入，Full AttnRes 不叠加普通混合。"""
    residual = _build_contract().to_dict()["token_encoding"]["backbone_residual"]
    assert residual["selection"] == "model_config.full_attention_residuals"
    assert residual["ordinary"]["formula"] == "mixed = r[layer] * hidden + a[layer] * x0"
    assert residual["ordinary"]["attention_and_skip_input"] == "mixed"
    assert residual["ordinary"]["x0"] == "input_encoder.output_after_token_rmsnorm"
    assert residual["ordinary"]["x0_detached"] is False
    assert residual["ordinary"]["parameters"] == ["encoder.residual_mix.r", "encoder.residual_mix.a"]
    assert residual["full_attention"]["learned_residual_mix"] is False


def test_model_input_contract_rejects_previous_ordinary_residual_version():
    payload = _build_contract().to_dict()
    payload["version"] = 17
    payload["token_encoding"].pop("backbone_residual")
    with pytest.raises(ValueError, match="unsupported input contract version"):
        ModelInputContract.from_dict(payload)


@pytest.mark.parametrize("field", ["formula", "x0", "x0_detached", "attention_and_skip_input"])
def test_model_input_contract_rejects_forged_residual_mix(field):
    payload = _build_contract().to_dict()
    payload["token_encoding"]["backbone_residual"]["ordinary"][field] = "legacy"
    with pytest.raises(ValueError, match="token_encoding"):
        ModelInputContract.from_dict(payload)


def _residual_model_config():
    return {"full_attention_residuals": False, "residual_mix_r_start": 1.15,
            "residual_mix_r_end": 1.05, "residual_mix_a_start": .20, "residual_mix_a_end": .05}


@pytest.mark.parametrize("field", RESIDUAL_MIX_CONFIG_FIELDS)
def test_residual_composition_requires_saved_initialization_fields(field):
    config = _residual_model_config()
    config.pop(field)
    with pytest.raises(ValueError, match="missing residual mix initialization fields"):
        residual_composition_contract(config)


@pytest.mark.parametrize("value", [True, "1.15", float("nan"), float("inf")])
def test_residual_composition_rejects_invalid_saved_initialization(value):
    config = _residual_model_config()
    config["residual_mix_r_start"] = value
    with pytest.raises(ValueError, match="must be a finite number"):
        residual_composition_contract(config)


def test_residual_composition_selects_full_attention_without_mix():
    config = _residual_model_config()
    descriptor = residual_composition_contract(config)
    assert descriptor["initialization"] == {
        "method": "linspace", "r": {"start": 1.15, "end": 1.05},
        "a": {"start": .20, "end": .05},
    }
    config["full_attention_residuals"] = True
    assert residual_composition_contract(config) == TOKEN_ENCODING_CONTRACT["backbone_residual"]["full_attention"]


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


def test_model_input_contract_requires_complete_vocabulary():
    payload = _build_contract().to_dict()
    payload.pop("skill_vocab")
    with pytest.raises(ValueError, match="missing complete skill_vocab"):
        ModelInputContract.from_dict(payload)


def test_model_input_contract_rejects_embedding_row_count_drift():
    contract = _build_contract()
    contract.assert_matches_embedding(4)
    with pytest.raises(ValueError, match="embedding row count"):
        contract.assert_matches_embedding(3)


@pytest.mark.parametrize("entries", [
    [], [(1, 0)], [(1, 1), (2, 1)], [(1, 1), (1, 2)], [(1, 1), (2, 3)],
    [(True, 1)], [(1, True)], [("1", 1)], [(1, 1.0)], [(1,)],
])
def test_full_skill_vocab_rejects_ambiguous_or_incomplete_rows(entries):
    with pytest.raises(ValueError, match="skill vocab"):
        SkillVocab.from_entries(entries)


@pytest.mark.parametrize("field,value", [("size", 99), ("padding_vocab_id", 1), ("size", True)])
def test_full_skill_vocab_rejects_inconsistent_metadata(field, value):
    payload = _build_contract().create_skill_vocab().to_dict()
    payload[field] = value
    with pytest.raises(ValueError, match="skill vocab"):
        SkillVocab.from_dict(payload)


def test_full_skill_vocab_compares_mapping_instead_of_entry_order():
    vocab = SkillVocab.from_entries([(152, 1), (900001, 2), (0, 3)])
    vocab.assert_matches([(0, 3), (900001, 2), (152, 1)], context="test")
    with pytest.raises(ValueError, match="raw_skill_id=152.*expected vocab_id=1.*actual vocab_id=2"):
        vocab.assert_matches([(152, 2), (900001, 1), (0, 3)], context="test")
