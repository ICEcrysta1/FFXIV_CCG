"""BC 优化器切换与 checkpoint 连续续训的契约边界。"""

from __future__ import annotations

from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from common.policy.config import ModelConfig
from common.policy.data import DataSpec, ModelInputContract, Normalizer, SkillVocab
from common.policy.data.schema import TrainingSchema
from common.pytorch.serialization import safe_torch_load
from common.training.optimizer_config import MuonConfig, OptimizerConfig, SinkhornConfig
from tests.training._causal_fixtures import make_state_groups
from training.config import RunConfig
from training.loop.checkpoint import _save_checkpoint, _validate_resume_checkpoint


@pytest.fixture
def resume_context(tmp_path):
    """构造真实输入契约，让测试经过正式续训校验入口。"""
    schema = TrainingSchema(
        serialization_format="test",
        sample_schema_version=1,
        context_schema_version=1,
        scene_context_mode="absolute",
        scene_windows=(),
        state_groups=make_state_groups({"player_state": ("request_state.time_seconds", "b", "c")}, ("a", "b")), state_snapshots=("previous_action_after", "request_state"),
        skill_history_fields=(),
    )
    normalizer = Normalizer()
    normalizer.configure_job_resources("black_mage")
    normalizer = Normalizer.from_contract({
        **normalizer.normalization_contract, "skill_charge_limits": {"a": 2, "b": 2},
    })
    dataset = SimpleNamespace(
        job_tag="black_mage",
        num_actions=2,
        base_state_dim=3, state_dim=schema.state_layout(("a", "b")).state_dim,
        scene_dim=0,
        num_scene_types=0,
        action_keys=("a", "b"),
        skill_feature_names=("potency",),
        action_to_vocab_id=(1, 2),
        action_is_gcd=(True, True),
        schema=schema,
        normalizer=normalizer,
    )
    data_spec = DataSpec.from_dataset(dataset)
    config = RunConfig(
        raw_data_dir=tmp_path / "raw",
        output_dir=tmp_path / "output",
        job_tag="black_mage",
        model_variant="artzip",
        model=ModelConfig(d_model=8, n_layers=1, n_heads=2, ff_dim=16),
    )
    input_contract = ModelInputContract.from_training(
        skill_vocab=SkillVocab.from_entries([(1001, 1), (1002, 2), (900001, 3), (900002, 4)]),
        data_spec=data_spec,
        schema=schema,
        normalizer=normalizer,
    )
    checkpoint = {
        "epoch": 1,
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "model_config": asdict(config.model),
        "data_spec": asdict(data_spec),
        "model_variant": config.model_variant,
        "input_contract": input_contract.to_dict(),
        "run_config": asdict(config),
    }
    return SimpleNamespace(
        config=config,
        dataset=dataset,
        data_spec=data_spec,
        input_contract=input_contract,
        checkpoint=checkpoint,
    )


def _validate(context, *, optimizer=None, force=False):
    config = context.config
    if optimizer is not None:
        config = replace(config, optimizer=optimizer)
    return _validate_resume_checkpoint(
        context.checkpoint,
        data_spec=context.data_spec,
        dataset=context.dataset,
        config=config,
        input_contract=context.input_contract,
        force_resume_data_mismatch=force,
    )


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("legacy_encoding", ["version_15", "content_layernorm"])
def test_resume_rejects_legacy_input_encoding_even_when_forcing_data_mismatch(
    resume_context, force, legacy_encoding
):
    """数据规模豁免不能绕过输入归一化架构契约。"""
    payload = resume_context.checkpoint["input_contract"]
    if legacy_encoding == "version_15":
        payload["version"] = 15
        message = "unsupported input contract version"
    else:
        payload["token_encoding"].update({
            "skill": "LayerNorm(E[id] + Linear(skill_features))",
            "state": "LayerNorm(Linear(state_values) + Linear(null_mask, bias=False))",
            "scene": "LayerNorm(Linear_by_scene_type(scene_values))",
        })
        payload["token_encoding"].pop("token_normalization")
        message = "token_encoding"
    # 同时制造数据规模差异；即使请求豁免，也必须先拒绝旧输入架构。
    resume_context.checkpoint["run_config"]["max_files"] = 10
    with pytest.raises(ValueError, match=message):
        _validate(resume_context, force=force)


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("legacy_backbone", ["version_16", "missing", "layernorm", "no_gamma_rms"])
def test_resume_rejects_legacy_backbone_even_when_forcing_data_mismatch(
    resume_context, force, legacy_backbone
):
    """输入字段相同时，数据规模豁免仍不能跳过主干归一化契约。"""
    payload = resume_context.checkpoint["input_contract"]
    if legacy_backbone == "version_16":
        payload["version"] = 16
        payload["token_encoding"].pop("backbone_normalization")
        message = "unsupported input contract version"
    elif legacy_backbone == "missing":
        payload["token_encoding"].pop("backbone_normalization")
        message = "token_encoding"
    else:
        backbone = payload["token_encoding"]["backbone_normalization"]
        backbone["elementwise_affine"] = legacy_backbone == "layernorm"
        if legacy_backbone == "layernorm":
            backbone["type"] = "LayerNorm"
        message = "token_encoding"
    resume_context.checkpoint["run_config"]["max_files"] = 10
    with pytest.raises(ValueError, match=message):
        _validate(resume_context, force=force)


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("incompatible", [
    "version_26", "missing_rank", "rank_mismatch", "legacy_rope",
    "missing_head_dim", "head_dim_mismatch", "missing_rope_head_dim", "rope_head_dim_mismatch",
])
def test_resume_cannot_bypass_qk_normalization_contract(resume_context, force, incompatible):
    """数据规模豁免不能绕过 Q/KV、头宽或部分旋转的架构契约。"""
    if incompatible == "version_26":
        resume_context.checkpoint["input_contract"]["version"] = 26
        message = "unsupported input contract version"
    elif incompatible == "missing_rank":
        resume_context.checkpoint["model_config"].pop("q_lora_rank")
        message = "missing model.q_lora_rank"
    elif incompatible == "rank_mismatch":
        resume_context.checkpoint["model_config"]["q_lora_rank"] = 1
        message = "model config mismatch"
    elif incompatible == "legacy_rope":
        resume_context.checkpoint["input_contract"]["token_encoding"].pop("rotary_position_encoding")
        message = "token_encoding"
    else:
        field = "rope_head_dim" if "rope_head_dim" in incompatible else "head_dim"
        if incompatible.startswith("missing_"):
            resume_context.checkpoint["model_config"].pop(field)
            message = f"missing model.{field}"
        else:
            resume_context.checkpoint["model_config"][field] = 2 if field == "rope_head_dim" else 6
            message = "model config mismatch"
    resume_context.checkpoint["run_config"]["max_files"] = 10
    with pytest.raises(ValueError, match=message):
        _validate(resume_context, force=force)


def test_resume_preserves_matching_nondefault_query_rank(resume_context):
    """合法续训沿用 checkpoint 的低秩、每头及旋转宽度。"""
    model_config = replace(resume_context.config.model, q_lora_rank=7, head_dim=12, rope_head_dim=4)
    resume_context.config = replace(resume_context.config, model=model_config)
    resume_context.checkpoint["model_config"] = asdict(model_config)
    assert _validate(resume_context) == 1


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("incompatible", ["version_19", "shared_projection", "missing_cap", "cap_mismatch"])
def test_resume_cannot_bypass_independent_head_and_softcap_contract(resume_context, force, incompatible):
    """数据规模豁免不能把旧共享头或不同 softcap 配置当作同一训练恢复。"""
    if incompatible == "version_19":
        resume_context.checkpoint["input_contract"]["version"] = 19
        message = "unsupported input contract version"
    elif incompatible == "shared_projection":
        resume_context.checkpoint["input_contract"]["token_encoding"]["output_projection"] = (
            "hidden @ E[action_to_vocab_id].T"
        )
        message = "token_encoding"
    elif incompatible == "missing_cap":
        resume_context.checkpoint["model_config"].pop("logit_softcap")
        message = "missing model.logit_softcap"
    else:
        resume_context.checkpoint["model_config"]["logit_softcap"] = 20.0
        message = "model config mismatch"
    resume_context.checkpoint["run_config"]["max_files"] = 10
    with pytest.raises(ValueError, match=message):
        _validate(resume_context, force=force)


def test_resume_preserves_matching_nondefault_logit_softcap(resume_context):
    """合法续训使用保存的 softcap，不回落到本机 YAML 默认值。"""
    model_config = replace(resume_context.config.model, logit_softcap=9.0)
    resume_context.config = replace(resume_context.config, model=model_config)
    resume_context.checkpoint["model_config"] = asdict(model_config)
    assert _validate(resume_context) == 1


@pytest.mark.parametrize("force", [False, True])
def test_resume_rejects_inactive_vocab_row_drift_even_when_outputs_match(resume_context, force):
    resume_context.input_contract = replace(
        resume_context.input_contract,
        skill_vocab_entries=((1001, 1), (1002, 2), (900001, 4), (900002, 3)),
    )
    with pytest.raises(ValueError, match="resume checkpoint skill vocab mismatch.*raw_skill_id=900001"):
        _validate(resume_context, force=force)


@pytest.mark.parametrize("missing_run_config", [False, True])
def test_legacy_adamw_checkpoint_remains_resumable(resume_context, missing_run_config):
    if missing_run_config:
        resume_context.checkpoint.pop("run_config")
    else:
        resume_context.checkpoint["run_config"].pop("optimizer")

    assert _validate(resume_context, force=missing_run_config) == 1


@pytest.mark.parametrize("force", [False, True])
def test_legacy_adamw_checkpoint_cannot_resume_as_muon(resume_context, force):
    resume_context.checkpoint["run_config"].pop("optimizer")

    with pytest.raises(ValueError, match="optimizer mismatch.*新训练"):
        _validate(resume_context, optimizer=OptimizerConfig(name="muon"), force=force)


@pytest.mark.parametrize("saved_name,current_name", [("adamw", "muon"), ("muon", "adamw")])
@pytest.mark.parametrize("force", [False, True])
def test_optimizer_switch_cannot_be_forced_as_data_mismatch(
    resume_context, saved_name, current_name, force
):
    saved_config = resume_context.checkpoint["run_config"]
    saved_config["optimizer"] = asdict(OptimizerConfig(name=saved_name))
    # 同时存在数据差异时，强制续训也不能绕过优化器契约。
    saved_config["max_files"] = 10
    with pytest.raises(ValueError, match="optimizer mismatch.*新训练"):
        _validate(resume_context, optimizer=OptimizerConfig(name=current_name), force=force)


def test_muon_checkpoint_accepts_identical_optimizer_config(resume_context):
    optimizer = OptimizerConfig(name="muon")
    resume_context.checkpoint["run_config"]["optimizer"] = asdict(optimizer)

    assert _validate(resume_context, optimizer=optimizer) == 1


@pytest.mark.parametrize("name", ["muon", "adamw"])
@pytest.mark.parametrize(
    "group,changed",
    [
        ("muon", {"momentum": 0.9}),
        ("muon", {"nesterov": False}),
        ("muon", {"ns_steps": 6}),
        ("muon", {"adjust_lr_fn": "original"}),
        ("muon", {"eps": 1e-6}),
        ("muon", {"ns_coefficients": (2.0, -1.5, 0.5)}),
        ("adamw", {"betas": (0.8, 0.95)}),
        ("adamw", {"eps": 1e-6}),
        ("adamw", {"amsgrad": True}),
        ("sinkhorn", {"momentum": 0.9}),
        ("sinkhorn", {"nesterov": False}),
        ("sinkhorn", {"iters": 3}),
        ("sinkhorn", {"threshold": 1e-4}),
        ("sinkhorn", {"eps": 1e-12}),
        ("sinkhorn", {"lr_scale": 0.18}),
    ],
)
def test_optimizer_checkpoint_checks_active_hyperparameters(resume_context, name, group, changed):
    optimizer = OptimizerConfig(name=name)
    resume_context.checkpoint["run_config"]["optimizer"] = asdict(optimizer)
    changed_optimizer = replace(optimizer, **{
        group: replace(getattr(optimizer, group), **changed),
    })
    if name == "adamw" and group != "adamw":
        assert _validate(resume_context, optimizer=changed_optimizer, force=True) == 1
        return

    with pytest.raises(ValueError, match="optimizer config mismatch.*新训练"):
        _validate(resume_context, optimizer=changed_optimizer, force=True)


@pytest.mark.parametrize("group", ["muon", "adamw", "sinkhorn"])
def test_muon_checkpoint_restores_omitted_group_defaults(resume_context, group):
    optimizer = OptimizerConfig(name="muon")
    saved_optimizer = asdict(optimizer)
    saved_optimizer.pop(group)
    resume_context.checkpoint["run_config"]["optimizer"] = saved_optimizer

    assert _validate(resume_context, optimizer=optimizer) == 1


def test_adamw_resume_ignores_unused_muon_settings(resume_context):
    optimizer = OptimizerConfig(
        name="adamw", muon=MuonConfig(momentum=0.8, nesterov=False, ns_steps=3, adjust_lr_fn="original"),
        sinkhorn=SinkhornConfig(momentum=0.8, nesterov=False, iters=3),
    )

    assert _validate(resume_context, optimizer=optimizer) == 1


@pytest.mark.parametrize(
    ("saved_optimizer", "error_type"),
    [(None, TypeError), ([], TypeError), ({}, ValueError), ({"name": "unknown"}, ValueError)],
)
def test_optimizer_metadata_must_explicitly_describe_supported_optimizer(
    resume_context, saved_optimizer, error_type
):
    resume_context.checkpoint["run_config"]["optimizer"] = saved_optimizer

    with pytest.raises(error_type, match="run_config.optimizer"):
        _validate(resume_context, force=True)


@pytest.mark.parametrize("saved,current", [(False, True), (True, False)])
@pytest.mark.parametrize("force", [False, True])
def test_gradient_precision_switch_requires_new_training(resume_context, saved, current, force):
    """数据规模豁免不能绕过参数与梯度精度的训练模式切换。"""
    resume_context.checkpoint["run_config"]["fp32_gradients"] = saved
    resume_context.checkpoint["run_config"]["max_files"] = 10
    resume_context.config = replace(resume_context.config, fp32_gradients=current)

    with pytest.raises(ValueError, match="fp32_gradients mismatch.*新训练"):
        _validate(resume_context, force=force)


@pytest.mark.parametrize("enabled", [False, True])
def test_matching_gradient_precision_remains_resumable(resume_context, enabled):
    resume_context.checkpoint["run_config"]["fp32_gradients"] = enabled
    resume_context.config = replace(resume_context.config, fp32_gradients=enabled)

    assert _validate(resume_context) == 1


@pytest.mark.parametrize("current", [False, True])
def test_missing_gradient_precision_flag_preserves_legacy_mode(resume_context, current):
    resume_context.checkpoint["run_config"].pop("fp32_gradients")
    resume_context.config = replace(resume_context.config, fp32_gradients=current)

    if current:
        with pytest.raises(ValueError, match="fp32_gradients mismatch.*新训练"):
            _validate(resume_context, force=True)
    else:
        assert _validate(resume_context) == 1


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", [], {}])
def test_resume_rejects_non_boolean_gradient_precision(resume_context, value):
    resume_context.checkpoint["run_config"]["fp32_gradients"] = value

    with pytest.raises(TypeError, match="run_config.fp32_gradients must be a boolean"):
        _validate(resume_context, force=True)


@pytest.mark.parametrize("value", [None, 0, "invalid", []])
def test_resume_rejects_non_mapping_run_configuration(resume_context, value):
    resume_context.checkpoint["run_config"] = value

    with pytest.raises(TypeError, match="run_config must be a mapping"):
        _validate(resume_context, force=True)


@pytest.mark.parametrize("precision", ["float32", "bf16", "float16"])
@pytest.mark.parametrize("enabled", [False, True])
def test_checkpoint_records_forward_and_parameter_precision(
    resume_context, tmp_path, monkeypatch, precision, enabled,
):
    """保存的前向精度、参数精度与梯度模式分别描述实际训练契约。"""
    config = replace(resume_context.config, precision=precision, fp32_gradients=enabled)
    parameter_precision = "float32" if enabled else precision
    dtype = {"float32": torch.float32, "bf16": torch.bfloat16, "float16": torch.float16}[
        parameter_precision
    ]
    model = torch.nn.Linear(1, 1, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters())
    monkeypatch.setattr("training.loop.checkpoint._capture_rng_state", dict)
    path = tmp_path / "checkpoint.pt"

    _save_checkpoint(
        path, model, optimizer, 1, config, resume_context.data_spec, {},
        input_contract=resume_context.input_contract,
    )
    saved = safe_torch_load(path)

    assert saved["training_precision"] == precision
    assert saved["training_parameter_precision"] == parameter_precision
    assert saved["run_config"]["fp32_gradients"] is enabled
    assert saved["run_config"]["raw_data_dir"] == str(config.raw_data_dir)
    assert saved["run_config"]["output_dir"] == str(config.output_dir)
    assert saved["model_state_dict"]["weight"].dtype == dtype
