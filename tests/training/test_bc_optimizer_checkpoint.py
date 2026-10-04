"""BC 优化器切换与 checkpoint 连续续训的契约边界。"""

from __future__ import annotations

from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from common.policy.config import ModelConfig
from common.policy.data import DataSpec, ModelInputContract, Normalizer, SkillVocab
from common.policy.data.schema import TrainingSchema
from common.training.optimizer_config import OptimizerConfig
from training.config import RunConfig
from training.loop.checkpoint import _validate_resume_checkpoint


@pytest.fixture
def resume_context(tmp_path):
    """构造真实输入契约，让测试经过正式续训校验入口。"""
    schema = TrainingSchema(
        serialization_format="test",
        sample_schema_version=1,
        context_schema_version=1,
        scene_context_mode="absolute",
        scene_windows=(),
        state_group_feature_keys={"player_state": ("a", "b", "c")},
        skill_history_fields=(),
    )
    normalizer = Normalizer()
    normalizer.configure_job_resources("black_mage")
    dataset = SimpleNamespace(
        job_tag="black_mage",
        num_actions=2,
        state_dim=3,
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


@pytest.mark.parametrize(
    "changed",
    [
        {"momentum": 0.9},
        {"nesterov": False},
        {"ns_steps": 6},
        {"adjust_lr_fn": "original"},
    ],
)
def test_muon_checkpoint_rejects_changed_hyperparameters(resume_context, changed):
    optimizer = OptimizerConfig(name="muon")
    resume_context.checkpoint["run_config"]["optimizer"] = asdict(optimizer)

    with pytest.raises(ValueError, match="optimizer config mismatch.*新训练"):
        _validate(resume_context, optimizer=replace(optimizer, **changed), force=True)


@pytest.mark.parametrize("missing_field", ["momentum", "nesterov", "ns_steps", "adjust_lr_fn"])
def test_muon_checkpoint_rejects_missing_hyperparameters(resume_context, missing_field):
    optimizer = OptimizerConfig(name="muon")
    saved_optimizer = asdict(optimizer)
    saved_optimizer.pop(missing_field)
    resume_context.checkpoint["run_config"]["optimizer"] = saved_optimizer

    with pytest.raises(ValueError, match=f"optimizer config mismatch: {missing_field}"):
        _validate(resume_context, optimizer=optimizer)


def test_adamw_resume_ignores_unused_muon_settings(resume_context):
    optimizer = OptimizerConfig(
        name="adamw", momentum=0.8, nesterov=False, ns_steps=3, adjust_lr_fn="original"
    )

    assert _validate(resume_context, optimizer=optimizer) == 1


@pytest.mark.parametrize("saved_optimizer", [None, [], {}, {"name": "unknown"}])
def test_optimizer_metadata_must_explicitly_describe_supported_optimizer(
    resume_context, saved_optimizer
):
    resume_context.checkpoint["run_config"]["optimizer"] = saved_optimizer

    with pytest.raises(ValueError, match="run_config.optimizer"):
        _validate(resume_context, force=True)
