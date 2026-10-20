"""BC 运行与优化器配置的默认值、严格校验和 GRPO 隔离测试。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from common.policy.config import ModelConfig
from common.policy.data.normalization import NormalizerConfig
from common.training.optimizer_config import OptimizerConfig
from grpo.config import GrpoConfig, load_grpo_config, load_grpo_run_config
from training.config import RunConfig, load_run_config


def _write_config(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    return path


def test_bc_optimizer_defaults_preserve_adamw(tmp_path):
    direct = RunConfig(raw_data_dir=tmp_path, output_dir=tmp_path, job_tag="black_mage")
    loaded = load_run_config(_write_config(tmp_path, {"training": {}}))

    assert direct.optimizer == loaded.optimizer == OptimizerConfig()
    assert loaded.optimizer.name == "adamw"
    assert loaded.learning_rate == 1e-4
    assert loaded.weight_decay == 0.01
    assert direct.fp32_gradients is loaded.fp32_gradients is False


def test_bc_nested_defaults_are_independent_and_use_current_normalizer(tmp_path, monkeypatch):
    load_normalizer = Mock(side_effect=[
        NormalizerConfig(time_scale=90.0), NormalizerConfig(time_scale=180.0),
    ])
    monkeypatch.setattr(
        "common.policy.data.normalization.load_normalizer_config", load_normalizer,
    )

    first = RunConfig(raw_data_dir=tmp_path, output_dir=tmp_path, job_tag="black_mage")
    second = RunConfig(raw_data_dir=tmp_path, output_dir=tmp_path, job_tag="black_mage")

    for name in ("runtime_debug", "repetition", "model"):
        assert getattr(first, name) is not getattr(second, name)
    assert first.runtime_debug == second.runtime_debug
    assert first.repetition == second.repetition
    assert first.model.time_delta_scale == 90.0
    assert second.model.time_delta_scale == 180.0
    assert load_normalizer.call_count == 2


def test_bc_explicit_model_bypasses_default_normalizer(tmp_path, monkeypatch):
    model = ModelConfig(time_delta_scale=123.0)
    load_normalizer = Mock(side_effect=AssertionError("显式模型不应读取默认归一化配置"))
    monkeypatch.setattr(
        "common.policy.data.normalization.load_normalizer_config", load_normalizer,
    )

    config = RunConfig(
        raw_data_dir=tmp_path, output_dir=tmp_path, job_tag="black_mage", model=model,
    )

    assert config.model is model
    load_normalizer.assert_not_called()


@pytest.mark.parametrize("model_raw,expected_scale,normalizer_reads", [
    ({}, 135.0, 1),
    ({"time_delta_scale": 123.0}, 123.0, 0),
])
def test_bc_loader_preserves_parsed_model_without_default_factory(
    tmp_path, monkeypatch, model_raw, expected_scale, normalizer_reads,
):
    load_normalizer = Mock(return_value=NormalizerConfig(time_scale=135.0))
    monkeypatch.setattr(
        "common.policy.data.normalization.load_normalizer_config", load_normalizer,
    )
    monkeypatch.setattr("training.config.load_root_dotenv", lambda *_: None)
    monkeypatch.setattr(
        "training.config.load_policy_config", lambda *args, **kwargs: {"model": model_raw},
    )

    loaded = load_run_config(tmp_path / "config.yaml")

    assert loaded.model.time_delta_scale == expected_scale
    assert load_normalizer.call_count == normalizer_reads


@pytest.mark.parametrize("enabled", [False, True])
def test_bc_fp32_gradients_preserves_forward_precision(tmp_path, enabled):
    loaded = load_run_config(
        _write_config(tmp_path, {"precision": "bf16", "training": {"fp32_gradients": enabled}})
    )

    assert loaded.fp32_gradients is enabled
    assert loaded.precision == "bf16"


@pytest.mark.parametrize("value", [None, 0, 1, 0.0, 1.0, "true", "false", [], {}])
def test_bc_fp32_gradients_rejects_non_boolean_yaml(tmp_path, value):
    path = _write_config(tmp_path, {"training": {"fp32_gradients": value}})
    with pytest.raises(TypeError, match="training.fp32_gradients must be a boolean"):
        load_run_config(path)


def test_bc_optimizer_reads_explicit_muon_parameters(tmp_path):
    optimizer = {
        "name": "muon",
        "momentum": 0.9,
        "nesterov": False,
        "ns_steps": 3,
        "adjust_lr_fn": "original",
    }
    loaded = load_run_config(
        _write_config(
            tmp_path,
            {"training": {"optimizer": optimizer, "learning_rate": 2e-4, "weight_decay": 0.02}},
        )
    )

    assert asdict(loaded.optimizer) == optimizer
    assert loaded.learning_rate == 2e-4
    assert loaded.weight_decay == 0.02


@pytest.mark.parametrize("name", ["adamw", "muon"])
def test_bc_optimizer_partial_mapping_uses_explicit_algorithm_defaults(tmp_path, name):
    loaded = load_run_config(
        _write_config(tmp_path, {"training": {"optimizer": {"name": name}}})
    )

    assert loaded.optimizer == OptimizerConfig(name=name)
    assert loaded.optimizer.adjust_lr_fn == "match_rms_adamw"


@pytest.mark.parametrize("raw", [None, False, 0, "muon", [], ["muon"]])
def test_bc_optimizer_rejects_non_mapping_sections(tmp_path, raw):
    path = _write_config(tmp_path, {"training": {"optimizer": raw}})
    with pytest.raises(TypeError, match="optimizer must be a mapping"):
        load_run_config(path)


def test_bc_optimizer_rejects_unknown_fields(tmp_path):
    path = _write_config(
        tmp_path,
        {"training": {"optimizer": {"name": "muon", "momentun": 0.9}}},
    )
    with pytest.raises(ValueError, match="unknown fields: momentun"):
        load_run_config(path)


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    [
        ("name", "sgd", ValueError),
        ("name", "Muon", ValueError),
        ("name", None, ValueError),
        ("name", True, ValueError),
        ("name", [], ValueError),
        ("momentum", -0.1, ValueError),
        ("momentum", 1.0, ValueError),
        ("momentum", float("nan"), ValueError),
        ("momentum", float("inf"), ValueError),
        ("momentum", float("-inf"), ValueError),
        ("momentum", "0.95", TypeError),
        ("momentum", True, TypeError),
        ("momentum", None, TypeError),
        ("nesterov", "false", TypeError),
        ("nesterov", 0, TypeError),
        ("nesterov", 1, TypeError),
        ("nesterov", None, TypeError),
        ("ns_steps", 0, ValueError),
        ("ns_steps", 100, ValueError),
        ("ns_steps", 5.0, TypeError),
        ("ns_steps", "5", TypeError),
        ("ns_steps", True, TypeError),
        ("ns_steps", float("nan"), TypeError),
        ("ns_steps", float("inf"), TypeError),
        ("adjust_lr_fn", "auto", ValueError),
        ("adjust_lr_fn", None, ValueError),
        ("adjust_lr_fn", True, ValueError),
        ("adjust_lr_fn", [], ValueError),
    ],
)
def test_bc_optimizer_rejects_invalid_parameters_in_yaml_and_direct_construction(
    tmp_path, field_name, value, error_type
):
    values = {"name": "muon", field_name: value}
    path = _write_config(tmp_path, {"training": {"optimizer": values}})

    with pytest.raises(error_type, match=f"optimizer.{field_name}"):
        load_run_config(path)
    with pytest.raises(error_type, match=f"optimizer.{field_name}"):
        OptimizerConfig(**values)


@pytest.mark.parametrize("momentum", [0, 0.999])
@pytest.mark.parametrize("ns_steps", [1, 99])
def test_bc_optimizer_accepts_boundary_values(momentum, ns_steps):
    config = OptimizerConfig(name="muon", momentum=momentum, ns_steps=ns_steps)
    assert config.momentum == momentum
    assert config.ns_steps == ns_steps


def test_bc_optimizer_does_not_change_grpo_configuration(tmp_path):
    payload = {
        "training": {"learning_rate": 2e-4, "weight_decay": 0.02},
        "grpo": {"learning_rate": 3e-6, "weight_decay": 0.0},
    }
    path = _write_config(tmp_path, payload)
    previous_grpo = load_grpo_config(path)
    previous_run = load_grpo_run_config(path)

    payload["training"]["optimizer"] = {"name": "muon"}
    payload["training"]["fp32_gradients"] = True
    _write_config(tmp_path, payload)

    assert load_run_config(path).optimizer.name == "muon"
    assert load_run_config(path).fp32_gradients is True
    assert load_grpo_config(path) == previous_grpo
    assert load_grpo_run_config(path) == previous_run
    assert previous_grpo.learning_rate == 3e-6
    assert previous_grpo.weight_decay == 0.0


@pytest.mark.parametrize("bc_name,grpo_name", [("muon", "adamw"), ("adamw", "muon")])
def test_central_optimizer_stages_choose_algorithms_independently(tmp_path, bc_name, grpo_name):
    path = _write_config(
        tmp_path,
        {
            "training": {"batch_size": 8},
            "grpo": {"group_size": 4},
            "optimizers": {
                "bc": {
                    "name": bc_name,
                    "learning_rate": 2e-4,
                    "weight_decay": 0.02,
                    "warmup_steps": 7,
                    "momentum": 0.9,
                    "nesterov": False,
                    "ns_steps": 3,
                    "adjust_lr_fn": "original",
                },
                "grpo": {
                    "name": grpo_name,
                    "learning_rate": 3e-6,
                    "weight_decay": 0.03,
                    "warmup_steps": 2,
                    "momentum": 0.8,
                    "nesterov": True,
                    "ns_steps": 6,
                    "adjust_lr_fn": "match_rms_adamw",
                },
            },
        },
    )
    bc = load_run_config(path)
    grpo = load_grpo_config(path)

    assert bc.optimizer == OptimizerConfig(
        name=bc_name, momentum=0.9, nesterov=False, ns_steps=3, adjust_lr_fn="original"
    )
    assert grpo.optimizer == OptimizerConfig(name=grpo_name, momentum=0.8, ns_steps=6)
    assert (bc.learning_rate, bc.weight_decay, bc.warmup_steps) == (2e-4, 0.02, 7)
    assert (grpo.learning_rate, grpo.weight_decay, grpo.warmup_steps) == (3e-6, 0.03, 2)
    assert bc.batch_size == 8
    assert grpo.group_size == 4


def test_legacy_grpo_optimizer_and_nested_grpo_remain_supported(tmp_path):
    values = {
        "optimizer": {"name": "muon", "momentum": 0.8},
        "learning_rate": 3e-6,
        "weight_decay": 0.02,
        "warmup_steps": 6,
    }
    loaded = load_grpo_config(_write_config(tmp_path, {"training": {"grpo": values}}))

    assert loaded == GrpoConfig.from_mapping(values)
    assert loaded.optimizer == OptimizerConfig(name="muon", momentum=0.8)
    assert (loaded.learning_rate, loaded.weight_decay, loaded.warmup_steps) == (3e-6, 0.02, 6)


def test_stage_defaults_and_missing_central_stage_preserve_legacy_behavior(tmp_path):
    path = _write_config(tmp_path, {"optimizers": {"bc": {"name": "muon"}}})
    grpo = load_grpo_config(path)
    assert grpo.optimizer == OptimizerConfig()
    assert (grpo.learning_rate, grpo.weight_decay, grpo.warmup_steps) == (1e-6, 0.0, 20)
    path = _write_config(
        tmp_path,
        {"optimizers": {"grpo": {"name": "muon"}}, "training": {"learning_rate": 0.003}},
    )
    bc = load_run_config(path)
    assert bc.optimizer == OptimizerConfig()
    assert (bc.learning_rate, bc.weight_decay, bc.warmup_steps) == (0.003, 0.01, 500)


@pytest.mark.parametrize("stage,legacy_section", [("bc", "training"), ("grpo", "grpo")])
@pytest.mark.parametrize("field", ["optimizer", "learning_rate", "weight_decay", "warmup_steps"])
def test_central_optimizer_rejects_duplicate_legacy_authority(tmp_path, stage, legacy_section, field):
    duplicate = {"optimizer": {"name": "muon"}, "learning_rate": 0.01, "weight_decay": 0.0, "warmup_steps": 0}
    path = _write_config(
        tmp_path,
        {"optimizers": {stage: {"name": "muon"}}, legacy_section: {field: duplicate[field]}},
    )
    loader = load_run_config if stage == "bc" else load_grpo_config
    with pytest.raises(ValueError, match=f"optimizers.{stage} conflicts with {legacy_section}"):
        loader(path)


@pytest.mark.parametrize("stage", ["bc", "grpo"])
@pytest.mark.parametrize(
    "values,field,error_type",
    [
        ({"name": "sgd"}, "name", ValueError),
        ({"momentum": float("nan")}, "momentum", ValueError),
        ({"ns_steps": True}, "ns_steps", TypeError),
        ({"learning_rate": float("nan")}, "learning_rate", ValueError),
        ({"learning_rate": float("inf")}, "learning_rate", ValueError),
        ({"learning_rate": 0.0}, "learning_rate", ValueError),
        ({"learning_rate": True}, "learning_rate", TypeError),
        ({"weight_decay": float("inf")}, "weight_decay", ValueError),
        ({"weight_decay": -0.01}, "weight_decay", ValueError),
        ({"weight_decay": "0.01"}, "weight_decay", TypeError),
        ({"warmup_steps": -1}, "warmup_steps", ValueError),
        ({"warmup_steps": 1.5}, "warmup_steps", ValueError),
        ({"warmup_steps": True}, "warmup_steps", ValueError),
        ({"momentun": 0.9}, "unknown fields", ValueError),
    ],
)
def test_central_optimizer_rejects_invalid_values(tmp_path, stage, values, field, error_type):
    path = _write_config(tmp_path, {"optimizers": {stage: values}})
    loader = load_run_config if stage == "bc" else load_grpo_config
    with pytest.raises(error_type, match=field):
        loader(path)


@pytest.mark.parametrize(
    "optimizers,error_type",
    [(None, TypeError), ([], TypeError), ("muon", TypeError),
     ({"bcs": {}}, ValueError), ({"bc": None}, ValueError)],
)
def test_central_optimizer_rejects_invalid_structure(tmp_path, optimizers, error_type):
    path = _write_config(tmp_path, {"optimizers": optimizers})
    with pytest.raises(error_type, match="optimizers"):
        load_run_config(path)


def test_artzip_optimizer_settings_are_centralized():
    root = Path(__file__).resolve().parents[2]
    model_dir = root / "config/models/black_mage/artzip"
    raw_training = yaml.safe_load((model_dir / "training.yaml").read_text(encoding="utf-8"))["training"]
    raw_grpo = yaml.safe_load((model_dir / "grpo.yaml").read_text(encoding="utf-8"))["grpo"]
    fields = {"optimizer", "learning_rate", "weight_decay", "warmup_steps"}
    assert not fields.intersection(raw_training)
    assert not fields.intersection(raw_grpo)

    bc = load_run_config(model_dir / "config.yaml")
    grpo = load_grpo_config(model_dir / "config.yaml")
    assert bc.optimizer == OptimizerConfig(name="muon")
    assert grpo.optimizer == OptimizerConfig(name="adamw")
    assert (bc.learning_rate, bc.weight_decay, bc.warmup_steps) == (1e-4, 0.125, 500)
    assert (grpo.learning_rate, grpo.weight_decay, grpo.warmup_steps) == (1e-6, 0.0, 20)
