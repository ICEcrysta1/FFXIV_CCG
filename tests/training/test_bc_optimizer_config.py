"""BC 优化器配置的兼容默认、严格校验和 GRPO 隔离测试。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

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
    with pytest.raises(ValueError, match="optimizer must be a mapping"):
        load_run_config(path)


def test_bc_optimizer_rejects_unknown_fields(tmp_path):
    path = _write_config(
        tmp_path,
        {"training": {"optimizer": {"name": "muon", "momentun": 0.9}}},
    )
    with pytest.raises(ValueError, match="unknown fields: momentun"):
        load_run_config(path)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("name", "sgd"),
        ("name", "Muon"),
        ("name", None),
        ("name", True),
        ("name", []),
        ("momentum", -0.1),
        ("momentum", 1.0),
        ("momentum", float("nan")),
        ("momentum", float("inf")),
        ("momentum", float("-inf")),
        ("momentum", "0.95"),
        ("momentum", True),
        ("momentum", None),
        ("nesterov", "false"),
        ("nesterov", 0),
        ("nesterov", 1),
        ("nesterov", None),
        ("ns_steps", 0),
        ("ns_steps", 100),
        ("ns_steps", 5.0),
        ("ns_steps", "5"),
        ("ns_steps", True),
        ("ns_steps", float("nan")),
        ("ns_steps", float("inf")),
        ("adjust_lr_fn", "auto"),
        ("adjust_lr_fn", None),
        ("adjust_lr_fn", True),
        ("adjust_lr_fn", []),
    ],
)
def test_bc_optimizer_rejects_invalid_parameters_in_yaml_and_direct_construction(
    tmp_path, field_name, value
):
    values = {"name": "muon", field_name: value}
    path = _write_config(tmp_path, {"training": {"optimizer": values}})

    with pytest.raises(ValueError, match=f"optimizer.{field_name}"):
        load_run_config(path)
    with pytest.raises(ValueError, match=f"optimizer.{field_name}"):
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
    _write_config(tmp_path, payload)

    assert load_run_config(path).optimizer.name == "muon"
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
    "values,field",
    [
        ({"name": "sgd"}, "name"),
        ({"momentum": float("nan")}, "momentum"),
        ({"ns_steps": True}, "ns_steps"),
        ({"learning_rate": float("nan")}, "learning_rate"),
        ({"learning_rate": float("inf")}, "learning_rate"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"learning_rate": True}, "learning_rate"),
        ({"weight_decay": float("inf")}, "weight_decay"),
        ({"weight_decay": -0.01}, "weight_decay"),
        ({"weight_decay": "0.01"}, "weight_decay"),
        ({"warmup_steps": -1}, "warmup_steps"),
        ({"warmup_steps": 1.5}, "warmup_steps"),
        ({"warmup_steps": True}, "warmup_steps"),
        ({"momentun": 0.9}, "unknown fields"),
    ],
)
def test_central_optimizer_rejects_invalid_values(tmp_path, stage, values, field):
    path = _write_config(tmp_path, {"optimizers": {stage: values}})
    loader = load_run_config if stage == "bc" else load_grpo_config
    with pytest.raises(ValueError, match=field):
        loader(path)


@pytest.mark.parametrize("optimizers", [None, [], "muon", {"bcs": {}}, {"bc": None}])
def test_central_optimizer_rejects_invalid_structure(tmp_path, optimizers):
    path = _write_config(tmp_path, {"optimizers": optimizers})
    with pytest.raises(ValueError, match="optimizers"):
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
    assert (bc.learning_rate, bc.weight_decay, bc.warmup_steps) == (1e-4, 0.01, 500)
    assert (grpo.learning_rate, grpo.weight_decay, grpo.warmup_steps) == (1e-6, 0.0, 20)
