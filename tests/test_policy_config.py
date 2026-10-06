"""策略配置清单的动作质量和独立优化器子配置加载测试。"""

from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from common.policy.config import ModelConfig, load_model_config, load_policy_config

_RESIDUAL_MIX_ENDPOINTS = (
    "residual_mix_r_start", "residual_mix_r_end",
    "residual_mix_a_start", "residual_mix_a_end",
)


def test_artzip_manifest_loads_action_quality_weights():
    root = Path(__file__).resolve().parents[1]
    config = load_policy_config(root / "config/models/black_mage/artzip/config.yaml")

    assert config["action_quality"] == {
        "severity_weights": {"minor": 0.25, "medium": 0.50, "major": 1.00},
    }
    assert "action_quality_config" not in config
    assert {"model", "training", "grpo"} <= config.keys()
    assert "optimizer_config" not in config
    assert config["optimizers"]["bc"]["name"] == "muon"
    assert config["optimizers"]["grpo"]["name"] == "adamw"


def test_action_quality_reference_resolves_relative_to_manifest(tmp_path, monkeypatch):
    config_dir = tmp_path / "model"
    config_dir.mkdir()
    manifest = config_dir / "config.yaml"
    manifest.write_text(
        "action_quality_config: action_quality.yaml\n",
        encoding="utf-8", newline="\n",
    )
    (config_dir / "action_quality.yaml").write_text(
        "action_quality:\n  severity_weights:\n    minor: 0.125\n",
        encoding="utf-8", newline="\n",
    )
    monkeypatch.chdir(tmp_path)

    assert load_policy_config(manifest) == {
        "action_quality": {"severity_weights": {"minor": 0.125}},
    }


def test_existing_config_without_action_quality_reference_remains_supported(tmp_path):
    manifest = tmp_path / "config.yaml"
    manifest.write_text("model_config: model.yaml\n", encoding="utf-8", newline="\n")
    (tmp_path / "model.yaml").write_text(
        "model:\n  d_model: 64\n", encoding="utf-8", newline="\n",
    )

    assert load_policy_config(manifest) == {"model": {"d_model": 64}}


def test_optimizer_reference_resolves_relative_to_manifest_and_merges_training(tmp_path, monkeypatch):
    config_dir = tmp_path / "model"
    config_dir.mkdir()
    manifest = config_dir / "config.yaml"
    manifest.write_text(
        "training_config: training.yaml\noptimizer_config: optimizer.yaml\n",
        encoding="utf-8", newline="\n",
    )
    (config_dir / "training.yaml").write_text(
        "training:\n  batch_size: 4\n", encoding="utf-8", newline="\n",
    )
    (config_dir / "optimizer.yaml").write_text(
        "optimizers:\n  bc:\n    name: muon\n    learning_rate: 0.0002\n"
        "  grpo:\n    name: adamw\n    learning_rate: 0.000003\n",
        encoding="utf-8", newline="\n",
    )
    monkeypatch.chdir(tmp_path)

    assert load_policy_config(manifest) == {
        "training": {"batch_size": 4},
        "optimizers": {
            "bc": {"name": "muon", "learning_rate": 0.0002},
            "grpo": {"name": "adamw", "learning_rate": 0.000003},
        },
    }


def test_optimizer_reference_missing_file_is_not_silently_ignored(tmp_path):
    manifest = tmp_path / "config.yaml"
    manifest.write_text("optimizer_config: missing.yaml\n", encoding="utf-8", newline="\n")

    with pytest.raises(FileNotFoundError, match="missing.yaml"):
        load_policy_config(manifest)


@pytest.mark.parametrize("reference", ["", "   ", False, 7, [], {}])
def test_optimizer_reference_rejects_invalid_paths(tmp_path, reference):
    manifest = tmp_path / "config.yaml"
    manifest.write_text(
        yaml.safe_dump({"optimizer_config": reference}), encoding="utf-8", newline="\n",
    )
    with pytest.raises(ValueError, match="optimizer_config must be a non-empty relative path"):
        load_policy_config(manifest)


def test_optimizer_reference_rejects_non_mapping_file(tmp_path):
    manifest = tmp_path / "config.yaml"
    manifest.write_text("optimizer_config: optimizer.yaml\n", encoding="utf-8", newline="\n")
    (tmp_path / "optimizer.yaml").write_text("- invalid\n", encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match="optimizer_config must be a mapping"):
        load_policy_config(manifest)


def test_qk_norm_scale_defaults_and_artzip_architecture():
    assert ModelConfig().qk_norm_scale == 1.2
    assert ModelConfig.from_mapping({}).qk_norm_scale == 1.2
    root = Path(__file__).resolve().parents[1]
    config = load_model_config(root / "config/models/black_mage/artzip/model.yaml")
    assert config.qk_norm_scale == 1.2
    assert (config.n_layers, config.d_model, config.n_heads, config.num_kv_heads, config.ff_dim) == (6, 384, 6, 1, 1536)


@pytest.mark.parametrize("value", [0.25, 1, 1.2, 2.5, "1.75"])
def test_qk_norm_scale_mapping_and_yaml_round_trip(value):
    config = ModelConfig.from_mapping({"qk_norm_scale": value})
    assert config.qk_norm_scale == float(value)
    assert isinstance(config.qk_norm_scale, float)
    restored = ModelConfig.from_mapping(yaml.safe_load(yaml.safe_dump(asdict(config))))
    assert restored == config


def test_qk_norm_scale_direct_constructor_normalizes_numeric_string():
    config = ModelConfig(qk_norm_scale="1.75")
    assert config.qk_norm_scale == 1.75
    assert isinstance(config.qk_norm_scale, float)
    assert config == ModelConfig.from_mapping({"qk_norm_scale": "1.75"})


@pytest.mark.parametrize("value", [True, False, 0, -1, float("nan"), float("inf"), -float("inf"),
                                  "nan", "inf", "-inf", "0", "-0.5", "invalid", None, [], {}])
def test_qk_norm_scale_rejects_invalid_mapping_values(value):
    with pytest.raises(ValueError, match="qk_norm_scale must be finite and positive"):
        ModelConfig.from_mapping({"qk_norm_scale": value})


@pytest.mark.parametrize("value", [True, False, 0, -1, float("nan"), float("inf"), -float("inf")])
def test_qk_norm_scale_direct_constructor_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="qk_norm_scale must be finite and positive"):
        ModelConfig(qk_norm_scale=value)


@pytest.mark.parametrize("field", _RESIDUAL_MIX_ENDPOINTS)
@pytest.mark.parametrize("value", [True, False])
def test_residual_mix_endpoints_reject_boolean_mapping_values(field, value):
    with pytest.raises(ValueError, match=f"model.{field}.*boolean"):
        ModelConfig.from_mapping({field: value})


@pytest.mark.parametrize("field", _RESIDUAL_MIX_ENDPOINTS)
@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize("full_attention_residuals", [False, True])
def test_residual_mix_endpoints_reject_boolean_yaml_values(
    tmp_path, field, value, full_attention_residuals,
):
    config_path = tmp_path / "model.yaml"
    config_path.write_text(
        yaml.safe_dump({"model": {
            field: value, "full_attention_residuals": full_attention_residuals,
        }}), encoding="utf-8", newline="\n",
    )
    with pytest.raises(ValueError, match=f"model.{field}.*boolean"):
        load_model_config(config_path)


@pytest.mark.parametrize("value", [0, -0.1, 1.15, "0", "-0.1"])
def test_residual_mix_endpoints_preserve_finite_numeric_values_and_round_trip(value):
    values = dict.fromkeys(_RESIDUAL_MIX_ENDPOINTS, value)
    config = ModelConfig.from_mapping(values)
    assert all(getattr(config, field) == float(value) for field in _RESIDUAL_MIX_ENDPOINTS)
    assert all(isinstance(getattr(config, field), float) for field in _RESIDUAL_MIX_ENDPOINTS)
    assert ModelConfig(**values) == config
    assert ModelConfig.from_mapping(yaml.safe_load(yaml.safe_dump(asdict(config)))) == config
