"""策略配置清单的动作质量和独立优化器子配置加载测试。"""

from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from common.policy.config import ModelConfig, load_model_config, load_policy_config

_REMOVED_RESIDUAL_MIX_FIELDS = (
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
    assert config["optimizers"]["grpo"]["name"] == "muon"


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

    with pytest.raises(TypeError, match="optimizer_config must be a mapping"):
        load_policy_config(manifest)


def test_query_rank_defaults_and_artzip_architecture():
    assert ModelConfig().q_lora_rank == 64
    assert ModelConfig.from_mapping({}).q_lora_rank == 64
    assert ModelConfig(d_model=16).q_lora_rank == 4
    assert ModelConfig.from_mapping({"d_model": 16}).q_lora_rank == 4
    assert ModelConfig(d_model=2).q_lora_rank == 1
    root = Path(__file__).resolve().parents[1]
    config = load_model_config(root / "config/models/black_mage/artzip/model.yaml")
    assert config.q_lora_rank == 96
    assert config.logit_softcap == 15.0
    assert (config.swiglu_limit, config.hc_mult, config.hc_sinkhorn_iters, config.hc_eps) == (
        10.0, 4, 20, 1e-6,
    )
    assert (config.n_layers, config.d_model, config.n_heads, config.num_kv_heads, config.ff_dim) == (6, 384, 6, 1, 1536)


@pytest.mark.parametrize("value", [1, 8, 64, 256])
def test_query_rank_mapping_and_yaml_round_trip(value):
    config = ModelConfig.from_mapping({"q_lora_rank": value})
    assert config.q_lora_rank == value
    assert ModelConfig(q_lora_rank=value) == config
    restored = ModelConfig.from_mapping(yaml.safe_load(yaml.safe_dump(asdict(config))))
    assert restored == config


@pytest.mark.parametrize("value", [True, False, 0, -1, 257, 1.5, "8", float("nan"),
                                  float("inf"), [], {}])
def test_query_rank_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="model.q_lora_rank"):
        ModelConfig.from_mapping({"q_lora_rank": value})
    with pytest.raises(ValueError, match="model.q_lora_rank"):
        ModelConfig(q_lora_rank=value)


def test_query_rank_rejects_explicit_null_and_removed_fixed_scale():
    with pytest.raises(ValueError, match="model.q_lora_rank"):
        ModelConfig.from_mapping({"q_lora_rank": None})
    with pytest.raises(ValueError, match="qk_norm_scale"):
        ModelConfig.from_mapping({"qk_norm_scale": 1.2})
    with pytest.raises(TypeError, match="qk_norm_scale"):
        ModelConfig(qk_norm_scale=1.2)


@pytest.mark.parametrize("field", _REMOVED_RESIDUAL_MIX_FIELDS)
@pytest.mark.parametrize("value", [True, 1.15])
def test_removed_residual_mix_fields_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        ModelConfig.from_mapping({field: value})
    with pytest.raises(TypeError, match=field):
        ModelConfig(**{field: value})


@pytest.mark.parametrize("full_attention_residuals", [False, True])
def test_removed_residual_mix_fields_are_rejected_from_yaml(
    tmp_path, full_attention_residuals,
):
    config_path = tmp_path / "model.yaml"
    config_path.write_text(
        yaml.safe_dump({"model": {
            "residual_mix_r_start": True, "full_attention_residuals": full_attention_residuals,
        }}), encoding="utf-8", newline="\n",
    )
    with pytest.raises(ValueError, match="residual_mix_r_start"):
        load_model_config(config_path)


def test_hyper_connection_and_swiglu_config_round_trip():
    values = {"hc_mult": 4, "hc_sinkhorn_iters": 12, "hc_eps": 1e-5, "swiglu_limit": 8.0}
    config = ModelConfig.from_mapping(values)
    assert all(getattr(config, field) == value for field, value in values.items())
    assert ModelConfig(**values) == config
    assert ModelConfig.from_mapping(yaml.safe_load(yaml.safe_dump(asdict(config)))) == config


@pytest.mark.parametrize("value", [1, 15.0, 30.0, "7.5", 1e-30, 1e38])
@pytest.mark.parametrize("field", ["logit_softcap"])
def test_positive_fp32_model_parameters_normalize_and_round_trip(value, field):
    config = ModelConfig.from_mapping({field: value})
    assert getattr(config, field) == float(value)
    assert isinstance(getattr(config, field), float)
    assert ModelConfig(**{field: value}) == config
    assert ModelConfig.from_mapping(yaml.safe_load(yaml.safe_dump(asdict(config)))) == config


@pytest.mark.parametrize("value", [True, False, 0, -1, float("nan"), float("inf"),
                                  -float("inf"), "invalid", None, [], {},
                                  1e-46, 1e-300, 1e39, 1e300])
@pytest.mark.parametrize("field", ["logit_softcap"])
def test_positive_fp32_model_parameters_reject_invalid_values(value, field):
    message = f"model.{field} must be finite and positive within the FP32 normal range"
    with pytest.raises(ValueError) as mapping_error:
        ModelConfig.from_mapping({field: value})
    assert str(mapping_error.value) == message
    with pytest.raises(ValueError) as direct_error:
        ModelConfig(**{field: value})
    assert str(direct_error.value) == message
