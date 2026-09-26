"""策略配置清单的动作质量子配置加载测试。"""

from pathlib import Path

from common.policy.config import load_policy_config


def test_artzip_manifest_loads_action_quality_weights():
    root = Path(__file__).resolve().parents[1]
    config = load_policy_config(root / "config/models/black_mage/artzip/config.yaml")

    assert config["action_quality"] == {
        "severity_weights": {"minor": 0.25, "medium": 0.50, "major": 1.00},
    }
    assert "action_quality_config" not in config
    assert {"model", "training", "grpo"} <= config.keys()


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
