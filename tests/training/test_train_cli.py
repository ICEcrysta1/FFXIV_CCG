"""训练 CLI 启动方式测试。"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import logging
import runpy
import subprocess
import sys
import types
from types import SimpleNamespace
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class StubRunConfig:
    """避免 CLI 单元测试为构造 replace 配置而加载 CUDA 训练栈。"""

    raw_data_dir: Path
    output_dir: Path
    job_tag: str | None
    model_variant: str | None = None


def _install_fake_training_modules(monkeypatch):
    """安装 CLI 测试用的轻量训练模块，隔离重量级训练依赖。"""
    fake_training = types.ModuleType("training")
    fake_training.__path__ = [str(PROJECT_ROOT / "training")]
    fake_config = types.ModuleType("training.config")
    fake_training_loop = types.ModuleType("training.loop")
    for name in (
        "load_run_config",
    ):
        setattr(fake_config, name, None)
    setattr(fake_training_loop, "run_training", None)

    monkeypatch.setitem(sys.modules, "training", fake_training)
    monkeypatch.setitem(sys.modules, "training.config", fake_config)
    monkeypatch.setitem(sys.modules, "training.loop", fake_training_loop)


def _load_train_cli(monkeypatch):
    """在假的 training.config/training.loop 上加载 CLI，隔离重量级训练依赖。"""
    _install_fake_training_modules(monkeypatch)

    module_name = "training.train"
    spec = importlib.util.spec_from_file_location(
        module_name,
        PROJECT_ROOT / "training" / "train.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_train_script_supports_direct_invocation():
    result = subprocess.run(
        [sys.executable, "training/train.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "训练职业行为克隆模型" in result.stdout


def test_train_script_main_guard_is_coverage_visible(monkeypatch, capsys):
    """直接执行脚本的路径注入和 main guard 也应纳入覆盖率统计。"""
    _install_fake_training_modules(monkeypatch)
    monkeypatch.setattr(
        sys,
        "path",
        [entry for entry in sys.path if Path(entry).resolve() != PROJECT_ROOT],
    )
    monkeypatch.setattr(sys, "argv", ["train.py", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(
            str(PROJECT_ROOT / "training" / "train.py"),
            run_name="__main__",
        )

    assert exc_info.value.code == 0
    assert "训练职业行为克隆模型" in capsys.readouterr().out


def test_train_main_forwards_cli_overrides(monkeypatch, caplog, tmp_path):
    """CLI 参数应覆盖职业配置，并先调用脚本层准备 cache。"""
    train_cli = _load_train_cli(monkeypatch)

    config_path = tmp_path / "job.yaml"
    raw_data_dir = tmp_path / "raw"
    output_dir = tmp_path / "checkpoints"
    loaded_config = StubRunConfig(
        raw_data_dir=Path("configured-data"),
        output_dir=Path("configured-checkpoints"),
        job_tag=None,
    )
    calls: dict[str, object] = {}

    def resolve_config(value):
        calls["config_arg"] = value
        return config_path

    def load_config(path):
        calls["loaded_config_path"] = path
        return loaded_config

    def resolve_job_tag(path):
        calls["job_config_path"] = path
        return "black_mage"

    def resolve_device(value):
        calls["device_arg"] = value
        return "cuda:0"

    def fake_run_training(config, **kwargs):
        calls["config"] = config
        calls["training_kwargs"] = kwargs
        return {
            "data_spec": SimpleNamespace(
                job_tag="black_mage",
                num_candidates=28,
                state_dim=149,
                scene_dim=7,
                skill_feature_dim=23,
            ),
            "output_dir": output_dir,
        }

    prepared_paths = [raw_data_dir / "prepared.json"]

    def fake_prepare_training_caches(config, max_files):
        calls["prepare_config"] = config
        calls["prepare_max_files"] = max_files
        return prepared_paths

    monkeypatch.setattr(train_cli, "resolve_policy_model_config_path", resolve_config)
    monkeypatch.setattr(train_cli, "load_run_config", load_config)
    monkeypatch.setattr(train_cli, "resolve_policy_model_job_tag", resolve_job_tag)
    monkeypatch.setattr(train_cli, "resolve_policy_model_variant", lambda _path: "artzip")
    monkeypatch.setattr(train_cli, "resolve_policy_device", resolve_device)
    monkeypatch.setattr(train_cli, "run_training", fake_run_training)
    monkeypatch.setattr(train_cli, "_prepare_training_caches", fake_prepare_training_caches)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
            "--raw-data-dir",
            str(raw_data_dir),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "3",
            "--batch-size",
            "64",
            "--lr",
            "0.0005",
            "--max-files",
            "7",
            "--device",
            "cuda",
            "--resume",
            str(tmp_path / "epoch_009.pt"),
        ],
    )

    with caplog.at_level(logging.INFO):
        train_cli.main()

    assert calls["config_arg"] == config_path
    assert calls["loaded_config_path"] == config_path
    assert calls["job_config_path"] == config_path
    assert calls["device_arg"] == "cuda"
    assert calls["config"].job_tag == "black_mage"
    assert calls["config"].raw_data_dir == raw_data_dir
    assert calls["prepare_config"].raw_data_dir == raw_data_dir
    assert calls["prepare_max_files"] == 7
    assert calls["training_kwargs"] == {
        "raw_paths": prepared_paths,
        "output_dir": output_dir,
        "max_epochs": 3,
        "batch_size": 64,
        "learning_rate": 0.0005,
        "device_name": "cuda:0",
        "resume_path": tmp_path / "epoch_009.pt",
    }
    assert any(
        "训练完成: job=black_mage model_variant=artzip" in record.getMessage()
        for record in caplog.records
    )


def test_train_main_uses_config_and_environment_defaults(monkeypatch):
    """未提供 CLI 覆盖时，应把 None 和解析出的默认设备交给训练层。"""
    train_cli = _load_train_cli(monkeypatch)

    config_path = Path("config/models/machinist/artzip/config.yaml")
    loaded_config = StubRunConfig(
        raw_data_dir=Path("configured-data"),
        output_dir=Path("configured-checkpoints"),
        job_tag=None,
    )
    calls: dict[str, object] = {}

    def resolve_config(value):
        calls["config_arg"] = value
        return config_path

    def resolve_job_tag(path):
        calls["job_config_path"] = path
        return "machinist"

    def resolve_device(value):
        calls["device_arg"] = value
        return "cuda"

    monkeypatch.setattr(train_cli, "resolve_policy_model_config_path", resolve_config)
    monkeypatch.setattr(train_cli, "load_run_config", lambda path: loaded_config)
    monkeypatch.setattr(train_cli, "resolve_policy_model_job_tag", resolve_job_tag)
    monkeypatch.setattr(train_cli, "resolve_policy_model_variant", lambda _path: "artzip")
    monkeypatch.setattr(train_cli, "resolve_policy_device", resolve_device)

    def fake_run_training(config, **kwargs):
        calls["config"] = config
        calls["training_kwargs"] = kwargs
        return {
            "data_spec": SimpleNamespace(
                job_tag="machinist",
                num_candidates=16,
                state_dim=100,
                scene_dim=4,
                skill_feature_dim=23,
            ),
            "output_dir": Path("output"),
        }

    monkeypatch.setattr(
        train_cli,
        "_prepare_training_caches",
        lambda config, max_files: [Path("prepared.json")],
    )
    monkeypatch.setattr(train_cli, "run_training", fake_run_training)
    monkeypatch.setattr(sys, "argv", ["train.py"])

    train_cli.main()

    assert calls["config_arg"] is None
    assert calls["job_config_path"] == config_path
    assert calls["device_arg"] is None
    assert calls["config"].job_tag == "machinist"
    assert calls["training_kwargs"] == {
        "raw_paths": [Path("prepared.json")],
        "output_dir": None,
        "max_epochs": None,
        "batch_size": None,
        "learning_rate": None,
        "device_name": "cuda",
        "resume_path": None,
    }
