"""TensorBoard 训练指标配置与共享写入工具测试。"""

from pathlib import Path

import pytest

from common.training import tensorboard
from common.training import tensorboard_server
from grpo.config import load_grpo_config
from training.config import load_run_config


def test_tensorboard_config_rejects_invalid_values():
    with pytest.raises(TypeError, match="enabled must be a boolean"):
        tensorboard.TensorBoardConfig.from_mapping({"enabled": "yes"})
    with pytest.raises(ValueError, match="log_every_steps must be >= 1"):
        tensorboard.TensorBoardConfig.from_mapping({"log_every_steps": 0})


def test_bc_and_grpo_share_yaml_tensorboard_settings():
    manifest = Path("config/models/black_mage/artzip/config.yaml")

    bc = load_run_config(manifest).tensorboard
    grpo = load_grpo_config(manifest).tensorboard

    assert bc == grpo
    assert bc.enabled is True
    assert bc.log_every_steps == 50
    assert bc.flush_secs == 30


def test_disabled_writer_does_not_import_optional_dependency(monkeypatch, tmp_path):
    monkeypatch.setattr(
        tensorboard,
        "_resolve_summary_writer",
        lambda: pytest.fail("disabled TensorBoard must not import SummaryWriter"),
    )

    assert tensorboard.create_tensorboard_writer(
        tensorboard.TensorBoardConfig(),
        tmp_path,
        run_name="bc",
    ) is None


def test_writer_uses_distinct_run_directories(monkeypatch, tmp_path):
    created = []

    class FakeSummaryWriter:
        def __init__(self, *, log_dir, flush_secs):
            self.log_dir = log_dir
            self.flush_secs = flush_secs
            self.closed = False
            Path(log_dir).mkdir(parents=True)
            created.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(tensorboard, "_resolve_summary_writer", lambda: FakeSummaryWriter)
    config = tensorboard.TensorBoardConfig(enabled=True, flush_secs=17)

    output_dir = tmp_path / "artifacts" / "checkpoints" / "black_mage" / "artzip_bc"
    first = tensorboard.create_tensorboard_writer(
        config,
        output_dir,
        run_name="bc",
        model_variant="artzip",
    )
    second = tensorboard.create_tensorboard_writer(
        config,
        output_dir,
        run_name="grpo",
        model_variant="artzip",
    )

    assert first is created[0]
    assert second is created[1]
    assert first.log_dir != second.log_dir
    expected_root = tmp_path / "artifacts" / "checkpoints" / "black_mage" / "artzip_tensorboard"
    assert Path(first.log_dir).parent == expected_root
    assert Path(second.log_dir).parent == expected_root
    assert first.flush_secs == 17
    tensorboard.close_tensorboard_writer(first)
    tensorboard.close_tensorboard_writer(second)
    assert first.closed and second.closed


def test_server_uses_dotenv_selected_model_tensorboard_root(
    monkeypatch,
    tmp_path,
):
    import sys
    from types import SimpleNamespace

    from common.policy import config as policy_config

    model_root = tmp_path / "config" / "models"
    model_config = model_root / "black_mage" / "artzip" / "config.yaml"
    model_config.parent.mkdir(parents=True)
    model_config.write_text("model_config: model.yaml\n", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "FFXIV_JOB_TAG=black_mage\n"
        "FFXIV_MODEL_VARIANT=artzip\n"
        "TENSORBOARD_PORT=6017\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("FFXIV_JOB_TAG", raising=False)
    monkeypatch.delenv("FFXIV_MODEL_VARIANT", raising=False)
    monkeypatch.delenv("TENSORBOARD_PORT", raising=False)
    monkeypatch.setattr(policy_config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(policy_config, "POLICY_MODEL_ROOT", model_root)
    monkeypatch.setattr(tensorboard_server, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        tensorboard_server,
        "load_run_config",
        lambda _path: SimpleNamespace(
            output_dir=tmp_path
            / "artifacts"
            / "checkpoints"
            / "black_mage"
            / "artzip_bc"
        ),
    )
    monkeypatch.setattr(sys, "argv", ["tensorboard_server"])
    calls = {}

    def fake_call(command, *, cwd):
        calls["command"] = command
        calls["cwd"] = cwd
        return 0

    monkeypatch.setattr(tensorboard_server.subprocess, "call", fake_call)

    assert tensorboard_server.main() == 0

    command = calls["command"]
    expected_root = (
        tmp_path
        / "artifacts"
        / "checkpoints"
        / "black_mage"
        / "artzip_tensorboard"
    )
    assert Path(command[4]) == expected_root
    assert command[-1] == "6017"
    assert calls["cwd"] == tmp_path


def test_enabled_writer_reports_missing_tensorboard_dependency(monkeypatch, tmp_path):
    def missing_dependency():
        raise ImportError("No module named tensorboard")

    monkeypatch.setattr(tensorboard, "_resolve_summary_writer", missing_dependency)

    with pytest.raises(RuntimeError, match="dependency is missing"):
        tensorboard.create_tensorboard_writer(
            tensorboard.TensorBoardConfig(enabled=True),
            tmp_path,
            run_name="bc",
            model_variant="artzip",
        )


def test_write_scalar_metrics_skips_non_finite_and_non_numeric_values():
    class FakeWriter:
        def __init__(self):
            self.values = []

        def add_scalar(self, tag, value, step):
            self.values.append((tag, value, step))

    writer = FakeWriter()
    tensorboard.write_scalar_metrics(
        writer,
        {"loss": 0.25, "bad": float("nan"), "label": "text"},
        prefix="validation",
        global_step=120,
    )

    assert writer.values == [("validation/loss", 0.25, 120)]
