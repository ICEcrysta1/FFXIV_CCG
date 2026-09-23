"""TensorBoard 训练指标配置与共享写入工具测试。"""

from pathlib import Path

import pytest

from common.training import tensorboard
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

    first = tensorboard.create_tensorboard_writer(config, tmp_path, run_name="bc")
    second = tensorboard.create_tensorboard_writer(config, tmp_path, run_name="bc")

    assert first is created[0]
    assert second is created[1]
    assert first.log_dir != second.log_dir
    assert Path(first.log_dir).parent == tmp_path / "tensorboard"
    assert first.flush_secs == 17
    tensorboard.close_tensorboard_writer(first)
    tensorboard.close_tensorboard_writer(second)
    assert first.closed and second.closed


def test_enabled_writer_reports_missing_tensorboard_dependency(monkeypatch, tmp_path):
    def missing_dependency():
        raise ImportError("No module named tensorboard")

    monkeypatch.setattr(tensorboard, "_resolve_summary_writer", missing_dependency)

    with pytest.raises(RuntimeError, match="dependency is missing"):
        tensorboard.create_tensorboard_writer(
            tensorboard.TensorBoardConfig(enabled=True),
            tmp_path,
            run_name="bc",
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
