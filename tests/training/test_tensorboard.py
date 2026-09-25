"""TensorBoard 训练指标配置与共享写入工具测试。"""

import json
import os
from pathlib import Path

import pytest

from common.training import tensorboard, tensorboard_server
from grpo.config import load_grpo_config
from training.config import load_run_config


def test_tensorboard_config_rejects_invalid_values():
    with pytest.raises(TypeError, match="enabled must be a boolean"):
        tensorboard.TensorBoardConfig.from_mapping({"enabled": "yes"})
    with pytest.raises(ValueError, match="log_every_steps must be >= 1"):
        tensorboard.TensorBoardConfig.from_mapping({"log_every_steps": 0})


def test_bc_and_grpo_yaml_tensorboard_settings():
    manifest = Path("config/models/black_mage/artzip/config.yaml")

    bc = load_run_config(manifest).tensorboard
    grpo = load_grpo_config(manifest).tensorboard

    assert bc == grpo
    assert bc.enabled is True
    assert bc.log_every_steps == 50
    assert bc.flush_secs == 30


def test_grpo_tensorboard_does_not_inherit_bc_switch(monkeypatch, tmp_path):
    from grpo import config as grpo_config

    raw = {
        "training": {"tensorboard": {"enabled": True}},
        "grpo": {},
    }
    monkeypatch.setattr(grpo_config, "load_policy_config", lambda *_args, **_kwargs: raw)
    manifest = tmp_path / "config.yaml"

    assert load_grpo_config(manifest).tensorboard.enabled is False
    raw["grpo"]["tensorboard"] = {"enabled": True}
    raw["training"]["tensorboard"]["enabled"] = False
    assert load_grpo_config(manifest).tensorboard.enabled is True


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
            / "artzip_bc",
            tensorboard=SimpleNamespace(enabled=True),
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

    started = []
    opened = []
    monkeypatch.setattr(
        tensorboard_server,
        "_start_background",
        lambda command, *, port, log_dir: started.append((command, port, log_dir)),
    )
    monkeypatch.setattr(tensorboard_server.webbrowser, "open", opened.append)
    monkeypatch.setattr(
        sys,
        "argv",
        ["tensorboard_server", "--background", "--open-browser", "--if-enabled"],
    )
    assert tensorboard_server.main() == 0
    assert started == [(command, 6017, expected_root)]
    assert opened == ["http://127.0.0.1:6017"]

    monkeypatch.setattr(
        tensorboard_server,
        "load_run_config",
        lambda _path: SimpleNamespace(tensorboard=SimpleNamespace(enabled=False)),
    )
    assert tensorboard_server.main() == 0
    assert len(started) == 1

    monkeypatch.setattr(
        tensorboard_server,
        "load_grpo_run_config",
        lambda _path: SimpleNamespace(output_dir=expected_root.parent / "artzip_bc"),
    )
    monkeypatch.setattr(
        tensorboard_server,
        "load_grpo_config",
        lambda _path: SimpleNamespace(tensorboard=SimpleNamespace(enabled=False)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tensorboard_server",
            "--background",
            "--open-browser",
            "--if-enabled",
            "--scope",
            "grpo",
        ],
    )
    assert tensorboard_server.main() == 0
    assert len(started) == 1

    monkeypatch.setattr(
        tensorboard_server,
        "load_grpo_config",
        lambda _path: SimpleNamespace(tensorboard=SimpleNamespace(enabled=True)),
    )
    assert tensorboard_server.main() == 0
    assert len(started) == 2
    assert opened[-1] == "http://127.0.0.1:6017"


def test_background_server_waits_for_readiness(monkeypatch, tmp_path):
    monkeypatch.setattr(tensorboard_server, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        tensorboard_server, "_check_existing_service", lambda _port, _log_dir: False
    )
    checks = iter((False, True))
    monkeypatch.setattr(tensorboard_server, "_is_ready", lambda _port: next(checks))
    monkeypatch.setattr(tensorboard_server.time, "sleep", lambda _seconds: None)
    calls = []
    records = []
    monkeypatch.setattr(
        tensorboard_server,
        "_write_record",
        lambda port, log_dir, pid: records.append((port, log_dir, pid)),
    )

    class FakeProcess:
        pid = 42

        def poll(self):
            return None

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return FakeProcess()

    monkeypatch.setattr(tensorboard_server.subprocess, "Popen", fake_popen)
    tensorboard_server._start_background(
        ["python", "-m", "tensorboard.main"], port=6017, log_dir=tmp_path
    )
    assert len(calls) == 1
    assert calls[0][1]["cwd"] == tmp_path
    assert records == [(6017, tmp_path, 42)]


def test_existing_server_must_match_event_directory(monkeypatch, tmp_path):
    record = tmp_path / "server.json"
    record.write_text(
        json.dumps({"pid": os.getpid(), "log_dir": "C:/old-model"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(tensorboard_server, "_record_path", lambda _port: record)
    monkeypatch.setattr(tensorboard_server, "_is_ready", lambda _port: True)

    with pytest.raises(RuntimeError, match="当前模型需要"):
        tensorboard_server._check_existing_service(6017, tmp_path / "new-model")

    tensorboard_server._write_record(6017, tmp_path / "new-model", os.getpid())
    assert tensorboard_server._check_existing_service(6017, tmp_path / "new-model")

    record.write_text('{"pid": 0, "log_dir": "C:/old-model"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="无法确认"):
        tensorboard_server._check_existing_service(6017, tmp_path / "new-model")

    record.unlink()
    with pytest.raises(RuntimeError, match="无法确认"):
        tensorboard_server._check_existing_service(6017, tmp_path / "new-model")


def test_ready_check_uses_tensorboard_plugins_route(monkeypatch):
    requested = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def fake_urlopen(url, *, timeout):
        requested.append((url, timeout))
        return Response()

    monkeypatch.setattr(tensorboard_server.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(tensorboard_server.json, "load", lambda _response: {})
    assert tensorboard_server._is_ready(6017)
    assert requested == [("http://127.0.0.1:6017/data/plugins_listing", 0.5)]


def test_background_server_reports_early_exit(monkeypatch, tmp_path):
    monkeypatch.setattr(tensorboard_server, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        tensorboard_server, "_check_existing_service", lambda _port, _log_dir: False
    )
    monkeypatch.setattr(tensorboard_server, "_is_ready", lambda _port: False)

    class FakeProcess:
        def poll(self):
            return 1

    def fake_popen(_command, **kwargs):
        kwargs["stdout"].write("port already in use\n")
        kwargs["stdout"].flush()
        return FakeProcess()

    monkeypatch.setattr(tensorboard_server.subprocess, "Popen", fake_popen)
    with pytest.raises(RuntimeError, match="port already in use"):
        tensorboard_server._start_background(["python"], port=6017, log_dir=tmp_path)


@pytest.mark.parametrize("exit_kind", ("owner", "spawner"))
def test_owned_worker_closes_tensorboard_when_parent_exits(
    monkeypatch, tmp_path, exit_kind
):
    record = tmp_path / "server.json"
    stop_file = tmp_path / "server.stop"
    monkeypatch.setattr(tensorboard_server, "_record_path", lambda _port: record)
    monkeypatch.setattr(tensorboard_server, "_stop_path", lambda _port, _pid: stop_file)
    monkeypatch.setattr(tensorboard_server, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(tensorboard_server.os, "getpid", lambda: 42)
    alive = {
        11: [True, False] if exit_kind == "owner" else [True, True],
        12: [False] if exit_kind == "spawner" else [True],
    }
    monkeypatch.setattr(
        tensorboard_server, "_pid_is_alive", lambda pid: alive[pid].pop(0)
    )
    monkeypatch.setattr(tensorboard_server.time, "sleep", lambda _seconds: None)

    class FakeProcess:
        returncode = None
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, *, timeout):
            assert timeout == 5
            return 0

    child = FakeProcess()
    monkeypatch.setattr(tensorboard_server.subprocess, "Popen", lambda *_args, **_kwargs: child)
    assert tensorboard_server._run_owned_worker(
        ["python"], port=6017, log_dir=tmp_path, owner_pid=11, spawner_pid=12
    ) == 0
    assert child.terminated
    assert not record.exists()


def test_stop_owned_only_signals_matching_owner(monkeypatch, tmp_path):
    record = tmp_path / "server.json"
    stop_file = tmp_path / "server.stop"
    monkeypatch.setattr(tensorboard_server, "_record_path", lambda _port: record)
    monkeypatch.setattr(tensorboard_server, "_stop_path", lambda _port, _pid: stop_file)
    alive = iter((True, False, False))
    monkeypatch.setattr(tensorboard_server, "_pid_is_alive", lambda _pid: next(alive))
    monkeypatch.setattr(tensorboard_server.time, "sleep", lambda _seconds: None)
    record.write_text(json.dumps({"pid": 42, "owner_pid": 11}), encoding="utf-8")

    tensorboard_server._stop_owned(6017, 12)
    assert not stop_file.exists()
    tensorboard_server._stop_owned(6017, 11)
    assert stop_file.exists()


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
