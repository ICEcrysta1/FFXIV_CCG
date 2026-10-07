"""模块入口、命令分发和拆分后的根目录环境变量定位测试。"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.fflogs_scraper import cli
from scripts.fflogs_scraper.config import environment


def test_module_help_runs_without_credentials():
    process_env = dict(os.environ)
    process_env.pop("FFLOGS_V2_CLIENT_ID", None)
    process_env.pop("FFLOGS_V2_CLIENT_SECRET", None)
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", "-m", "scripts.fflogs_scraper", "--help"],
        cwd=Path(__file__).resolve().parents[3],
        env=process_env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "{single,batch,encounters}" in result.stdout
    assert "FFLogs 战斗数据拉取" in result.stdout


def test_dotenv_resolves_project_root_after_move_and_preserves_environment(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / ".env").write_text(
        "FFLOGS_V2_CLIENT_ID=root-id\nFFLOGS_V2_CLIENT_SECRET=root-secret\n",
        encoding="utf-8",
        newline="\n",
    )
    (tmp_path / ".env").write_text(
        "FFLOGS_V2_CLIENT_ID=wrong-id\n", encoding="utf-8", newline="\n",
    )
    monkeypatch.setattr(
        environment, "__file__",
        str(project_root / "scripts/fflogs_scraper/config/environment.py"),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FFLOGS_V2_CLIENT_ID", raising=False)
    monkeypatch.setenv("FFLOGS_V2_CLIENT_SECRET", "existing-secret")
    environment._load_dotenv()
    assert os.environ["FFLOGS_V2_CLIENT_ID"] == "root-id"
    assert os.environ["FFLOGS_V2_CLIENT_SECRET"] == "existing-secret"


@pytest.mark.parametrize("arguments, command, expected", [
    (["single", "https://www.fflogs.com/reports/ABC123?fight=last&source=1"],
     "single", {"url": "https://www.fflogs.com/reports/ABC123?fight=last&source=1"}),
    (["single", "--report", "ABC123", "--fight", "33", "--source", "1"],
     "single", {"report": "ABC123", "fight": 33, "source": 1}),
    (["batch", "-e", "1079", "--mode", "events-only", "--output", "raw/FRU"],
     "batch", {"encounter": 1079, "mode": "events-only", "metric": "rdps", "count": 200, "partition": None}),
    (["batch", "-e", "1079", "--count", "203", "--partition", "25", "--output", "raw/FRU"],
     "batch", {"count": 203, "partition": 25, "output": "raw/FRU"}),
    (["batch", "FRU", "--count", "200"],
     "batch", {"target": "FRU", "count": 200, "output": None}),
    (["batch", "65", "--count", "200"],
     "batch", {"target": "65", "count": 200, "output": None}),
    (["encounters", "-z", "39"], "encounters", {"zone": 39}),
])
def test_cli_dispatch_preserves_arguments(monkeypatch, arguments, command, expected):
    monkeypatch.setattr(sys, "argv", ["fflogs_scraper", *arguments])
    monkeypatch.setattr(cli, "_load_dotenv", lambda: None)
    monkeypatch.setenv("FFLOGS_V2_CLIENT_ID", "test-id")
    monkeypatch.setenv("FFLOGS_V2_CLIENT_SECRET", "test-secret")
    closed = []
    credentials = []
    clients = []
    def make_client(*args, **kwargs):
        credentials.append((args, kwargs))
        client = SimpleNamespace(
            server_region=kwargs.get("server_region"), cancel=lambda: None,
            close=lambda: closed.append(True),
        )
        clients.append(client)
        return client
    monkeypatch.setattr(cli, "FFLogsV2Client", make_client)
    monkeypatch.setattr(cli.signal, "signal", lambda *args: None)
    calls = []
    for name in ("single", "encounters"):
        monkeypatch.setattr(
            cli, f"_cmd_{name}",
            lambda actual_client, args, name=name: calls.append((name, actual_client, args)),
        )
    monkeypatch.setattr(
        cli, "_cmd_batch",
        lambda training_client, validation_client, args: calls.append(
            ("batch", training_client, args, validation_client),
        ),
    )
    cli.main()
    expected_regions = ("CN", "NA") if command == "batch" else (None,)
    assert credentials == [
        (("test-id", "test-secret"), {"server_region": region} if region else {})
        for region in expected_regions
    ]
    assert len(calls) == 1
    actual_command, actual_client, args = calls[0][:3]
    assert actual_command == command
    assert actual_client is clients[0]
    if command == "batch":
        assert calls[0][3] is clients[1]
    assert closed == [True] * len(clients)
    assert all(getattr(args, key) == value for key, value in expected.items())


def test_cli_closes_http_session_when_download_fails(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fflogs_scraper", "batch", "-e", "98", "--output", "raw/FRU"])
    monkeypatch.setattr(cli, "_load_dotenv", lambda: None)
    monkeypatch.setenv("FFLOGS_V2_CLIENT_ID", "test-id")
    monkeypatch.setenv("FFLOGS_V2_CLIENT_SECRET", "test-secret")
    closed = []
    monkeypatch.setattr(
        cli, "FFLogsV2Client",
        lambda *args, **kwargs: SimpleNamespace(cancel=lambda: None, close=lambda: closed.append(True)),
    )
    monkeypatch.setattr(cli.signal, "signal", lambda *args: None)

    def fail_download(*args):
        raise RuntimeError("下载失败")

    monkeypatch.setattr(cli, "_cmd_batch", fail_download)

    with pytest.raises(RuntimeError, match="下载失败"):
        cli.main()
    assert closed == [True, True]


def test_batch_interrupt_cancels_both_region_clients(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fflogs_scraper", "batch", "-e", "1079", "--output", "raw/FRU"])
    monkeypatch.setattr(cli, "_load_dotenv", lambda: None)
    monkeypatch.setenv("FFLOGS_V2_CLIENT_ID", "test-id")
    monkeypatch.setenv("FFLOGS_V2_CLIENT_SECRET", "test-secret")
    cancelled = []
    clients = []
    callbacks = []

    def make_client(*_args, server_region):
        client = SimpleNamespace(
            server_region=server_region,
            cancel=lambda region=server_region: cancelled.append(region),
            close=lambda: None,
        )
        clients.append(client)
        return client

    monkeypatch.setattr(cli, "FFLogsV2Client", make_client)
    monkeypatch.setattr(cli.signal, "signal", lambda _signal, callback: callbacks.append(callback))
    monkeypatch.setattr(cli, "_cmd_batch", lambda *_args: callbacks[0](None, None))
    cli.main()
    assert [client.server_region for client in clients] == ["CN", "NA"]
    assert cancelled == ["CN", "NA"]


@pytest.mark.parametrize("option,value", [
    ("--count", "0"), ("--count", "-1"), ("--max-pages", "0"),
    ("--partition", "0"), ("--partition", "-1"), ("--partition", "-2"), ("--bracket", "-1"),
])
def test_batch_invalid_limits_fail_before_authentication(monkeypatch, option, value):
    monkeypatch.setattr(sys, "argv", ["fflogs_scraper", "batch", "-e", "1079", option, value])
    def unexpected_auth():
        pytest.fail("无效参数不应读取凭证或请求 API")
    monkeypatch.setattr(cli, "_load_dotenv", unexpected_auth)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_batch_requires_output_before_authentication(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fflogs_scraper", "batch", "-e", "1079"])
    monkeypatch.setattr(cli, "_load_dotenv", lambda: pytest.fail("未提供输出目录不应读取凭证"))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
