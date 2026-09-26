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
        [sys.executable, "-B", "-m", "scripts.fflogs_scraper", "--help"],
        cwd=Path(__file__).resolve().parents[3],
        env=process_env,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "{single,batch,encounters}" in result.stdout


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
    (["batch", "-e", "1079", "--mode", "events-only"],
     "batch", {"encounter": 1079, "mode": "events-only", "metric": "rdps", "count": 200, "partition": None}),
    (["batch", "-e", "1079", "--count", "203", "--partition", "25", "--output", "raw/FRU"],
     "batch", {"count": 203, "partition": 25, "output": "raw/FRU"}),
    (["encounters", "-z", "39"], "encounters", {"zone": 39}),
])
def test_cli_dispatch_preserves_arguments(monkeypatch, arguments, command, expected):
    monkeypatch.setattr(sys, "argv", ["fflogs_scraper", *arguments])
    monkeypatch.setattr(cli, "_load_dotenv", lambda: None)
    monkeypatch.setenv("FFLOGS_V2_CLIENT_ID", "test-id")
    monkeypatch.setenv("FFLOGS_V2_CLIENT_SECRET", "test-secret")
    client = SimpleNamespace(cancel=lambda: None)
    credentials = []
    monkeypatch.setattr(cli, "FFLogsV2Client", lambda *args: credentials.append(args) or client)
    monkeypatch.setattr(cli.signal, "signal", lambda *args: None)
    calls = []
    for name in ("single", "batch", "encounters"):
        monkeypatch.setattr(
            cli, f"_cmd_{name}",
            lambda actual_client, args, name=name: calls.append((name, actual_client, args)),
        )
    cli.main()
    assert credentials == [("test-id", "test-secret")]
    assert len(calls) == 1
    actual_command, actual_client, args = calls[0]
    assert actual_command == command
    assert actual_client is client
    assert all(getattr(args, key) == value for key, value in expected.items())


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
