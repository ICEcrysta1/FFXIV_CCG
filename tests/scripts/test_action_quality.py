"""离线标注契约、目录映射与失败保护测试。"""

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.action_quality import config, runner
from scripts.common.json_io import atomic_write_json


@pytest.fixture
def raw_file(tmp_path):
    source = tmp_path / "black_mage/raw/FRU/00-10/fight.json"
    atomic_write_json(source, {
        "source_id": 2, "fight_id": 5, "events_complete": True,
        "friendlies": [{"id": 2, "type": "BlackMage"}],
        "events": [{"type": "cast", "timestamp": 123, "ability": {"guid": 152}}],
        "ranking": {"percentile": 3.4}, "player_name": "中文玩家",
    })
    return source


@pytest.fixture
def bridge_config(tmp_path):
    return config.BridgeConfig(tmp_path / "engine", tmp_path / "deps", "node", 30)


def fake_process(monkeypatch, *, error=False, wrong_source=False):
    def run(arguments, **kwargs):
        request = json.loads(Path(arguments[-2]).read_text(encoding="utf-8"))
        content = Path(request["source"]).read_bytes()
        assert "FFLOGS_V2_CLIENT_SECRET" not in kwargs["env"]
        atomic_write_json(arguments[-1], {
            "schema_version": 1, "source": {"sha256": "wrong" if wrong_source else hashlib.sha256(content).hexdigest()},
            "actor": {"id": str(request["source_id"])},
            "module_errors": ["failed"] if error else [],
            "fight_labels": [{"severity": "medium"}, {"severity": None}],
            "action_labels": [{"severity": None}],
        })
        return SimpleNamespace(returncode=0, stderr="")
    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(runner, "severity_weights", lambda job: {"medium": 0.5})


def test_merge_preserves_raw_and_uses_shared_stage_layout(raw_file, bridge_config, monkeypatch):
    original = raw_file.read_bytes()
    monkeypatch.setenv("FFLOGS_V2_CLIENT_SECRET", "not-forwarded")
    fake_process(monkeypatch)
    output = runner.annotate_file(raw_file, config=bridge_config)
    assert output == raw_file.parents[3] / "annotated/FRU/00-10/fight.json"
    payload = json.loads(output.read_bytes())
    assert payload.pop("analysis")["fight_labels"][0]["severity_weight"] == 0.5
    assert payload == json.loads(original)
    assert raw_file.read_bytes() == original
    assert b"\r" not in output.read_bytes()


@pytest.mark.parametrize("error,wrong_source", [(True, False), (False, True)])
def test_invalid_analysis_preserves_existing_output(raw_file, bridge_config, monkeypatch, error, wrong_source):
    output = runner.output_path_for_source(raw_file, None, None)
    atomic_write_json(output, {"existing": True})
    before = output.read_bytes()
    fake_process(monkeypatch, error=error, wrong_source=wrong_source)
    with pytest.raises(ValueError):
        runner.annotate_file(raw_file, config=bridge_config)
    assert output.read_bytes() == before


def test_process_failure_does_not_publish(raw_file, bridge_config, monkeypatch):
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="broken module"))
    with pytest.raises(RuntimeError, match="broken module"):
        runner.annotate_file(raw_file, config=bridge_config)
    assert not runner.output_path_for_source(raw_file, None, None).exists()


@pytest.mark.parametrize("field,value", [("events_complete", False), ("source_id", True), ("source_id", 99)])
def test_invalid_raw_fails_before_starting_node(raw_file, bridge_config, monkeypatch, field, value):
    raw = json.loads(raw_file.read_bytes())
    raw[field] = value
    atomic_write_json(raw_file, raw)
    def unexpected(*args, **kwargs):
        pytest.fail("invalid raw must not start Node")
    monkeypatch.setattr(runner.subprocess, "run", unexpected)
    with pytest.raises(ValueError):
        runner.annotate_file(raw_file, config=bridge_config)


def test_custom_root_keeps_encounter_and_bucket(tmp_path):
    source = tmp_path / "custom/FRU/00-10/a.json"
    assert runner.output_path_for_source(source, tmp_path / "evaluated", tmp_path / "custom") == tmp_path / "evaluated/FRU/00-10/a.json"
    with pytest.raises(ValueError, match="replace"):
        runner.output_path_for_source(source, tmp_path / "custom", tmp_path / "custom")


def test_dotenv_reuses_loader_and_environment_wins(tmp_path, monkeypatch):
    atomic_write_json(tmp_path / "config/action_quality.yaml", {"bridge": {"analyzer_root": "engine", "timeout_seconds": 9}})
    (tmp_path / ".env").write_text("ACTION_QUALITY_NODE=from-file\n", encoding="utf-8", newline="\n")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("ACTION_QUALITY_NODE", "from-process")
    monkeypatch.delenv("ACTION_QUALITY_NODE_MODULES", raising=False)
    settings = config.load_bridge_config()
    assert settings.node == "from-process"
    assert settings.node_modules == tmp_path / "engine/node_modules"
    assert settings.timeout == 9


def test_job_weight_configuration_isolated(tmp_path, monkeypatch):
    atomic_write_json(tmp_path / "config/models/black_mage/artzip/config.yaml", {
        "action_quality": {"severity_weights": {"minor": 0.25, "medium": 0.5, "major": 1.0}},
    })
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("FFXIV_MODEL_VARIANT", "artzip")
    assert config.severity_weights("black_mage")["major"] == 1.0
    assert config.severity_weights("machinist") == {}


def test_node_runtime_contracts():
    result = subprocess.run(
        ["node", "--test", str(Path(__file__).with_name("action_quality_runtime.test.cjs"))],
        capture_output=True, text=True, encoding="utf-8", timeout=15, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
