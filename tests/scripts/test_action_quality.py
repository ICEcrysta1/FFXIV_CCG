"""离线标注契约、目录映射与失败保护测试。"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from threading import Barrier
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.action_quality import cli, config, runner
from scripts.common.json_io import atomic_write_json
from scripts.convert_fflogs import (
    build_skill_book,
    convert_report_payload,
    load_job_project_config,
)
from scripts.convert_fflogs.extraction.extraction import _get_ability_id
from training.config import load_run_config


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
    return config.BridgeConfig(tmp_path / "engine", tmp_path / "deps", "node", 30, 2)


def fake_process(monkeypatch, *, error=False, wrong_source=False, schema=2,
                 origin="pull_start", mutate=None):
    def run(arguments, **kwargs):
        assert arguments[1:] == [str(runner.RUNTIME)]
        request = json.loads(kwargs["input"])
        content = Path(request["source"]).read_bytes()
        assert "FFLOGS_V2_CLIENT_SECRET" not in kwargs["env"]
        analysis = {
            "schema_version": schema, "bridge_version": 3, "status": "annotated",
            "training_ready": False,
            "source": {
                "sha256": "wrong" if wrong_source else hashlib.sha256(content).hexdigest(),
                "fight_id": 5,
            },
            "engine": {"commit": "test-commit"},
            "time_basis": {"unit": "ms", "origin": origin, "report_offset_ms": 1000},
            "actor": {"id": str(request["source_id"])},
            "module_errors": ["failed"] if error else [],
            "fight_labels": [{"severity": "medium"}, {"severity": None}],
            "action_labels": [{"severity": None}],
        }
        if mutate is not None:
            mutate(analysis)
        return SimpleNamespace(returncode=0, stdout=json.dumps(analysis), stderr="")
    monkeypatch.setattr(runner.subprocess, "run", run)


def test_merge_preserves_raw_and_uses_shared_stage_layout(raw_file, bridge_config, monkeypatch):
    original = raw_file.read_bytes()
    monkeypatch.setenv("FFLOGS_V2_CLIENT_SECRET", "not-forwarded")
    fake_process(monkeypatch)
    output = runner.annotate_file(raw_file, config=bridge_config)
    assert output == raw_file.parents[3] / "annotated/FRU/00-10/fight.json"
    payload = json.loads(output.read_bytes())
    analysis = payload.pop("analysis")
    assert analysis["fight_labels"][0]["severity"] == "medium"
    assert "severity_weight" not in analysis["fight_labels"][0]
    assert "severity_weights" not in analysis
    assert payload == json.loads(original)
    assert raw_file.read_bytes() == original
    assert b"\r" not in output.read_bytes()


def test_incremental_skip_requires_matching_source_and_engine(raw_file, bridge_config, monkeypatch):
    fake_process(monkeypatch)
    output = runner.annotate_file(raw_file, config=bridge_config)
    assert runner.annotation_is_current(raw_file, output, commit="test-commit")
    assert not runner.annotation_is_current(raw_file, output, commit="other-commit")
    legacy = json.loads(output.read_text(encoding="utf-8"))
    legacy["analysis"]["bridge_version"] = 2
    legacy["analysis"]["severity_weights"] = {"medium": 0.5}
    legacy["analysis"]["fight_labels"][0]["severity_weight"] = 0.5
    atomic_write_json(output, legacy)
    assert not runner.annotation_is_current(raw_file, output, commit="test-commit")
    atomic_write_json(output, {**json.loads(raw_file.read_text(encoding="utf-8")),
                              "analysis": {**legacy["analysis"], "bridge_version": 3}})
    assert not runner.annotation_is_current(raw_file, output, commit="test-commit")
    output = runner.annotate_file(raw_file, config=bridge_config)
    assert runner.annotation_is_current(raw_file, output, commit="test-commit")

    monkeypatch.setattr(cli, "load_bridge_config", lambda: bridge_config)
    monkeypatch.setattr(cli, "analyzer_commit", lambda _config: "test-commit")
    monkeypatch.setattr(cli, "annotate_file", lambda *_args, **_kwargs: pytest.fail("valid output must skip Node"))
    monkeypatch.setattr(sys, "argv", ["action_quality", str(raw_file)])
    assert cli.main() == 0

    raw = json.loads(raw_file.read_text(encoding="utf-8"))
    raw["ranking"]["percentile"] = 4.0
    atomic_write_json(raw_file, raw)
    assert not runner.annotation_is_current(raw_file, output, commit="test-commit")
    reruns = []
    monkeypatch.setattr(cli, "annotate_file", lambda source, **_kwargs: reruns.append(source) or output)
    assert cli.main() == 0
    assert reruns == [raw_file]


def test_dirty_analyzer_does_not_reuse_old_annotation(bridge_config, monkeypatch):
    replies = iter([
        SimpleNamespace(returncode=0, stdout="test-commit\n"),
        SimpleNamespace(returncode=0, stdout=" M src/parser/core/Parser.ts\n"),
    ])
    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: next(replies))
    assert runner.analyzer_commit(bridge_config) is None


def test_cli_force_and_bounded_parallel_workers(raw_file, bridge_config, monkeypatch):
    second = raw_file.with_name("second.json")
    second.write_bytes(raw_file.read_bytes())
    barrier = Barrier(2, timeout=5)
    called = []

    def annotate(source, **_kwargs):
        called.append(source)
        barrier.wait()
        return source.with_suffix(".annotated.json")

    monkeypatch.setattr(cli, "load_bridge_config", lambda: bridge_config)
    monkeypatch.setattr(cli, "analyzer_commit", lambda _config: pytest.fail("--force must bypass reuse lookup"))
    monkeypatch.setattr(cli, "annotate_file", annotate)
    monkeypatch.setattr(sys, "argv", ["action_quality", str(raw_file.parent), "--force"])
    assert cli.main() == 0
    assert set(called) == {raw_file, second}


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


@pytest.mark.parametrize("mutate", [
    lambda analysis: analysis.pop("fight_labels"),
    lambda analysis: analysis.update(fight_labels={}),
    lambda analysis: analysis["fight_labels"][0].pop("severity"),
    lambda analysis: analysis["fight_labels"].append(None),
    lambda analysis: analysis.update(source=None),
])
def test_malformed_node_result_fails_without_publishing(raw_file, bridge_config, monkeypatch, mutate):
    fake_process(monkeypatch, mutate=mutate)
    with pytest.raises((TypeError, ValueError)):
        runner.annotate_file(raw_file, config=bridge_config)
    assert not runner.output_path_for_source(raw_file, None, None).exists()


@pytest.mark.parametrize("schema,origin", [(1, "pull_start"), (2, "report_start")])
def test_old_schema_or_wrong_time_origin_is_rejected(raw_file, bridge_config, monkeypatch, schema, origin):
    fake_process(monkeypatch, schema=schema, origin=origin)
    with pytest.raises(ValueError):
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
    atomic_write_json(tmp_path / "config/action_quality.yaml", {
        "bridge": {"analyzer_root": "engine", "timeout_seconds": 9, "max_workers": 2},
    })
    (tmp_path / ".env").write_text("ACTION_QUALITY_NODE=from-file\n", encoding="utf-8", newline="\n")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("ACTION_QUALITY_NODE", "from-process")
    monkeypatch.delenv("ACTION_QUALITY_NODE_MODULES", raising=False)
    settings = config.load_bridge_config()
    assert settings.node == "from-process"
    assert settings.node_modules == tmp_path / "engine/node_modules"
    assert settings.timeout == 9
    assert settings.max_workers == 2


@pytest.mark.parametrize("configured_node", [None, "node"])
def test_bridge_defaults_to_project_node(tmp_path, monkeypatch, configured_node):
    atomic_write_json(tmp_path / "config/action_quality.yaml", {
        "bridge": {"analyzer_root": "engine", "timeout_seconds": 9, "max_workers": 2},
    })
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    if configured_node is None:
        monkeypatch.delenv("ACTION_QUALITY_NODE", raising=False)
    else:
        monkeypatch.setenv("ACTION_QUALITY_NODE", configured_node)
    assert config.load_bridge_config().node == str(tmp_path / ".node/runtime/node.exe")


def test_invalid_parallel_worker_limit_is_rejected(tmp_path, monkeypatch):
    atomic_write_json(tmp_path / "config/action_quality.yaml", {
        "bridge": {"analyzer_root": "engine", "timeout_seconds": 9, "max_workers": 0},
    })
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    with pytest.raises(ValueError, match="max_workers"):
        config.load_bridge_config()


@pytest.mark.parametrize("input_stage,expected", [
    ("raw", "raw"),
    ("annotated", "raw"),
    ("annotated/FRU/00-10", "raw/FRU/00-10"),
])
def test_default_analysis_input_uses_raw_stage(tmp_path, monkeypatch, input_stage, expected):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "resolve_policy_model_config_path", lambda: tmp_path / "model.yaml")
    monkeypatch.setattr(config, "load_policy_config", lambda _path: {
        "raw_data_dir": f"data/human/job/black_mage/{input_stage}",
    })
    assert config.default_raw_root() == tmp_path / "data/human/job/black_mage" / expected


def test_artzip_model_defaults_to_annotated_for_conversion_and_training(monkeypatch):
    manifest = config.PROJECT_ROOT / "config/models/black_mage/artzip/config.yaml"
    run = load_run_config(manifest)
    assert run.raw_data_dir == config.PROJECT_ROOT / "data/human/job/black_mage/annotated"
    monkeypatch.setattr(config, "resolve_policy_model_config_path", lambda: manifest)
    assert config.default_raw_root() == config.PROJECT_ROOT / "data/human/job/black_mage/raw"


def test_cli_accepts_unset_model_variant(raw_file, bridge_config, monkeypatch, caplog):
    monkeypatch.delenv("FFXIV_MODEL_VARIANT", raising=False)
    monkeypatch.setattr(sys, "argv", ["action_quality", str(raw_file)])
    monkeypatch.setattr(cli, "load_bridge_config", lambda: bridge_config)
    monkeypatch.setattr(cli, "annotate_file", lambda *args, **kwargs: raw_file.with_name("done.json"))
    assert cli.main() == 0
    assert "未设置 FFXIV_MODEL_VARIANT" not in caplog.text


def test_real_node_output_can_attach_exact_cast_labels(tmp_path):
    """显式提供本机 raw 路径时，贯通真实 Node 分析和转换。"""
    source_name = os.environ.get("ACTION_QUALITY_E2E_SOURCE")
    if not source_name:
        pytest.skip("set ACTION_QUALITY_E2E_SOURCE to a complete raw report for this integration test")
    source = Path(source_name)
    output = runner.annotate_file(source, config=config.load_bridge_config(), output_root=tmp_path / "annotated")
    report = json.loads(output.read_text(encoding="utf-8"))
    analysis = report["analysis"]
    project = load_job_project_config(analysis["job_tag"])
    fight, _ = convert_report_payload(
        report,
        job_tag=analysis["job_tag"], project_config=project,
        skill_book=build_skill_book(project), source_id=report["source_id"],
        encounter_name="Integration", report_code=report["report_code"],
        player_name=report["player_name"], generated_at="2026-09-28T00:00:00Z",
    )
    assert fight is not None
    labelled = [action for action in fight["actions"] if action["quality_labels"]]
    assert labelled, "integration source must contain at least one enabled exact cast label"
    for action in labelled:
        event = report["events"][action["raw_event_index"]]
        assert event["type"] == "cast"
        assert action["skill_id"] == _get_ability_id(event)


def test_node_runtime_contracts():
    node = config.load_bridge_config().node
    if shutil.which(node) is None:
        pytest.skip(f"Node executable unavailable: {node}")
    result = subprocess.run(
        [node, "--test", str(Path(__file__).with_name("action_quality_runtime.test.cjs"))],
        capture_output=True, text=True, encoding="utf-8", timeout=15, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
