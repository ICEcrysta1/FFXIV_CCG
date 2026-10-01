"""离线标注契约、目录映射与失败保护测试。"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import CancelledError
from pathlib import Path
from threading import Barrier, Event, Thread, Timer
from types import SimpleNamespace

import pytest

from scripts.action_quality import cli, config, runner
from scripts.common.json_io import atomic_write_json, read_json
from scripts.convert_fflogs import (
    build_skill_book,
    convert_report_payload,
    load_job_project_config,
)
from scripts.convert_fflogs.extraction.extraction import _get_ability_id
from training.config import load_run_config


@pytest.fixture
def raw_file(tmp_path):
    source = tmp_path / "black_mage/raw/FRU/00-10/fight.json.br"
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
        assert "FFLOGS_V2_CLIENT_SECRET" not in kwargs["env"]
        analysis = {
            "schema_version": schema, "bridge_version": 4, "status": "annotated",
            "training_ready": False,
            "source": {
                "report_code": "wrong" if wrong_source else None,
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
    monkeypatch.setattr(runner, "_run_node", run)


def test_merge_preserves_raw_and_uses_shared_stage_layout(raw_file, bridge_config, monkeypatch):
    original = raw_file.read_bytes()
    monkeypatch.setenv("FFLOGS_V2_CLIENT_SECRET", "not-forwarded")
    fake_process(monkeypatch)
    output = runner.annotate_file(raw_file, config=bridge_config)
    assert output == raw_file.parents[3] / "annotated/FRU/00-10/fight.json.br"
    payload = read_json(output)
    analysis = payload.pop("analysis")
    assert "sha256" not in analysis["source"]
    assert analysis["fight_labels"][0]["severity"] == "medium"
    assert "severity_weight" not in analysis["fight_labels"][0]
    assert "severity_weights" not in analysis
    assert payload == read_json(raw_file)
    assert raw_file.read_bytes() == original
    assert output.read_bytes() != original


def test_incremental_skip_requires_matching_source_and_engine(raw_file, bridge_config, monkeypatch):
    fake_process(monkeypatch)
    output = runner.annotate_file(raw_file, config=bridge_config)
    assert runner.annotation_is_current(raw_file, output, commit="test-commit")
    assert not runner.annotation_is_current(raw_file, output, commit="other-commit")
    legacy = read_json(output)
    legacy["analysis"]["bridge_version"] = 2
    legacy["analysis"]["severity_weights"] = {"medium": 0.5}
    legacy["analysis"]["fight_labels"][0]["severity_weight"] = 0.5
    atomic_write_json(output, legacy)
    assert not runner.annotation_is_current(raw_file, output, commit="test-commit")
    atomic_write_json(output, {**read_json(raw_file),
                              "analysis": {**legacy["analysis"], "bridge_version": 4}})
    assert not runner.annotation_is_current(raw_file, output, commit="test-commit")
    output = runner.annotate_file(raw_file, config=bridge_config)
    assert runner.annotation_is_current(raw_file, output, commit="test-commit")

    monkeypatch.setattr(cli, "load_bridge_config", lambda: bridge_config)
    monkeypatch.setattr(cli, "analyzer_commit", lambda _config: "test-commit")
    monkeypatch.setattr(cli, "annotate_file", lambda *_args, **_kwargs: pytest.fail("valid output must skip Node"))
    monkeypatch.setattr(sys, "argv", ["action_quality", str(raw_file)])
    assert cli.main() == 0

    raw = read_json(raw_file)
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
    second = raw_file.with_name("second.json.br")
    second.write_bytes(raw_file.read_bytes())
    barrier = Barrier(2, timeout=5)
    called = []

    def annotate(source, **_kwargs):
        called.append(source)
        barrier.wait()
        return source.with_name(source.name + ".annotated")

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
    monkeypatch.setattr(runner, "_run_node", lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="broken module"))
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
    raw = read_json(raw_file)
    raw[field] = value
    atomic_write_json(raw_file, raw)
    def unexpected(*args, **kwargs):
        pytest.fail("invalid raw must not start Node")
    monkeypatch.setattr(runner, "_run_node", unexpected)
    with pytest.raises(ValueError):
        runner.annotate_file(raw_file, config=bridge_config)


def test_custom_root_keeps_encounter_and_bucket(tmp_path):
    source = tmp_path / "custom/FRU/00-10/a.json.br"
    assert runner.output_path_for_source(source, tmp_path / "evaluated", tmp_path / "custom") == tmp_path / "evaluated/FRU/00-10/a.json.br"
    with pytest.raises(ValueError, match="replace"):
        runner.output_path_for_source(source, tmp_path / "custom", tmp_path / "custom")


def test_validation_annotation_keeps_val_encounter_layout(tmp_path):
    source = tmp_path / "raw" / "VAL" / "FRU" / "a.json.br"
    assert runner.output_path_for_source(source, None, None) == (
        tmp_path / "annotated" / "VAL" / "FRU" / "a.json.br"
    )


def test_dotenv_reuses_loader_and_environment_wins(tmp_path, monkeypatch):
    atomic_write_json(tmp_path / "config/action_quality.yaml", {
        "bridge": {"analyzer_root": "engine", "timeout_seconds": 9},
    })
    (tmp_path / ".env").write_text(
        "ACTION_QUALITY_NODE=from-file\nACTION_QUALITY_WORKERS=2\n",
        encoding="utf-8", newline="\n",
    )
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("ACTION_QUALITY_NODE", "from-process")
    monkeypatch.delenv("ACTION_QUALITY_WORKERS", raising=False)
    monkeypatch.delenv("ACTION_QUALITY_NODE_MODULES", raising=False)
    settings = config.load_bridge_config()
    assert settings.node == "from-process"
    assert settings.node_modules == tmp_path / "engine/node_modules"
    assert settings.timeout == 9
    assert settings.max_workers == 2


@pytest.mark.parametrize("configured_node", [None, "node"])
def test_bridge_defaults_to_project_node(tmp_path, monkeypatch, configured_node):
    atomic_write_json(tmp_path / "config/action_quality.yaml", {
        "bridge": {"analyzer_root": "engine", "timeout_seconds": 9},
    })
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    if configured_node is None:
        monkeypatch.delenv("ACTION_QUALITY_NODE", raising=False)
    else:
        monkeypatch.setenv("ACTION_QUALITY_NODE", configured_node)
    assert config.load_bridge_config().node == str(tmp_path / ".node/runtime/node.exe")


def test_legacy_parallel_worker_limit_is_rejected(tmp_path, monkeypatch):
    atomic_write_json(tmp_path / "config/action_quality.yaml", {
        "bridge": {"analyzer_root": "engine", "timeout_seconds": 9, "max_workers": 0},
    })
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    with pytest.raises(ValueError, match="ACTION_QUALITY_WORKERS"):
        config.load_bridge_config()


@pytest.mark.parametrize("value", ["0", "-1", "2.5", "many"])
def test_invalid_parallel_worker_env_is_rejected(tmp_path, monkeypatch, value):
    atomic_write_json(tmp_path / "config/action_quality.yaml", {
        "bridge": {"analyzer_root": "engine", "timeout_seconds": 9},
    })
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("ACTION_QUALITY_WORKERS", value)
    with pytest.raises(ValueError, match="ACTION_QUALITY_WORKERS"):
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
    report = read_json(output)
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


@pytest.fixture
def lifecycle_node():
    node = config.load_bridge_config().node
    if shutil.which(node) is None:
        pytest.skip(f"Node executable unavailable: {node}")
    return node


@pytest.fixture
def launched_nodes(monkeypatch):
    """记录本测试持有的进程句柄，失败时也只清理这些进程。"""
    original = subprocess.Popen
    processes = []

    def launch(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    yield processes
    for process in processes:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)


@pytest.mark.parametrize("cancel", [False, True])
def test_node_timeout_or_cancel_reaps_process(lifecycle_node, launched_nodes, cancel):
    stop = Event()
    timer = Timer(0.4, stop.set)
    if cancel:
        timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(CancelledError if cancel else subprocess.TimeoutExpired):
            runner._run_node(
                [lifecycle_node, "-e", "setInterval(() => {}, 1000)"],
                input="{}", timeout=30 if cancel else 0.4, stop=stop,
                text=True, encoding="utf-8",
            )
    finally:
        timer.cancel()
        if cancel:
            timer.join()
    assert time.monotonic() - started < 5
    assert len(launched_nodes) == 1
    assert launched_nodes[0].poll() is not None
    assert all(pipe.closed for pipe in (launched_nodes[0].stdin, launched_nodes[0].stdout, launched_nodes[0].stderr))


def test_node_polling_preserves_large_input_and_output(lifecycle_node, launched_nodes):
    code = """
      let input = '';
      process.stdin.setEncoding('utf8');
      process.stdin.on('data', chunk => input += chunk);
      process.stdin.on('data', () => { if (!input.endsWith('\\n')) return; setTimeout(() => {
        process.stderr.write('diagnostic'.repeat(20000));
        process.stdout.write(input.replace(/\\r?\\n$/, ''));
        process.exitCode = 0;
        process.stdin.destroy();
      }, 450); });
    """
    payload = "中文测试" * 100000
    result = runner._run_node(
        [lifecycle_node, "-e", code], input=payload, timeout=10,
        text=True, encoding="utf-8",
    )
    assert result.returncode == 0
    assert result.stdout == payload
    assert result.stderr == "diagnostic" * 20000
    assert launched_nodes[0].poll() == 0


def test_cancelled_annotation_does_not_start_node(raw_file, bridge_config, monkeypatch):
    stop = Event()
    stop.set()
    monkeypatch.setattr(runner, "_run_node", lambda *_args, **_kwargs: pytest.fail("取消后不得启动 Node"))
    with pytest.raises(CancelledError):
        runner.annotate_file(raw_file, config=bridge_config, stop=stop)
    assert not runner.output_path_for_source(raw_file, None, None).exists()


def test_cancellation_before_publish_preserves_output(raw_file, bridge_config, monkeypatch):
    output = runner.output_path_for_source(raw_file, None, None)
    atomic_write_json(output, {"existing": True})
    before = output.read_bytes()
    stop = Event()
    fake_process(monkeypatch, mutate=lambda _analysis: stop.set())
    with pytest.raises(CancelledError):
        runner.annotate_file(raw_file, config=bridge_config, stop=stop)
    assert output.read_bytes() == before


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM] + (
    [signal.SIGBREAK] if hasattr(signal, "SIGBREAK") else []
))
def test_cli_repeated_interrupt_stops_running_nodes_and_pending_files(
    raw_file, bridge_config, lifecycle_node, launched_nodes, monkeypatch, tmp_path, signum,
):
    from dataclasses import replace

    for index in range(5):
        raw_file.with_name(f"pending-{index}.json.br").write_bytes(raw_file.read_bytes())
    output = runner.output_path_for_source(raw_file, None, None)
    atomic_write_json(output, {"existing": True})
    before = output.read_bytes()
    runtime = tmp_path / "blocked.cjs"
    runtime.write_text("setInterval(() => {}, 1000);", encoding="utf-8", newline="\n")
    monkeypatch.setattr(runner, "RUNTIME", runtime)
    monkeypatch.setattr(cli, "load_bridge_config", lambda: replace(bridge_config, node=lifecycle_node))
    monkeypatch.setattr(sys, "argv", ["action_quality", str(raw_file.parent), "--force"])
    original_wait = cli.wait
    previous_handler = signal.getsignal(signum)

    def interrupt(futures, **kwargs):
        deadline = time.monotonic() + 5
        while len(launched_nodes) < 2 and time.monotonic() < deadline:
            original_wait(futures, timeout=0.05)
        assert len(launched_nodes) == 2
        signal.raise_signal(signum)
        signal.raise_signal(signum)
        return original_wait(futures, **kwargs)

    monkeypatch.setattr(cli, "wait", interrupt)
    started = time.monotonic()
    assert cli.main() == 130
    assert time.monotonic() - started < 8
    assert len(launched_nodes) == 2
    assert all(process.poll() is not None for process in launched_nodes)
    assert signal.getsignal(signum) == previous_handler
    assert output.read_bytes() == before
    assert list(output.parent.glob("*.json.br")) == [output]


@pytest.fixture
def isolated_runtime(tmp_path):
    runtime = tmp_path / "run.cjs"
    runtime.write_bytes(runner.RUNTIME.read_bytes())
    (tmp_path / "bootstrap.cjs").write_text(
        "exports.bootstrap = () => {};", encoding="utf-8", newline="\n",
    )
    (tmp_path / "analyze.cjs").write_text(
        "exports.analyze = async request => request;", encoding="utf-8", newline="\n",
    )
    return runtime


def test_runtime_pipe_protocol_returns_result(lifecycle_node, isolated_runtime):
    result = runner._run_node(
        [lifecycle_node, str(isolated_runtime)], input='{"source_id": 2}',
        timeout=5, text=True, encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"source_id": 2}


@pytest.mark.parametrize("busy", [False, True])
def test_node_exits_when_owner_is_killed(lifecycle_node, isolated_runtime, busy):
    if busy:
        isolated_runtime.with_name("analyze.cjs").write_text(
            "exports.analyze = async () => { require('fs').writeSync(1, 'ready\\n'); while (true) {} };",
            encoding="utf-8", newline="\n",
        )
    code = """
import subprocess, sys, time
node = subprocess.Popen(sys.argv[1:3], stdin=subprocess.PIPE, stdout=sys.stdout, stderr=sys.stderr)
print(node.pid, flush=True)
if sys.argv[3] == 'busy':
    node.stdin.write(b'{}\\n')
    node.stdin.flush()
time.sleep(60)
"""
    owner = subprocess.Popen(
        [sys.executable, "-c", code, lifecycle_node, str(isolated_runtime), "busy" if busy else "waiting"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
    )
    node_pid = None
    cleaned = False
    try:
        node_pid = int(owner.stdout.readline())
        if busy:
            ready = Event()
            lines = []
            def read_ready():
                lines.append(owner.stdout.readline())
                ready.set()
            reader = Thread(target=read_ready, daemon=True)
            reader.start()
            assert ready.wait(timeout=5)
            reader.join(timeout=1)
            assert lines == ["ready\n"]
        assert owner.poll() is None
        owner.kill()
        # Node 继承输出管道，只有它也退出后 communicate 才会返回。
        _, stderr = owner.communicate(timeout=5)
        cleaned = True
        assert not stderr
    finally:
        if owner.poll() is None:
            owner.kill()
        if not cleaned and node_pid is not None:
            try:
                os.kill(node_pid, signal.SIGTERM)
            except OSError:
                pass
        owner.communicate(timeout=5)
