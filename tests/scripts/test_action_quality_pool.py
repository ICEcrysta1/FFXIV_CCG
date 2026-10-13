"""常驻 Node 客户端的协议、取消、背压与资源回收测试。"""

from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Event
import time

import pytest

from scripts.action_quality.config import BridgeConfig
from scripts.action_quality import node_pool


FAKE_NODE = r'''
import json
import os
import sys
from threading import Event, Lock, Thread
import time

lock = Lock()
jobs = {}
active = 0
peak = 0
config = {}

def send(value):
    with lock:
        print(json.dumps(value), flush=True)

def analyze(message):
    global active, peak
    source = json.loads(message['source'])
    cancelled = Event()
    with lock:
        jobs[message['id']] = cancelled
        active += 1
        peak = max(peak, active)
    try:
        mode = source.get('mode', 'ok')
        if mode == 'crash':
            time.sleep(source.get('delay', 0))
            os._exit(7)
        if mode == 'malformed':
            with lock:
                print('broken json', flush=True)
            return
        if mode == 'eof':
            os.close(1)
            time.sleep(30)
            return
        if mode == 'stderr':
            sys.stderr.write('x' * (256 * 1024) + 'TAIL-MARKER')
            sys.stderr.flush()
        if cancelled.wait(30 if mode == 'hang' else source.get('delay', 0)):
            send({'id': message['id'], 'type': 'error', 'error': {'message': 'cancelled'}})
        elif mode == 'error':
            send({'id': message['id'], 'type': 'error',
                  'error': {'name': 'Error', 'message': 'task failed', 'stack': 'stub task failed'}})
        else:
            send({'id': message['id'], 'type': 'result', 'analysis': {
                'source': source, 'source_id': message['source_id'], 'pid': os.getpid(),
                'peak': peak, 'workers': config['workers'],
                'memory_budget_mb': config['memory_budget_mb'], 'arguments': sys.argv[1:],
                'credentials': [key for key in os.environ if key.startswith('FFLOGS_V2_CLIENT_')],
            }})
    finally:
        with lock:
            jobs.pop(message['id'], None)
            active -= 1

for line in sys.stdin:
    message = json.loads(line)
    kind = message['type']
    if kind == 'init':
        config = message
        mode = os.path.basename(config['node_modules'])
        if mode == 'no-ready':
            time.sleep(30)
        if mode == 'slow-ready':
            time.sleep(0.2)
        send({'id': 0, 'type': 'ready'})
    elif kind == 'analyze':
        Thread(target=analyze, args=(message,), daemon=True).start()
    elif kind == 'cancel':
        with lock:
            cancelled = jobs.get(message['id'])
        if cancelled:
            cancelled.set()
    elif kind == 'shutdown':
        break
'''


@pytest.fixture
def fake_config(tmp_path):
    executable = tmp_path / "fake-node"
    executable.write_text(f"#!{sys.executable}\n" + FAKE_NODE, encoding="utf-8")
    executable.chmod(0o755)
    runtime = tmp_path / "run.cjs"
    runtime.write_text("", encoding="utf-8")
    return BridgeConfig(tmp_path, tmp_path / "deps", str(executable), 3, 2), runtime


def request(name="sample", **values):
    return {"source": json.dumps({"name": name, **values}), "source_id": name}


def real_request(name="sample", **values):
    return {**request(name, **values), "source_id": 2}


def wait_until(predicate, *, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("等待异步清理超时")
        time.sleep(0.01)


def assert_reaped(pool):
    process = pool._process
    assert process.poll() is not None
    assert all(not thread.is_alive() for thread in pool._threads)
    assert all(stream.closed for stream in (process.stdin, process.stdout, process.stderr))
    assert not pool._pending


def test_pool_is_lazy_and_pre_cancelled_request_does_not_spawn(fake_config, monkeypatch):
    config, runtime = fake_config

    def unexpected(*args, **kwargs):
        raise AssertionError("没有任务时不应启动 Node")

    monkeypatch.setattr(node_pool.subprocess, "Popen", unexpected)
    with node_pool.NodePool(config, runtime=runtime) as pool:
        assert pool._process is None
        stop = Event()
        stop.set()
        with pytest.raises(CancelledError):
            pool.annotate(request(), stop=stop)
    pool.close()


def test_default_workers_obey_memory_and_cpu_budget(fake_config, monkeypatch):
    config, runtime = fake_config
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1, 2})
    with node_pool.NodePool(replace(config, max_workers=20, memory_budget_mb=1024), runtime=runtime) as pool:
        assert pool.workers == 1
    with node_pool.NodePool(config, workers=3, runtime=runtime) as pool:
        assert pool.workers == 3


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_invalid_worker_count_is_rejected(fake_config, workers):
    config, runtime = fake_config
    with pytest.raises(ValueError, match="positive integer"):
        node_pool.NodePool(config, workers=workers, runtime=runtime)


def test_concurrent_ids_reuse_host_and_apply_backpressure(fake_config, monkeypatch):
    config, runtime = fake_config
    monkeypatch.setenv("FFLOGS_V2_CLIENT_ID", "secret-id")
    monkeypatch.setenv("FFLOGS_V2_CLIENT_SECRET", "secret-value")
    with node_pool.NodePool(config, workers=2, runtime=runtime) as pool:
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(pool.annotate, request(str(index), delay=0.1 if index == 0 else 0.02))
                       for index in range(6)]
            results = [future.result(timeout=5) for future in futures]
        assert [result["source_id"] for result in results] == [str(index) for index in range(6)]
        assert [result["source"]["name"] for result in results] == [str(index) for index in range(6)]
        assert {result["pid"] for result in results} == {pool._process.pid}
        assert max(result["peak"] for result in results) == 2
        assert all(result["credentials"] == [] for result in results)
        assert all(result["workers"] == 2 and result["memory_budget_mb"] == 2048 for result in results)
        assert results[0]["arguments"] == ["--expose-gc", str(runtime.resolve()), "--pool"]
    assert_reaped(pool)


def test_task_error_does_not_poison_host(fake_config):
    config, runtime = fake_config
    with node_pool.NodePool(config, runtime=runtime) as pool:
        with pytest.raises(RuntimeError, match="stub task failed"):
            pool.annotate(request(mode="error"))
        pid = pool._process.pid
        assert pool.annotate(request("after-error"))["pid"] == pid
    assert_reaped(pool)


def test_timeout_releases_slot_and_ignores_late_cancel_result(fake_config):
    config, runtime = fake_config
    with node_pool.NodePool(replace(config, timeout=0.25), workers=1, runtime=runtime) as pool:
        with pytest.raises(subprocess.TimeoutExpired):
            pool.annotate(request(mode="hang"))
        pid = pool._process.pid
        result = pool.annotate(request("after-timeout"))
        assert result["source_id"] == "after-timeout"
        assert result["pid"] == pid
        assert not pool._pending
    assert_reaped(pool)


def test_cancel_running_and_waiting_requests_without_killing_host(fake_config):
    config, runtime = fake_config
    with node_pool.NodePool(config, workers=1, runtime=runtime) as pool:
        pool.annotate(request("warmup"))
        running_stop, waiting_stop = Event(), Event()
        with ThreadPoolExecutor(max_workers=2) as executor:
            running = executor.submit(pool.annotate, request(mode="hang"), running_stop)
            wait_until(lambda: bool(pool._pending))
            waiting = executor.submit(pool.annotate, request("waiting"), waiting_stop)
            waiting_stop.set()
            with pytest.raises(CancelledError):
                waiting.result(timeout=2)
            running_stop.set()
            with pytest.raises(CancelledError):
                running.result(timeout=2)
        assert pool.annotate(request("after-cancel"))["source_id"] == "after-cancel"
    assert_reaped(pool)


def test_initialization_and_task_share_one_deadline(fake_config):
    config, runtime = fake_config
    config = replace(config, node_modules=config.node_modules.with_name("slow-ready"), timeout=0.3)
    with node_pool.NodePool(config, runtime=runtime) as pool:
        started = time.monotonic()
        with pytest.raises(subprocess.TimeoutExpired):
            pool.annotate(request(delay=0.2))
        assert time.monotonic() - started < 0.45
        assert pool._ready
        assert pool.annotate(request("after-timeout"))["source_id"] == "after-timeout"


def test_initialization_timeout_reaps_host_and_fails_future_requests(fake_config):
    config, runtime = fake_config
    config = replace(config, node_modules=config.node_modules.with_name("no-ready"), timeout=0.2)
    with node_pool.NodePool(config, runtime=runtime) as pool:
        with pytest.raises(subprocess.TimeoutExpired):
            pool.annotate(request())
        with pytest.raises(RuntimeError, match="initialization failed"):
            pool.annotate(request())
    assert_reaped(pool)


@pytest.mark.parametrize("mode", ["crash", "eof", "malformed"])
def test_host_failure_finishes_all_pending_and_is_terminal(fake_config, mode):
    config, runtime = fake_config
    with node_pool.NodePool(config, workers=2, runtime=runtime) as pool:
        pool.annotate(request("warmup"))
        with ThreadPoolExecutor(max_workers=2) as executor:
            pending = executor.submit(pool.annotate, request(mode="hang"))
            wait_until(lambda: bool(pool._pending))
            broken = executor.submit(pool.annotate, request(mode=mode, delay=0.03))
            for future in (pending, broken):
                with pytest.raises(RuntimeError, match="analysis pool"):
                    future.result(timeout=3)
        with pytest.raises(RuntimeError, match="analysis pool"):
            pool.annotate(request("after-failure"))
    assert_reaped(pool)


def test_stderr_tail_is_bounded_and_drained(fake_config):
    config, runtime = fake_config
    with node_pool.NodePool(config, runtime=runtime) as pool:
        pool.annotate(request(mode="stderr"))
        wait_until(lambda: pool.stderr_tail.endswith("TAIL-MARKER"))
        assert len(pool.stderr_tail.encode()) == node_pool._STDERR_LIMIT
        assert pool._stderr_size == node_pool._STDERR_LIMIT
    assert_reaped(pool)


def test_concurrent_close_finishes_pending_and_only_stops_own_process_group(fake_config):
    config, runtime = fake_config
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    pool = node_pool.NodePool(config, runtime=runtime)
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            pending = executor.submit(pool.annotate, request(mode="hang"))
            wait_until(lambda: pool._ready and bool(pool._pending))
            first = executor.submit(pool.close)
            second = executor.submit(pool.close)
            first.result(timeout=3)
            second.result(timeout=3)
            with pytest.raises(CancelledError):
                pending.result(timeout=3)
        assert unrelated.poll() is None
        assert_reaped(pool)
        pool.close()
        with pytest.raises(RuntimeError, match="closed"):
            pool.annotate(request())
    finally:
        pool.close()
        unrelated.terminate()
        unrelated.wait(timeout=3)


def test_close_during_reader_startup_waits_for_reader_cleanup(fake_config, monkeypatch):
    config, runtime = fake_config
    before_start, allow_start = Event(), Event()
    original_start = node_pool.Thread.start

    def delayed_start(thread):
        if thread.name == "action-quality-stdout":
            before_start.set()
            assert allow_start.wait(timeout=3)
        return original_start(thread)

    monkeypatch.setattr(node_pool.Thread, "start", delayed_start)
    pool = node_pool.NodePool(config, runtime=runtime)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            pending = executor.submit(pool.annotate, request())
            assert before_start.wait(timeout=3)
            closing = executor.submit(pool.close)
            wait_until(lambda: pool._closed)
            assert not closing.done()
            allow_start.set()
            closing.result(timeout=3)
            with pytest.raises((CancelledError, RuntimeError)):
                pending.result(timeout=3)
        assert_reaped(pool)
    finally:
        allow_start.set()
        pool.close()


@pytest.fixture
def real_runtime(tmp_path):
    node = Path(__file__).resolve().parents[2] / ".node/runtime/bin/node"
    node_executable = str(node) if node.is_file() else shutil.which("node")
    if not node_executable:
        pytest.skip("需要 Node 来验证真实 worker_threads 协议")
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    for source in node_pool.RUNTIME.parent.glob("*.cjs"):
        shutil.copyfile(source, runtime_dir / source.name)
    (runtime_dir / "bootstrap.cjs").write_text("exports.bootstrap = () => {};\n", encoding="utf-8")
    (runtime_dir / "analyze.cjs").write_text("""
const {threadId} = require('worker_threads');
const fs = require('fs');
let count = 0;
exports.analyze = async request => {
  const source = JSON.parse(request.source);
  count += 1;
  if (source.mode === 'hang') while (true) {}
  if (source.mode === 'error') throw new Error('stub analysis failed');
  if (source.marker) {
    fs.writeFileSync(source.marker, 'ready');
    while (!source.barrier.every(marker => fs.existsSync(marker))) {
      await new Promise(resolve => setTimeout(resolve, 5));
    }
  }
  await new Promise(resolve => setTimeout(resolve, source.delay || 0));
  return {threadId, count, source, source_id: request.source_id, finished_at: Date.now(),
    has_gc: typeof global.gc === 'function'};
};
""", encoding="utf-8")
    return BridgeConfig(tmp_path, tmp_path / "deps", node_executable, 3, 2), runtime_dir / "run.cjs"


def test_real_runtime_parallel_ids_and_worker_module_reuse(real_runtime):
    config, runtime = real_runtime
    with node_pool.NodePool(config, workers=2, runtime=runtime) as pool:
        with ThreadPoolExecutor(max_workers=2) as executor:
            slow = executor.submit(pool.annotate, real_request("slow", delay=150))
            fast = executor.submit(pool.annotate, real_request("fast", delay=20))
            first, second = slow.result(timeout=5), fast.result(timeout=5)
        assert first["source"]["name"] == "slow" and second["source"]["name"] == "fast"
        assert first["threadId"] != second["threadId"]
        assert first["has_gc"] and second["has_gc"]
        pid = pool._process.pid
        followups = [pool.annotate(real_request(f"followup-{index}")) for index in range(4)]
        assert pool._process.pid == pid
        assert {result["threadId"] for result in followups} <= {first["threadId"], second["threadId"]}
        assert max(result["count"] for result in followups) > 1
    assert_reaped(pool)


def test_real_runtime_error_then_success_and_timeout_replaces_only_worker(real_runtime):
    config, runtime = real_runtime
    config = replace(config, timeout=0.7)
    with node_pool.NodePool(config, workers=1, runtime=runtime) as pool:
        before = pool.annotate(real_request("before"))
        pid = pool._process.pid
        with pytest.raises(RuntimeError, match="stub analysis failed"):
            pool.annotate(real_request(mode="error"))
        assert pool.annotate(real_request("after-error"))["source"]["name"] == "after-error"
        with pytest.raises(subprocess.TimeoutExpired):
            pool.annotate(real_request(mode="hang"))
        after = pool.annotate(real_request("after-timeout"))
        assert pool._process.pid == pid
        assert after["threadId"] != before["threadId"]
        assert after["count"] == 1
    assert_reaped(pool)


def test_real_runtime_soft_budget_retires_idle_workers_and_preserves_active_tasks(real_runtime):
    config, runtime = real_runtime
    config = replace(config, memory_budget_mb=1)
    markers = [str(config.analyzer_root / f"{name}.ready") for name in ("slow", "fast")]
    with node_pool.NodePool(config, workers=2, runtime=runtime) as pool:
        with ThreadPoolExecutor(max_workers=2) as executor:
            slow = executor.submit(pool.annotate, real_request("slow", delay=150, marker=markers[0], barrier=markers))
            fast = executor.submit(pool.annotate, real_request("fast", delay=5, marker=markers[1], barrier=markers))
            first, second = slow.result(timeout=5), fast.result(timeout=5)
        # 两线程都进入分析后，快任务触发预算回收；慢任务仍须完成，不能被误杀。
        assert first["source"]["name"] == "slow" and second["source"]["name"] == "fast"
        assert first["threadId"] != second["threadId"]
        assert first["finished_at"] > second["finished_at"]
        wait_until(lambda: pool.stderr_tail.count("retire idle worker") >= 2)
        assert "capacity=1" in pool.stderr_tail
        third = pool.annotate(real_request("after-shrink"))
        assert third["threadId"] not in {first["threadId"], second["threadId"]}
        assert third["count"] == 1
    assert_reaped(pool)


def test_real_runtime_stdin_eof_reaps_host_and_threads(real_runtime):
    config, runtime = real_runtime
    environment = {key: value for key, value in os.environ.items()
                   if key not in {"FFLOGS_V2_CLIENT_ID", "FFLOGS_V2_CLIENT_SECRET"}}
    process = subprocess.Popen([config.node, "--expose-gc", str(runtime), "--pool"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               cwd=config.analyzer_root, env=environment, text=True)
    try:
        init = {"id": 0, "type": "init", "analyzer_root": str(config.analyzer_root),
                "node_modules": str(config.node_modules), "workers": 2,
                "timeout_seconds": 3, "memory_budget_mb": 2048}
        process.stdin.write(json.dumps(init) + "\n")
        process.stdin.flush()
        assert json.loads(process.stdout.readline()) == {"id": 0, "type": "ready"}
        process.stdin.close()
        assert process.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()
