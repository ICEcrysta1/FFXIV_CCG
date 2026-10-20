"""转换子进程、共享状态机 RPC 与资源归属的回归测试。"""

from __future__ import annotations

import multiprocessing
import os
import pickle
from contextlib import contextmanager
from math import copysign
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace

import pytest

from common.policy.data import Normalizer
from scripts.common.inprocess_backend import InProcessEngine
from scripts.common.json_io import atomic_write_json
from scripts.common.state_machine_types import ExternalEventResult
from scripts.convert_fflogs.cache import precompile_raw_training_caches
from scripts.convert_fflogs.cache import cache_compile as cache_compile_module
from scripts.convert_fflogs.cache import process_compile as process_compile_module


class _FakeBackend:
    def __init__(self, queue_id):
        self.queue_id = queue_id
        self.closed = False
        self.close_count = 0
        self.calls = []

    def init(self, **kwargs):
        self.calls.append(("init", kwargs))
        if kwargs.get("initial_timestamp") == -1:
            raise ValueError("invalid initial timestamp", -1)
        return kwargs.get("initial_timestamp", 0.0)

    def apply_external_events(self, events):
        self.calls.append(("facts", events))
        return ExternalEventResult(True, "", "movement", 1.0, None)

    def replay_policy_sequence_packet(self, requests_json, facts_json, **kwargs):
        self.calls.append(("replay", requests_json, facts_json, kwargs))
        return pickle.dumps({"observations": [{
            "context": {"value": -0.0},
            "history_markers": [{
                "token": "<dead/>", "timestamp": 1.0, "after_history_length": 0, "step": 1,
            }],
        }]})

    def close(self):
        self.closed = True
        self.close_count += 1


class _FakeEngine:
    job_tag = "black_mage"
    capacity = 4

    def __init__(self):
        self.backends = []

    def create_backend(self, **kwargs):
        assert kwargs["max_history"] is None
        backend = _FakeBackend(len(self.backends) + 1)
        self.backends.append(backend)
        return backend

    @property
    def active_count(self):
        return sum(not backend.closed for backend in self.backends)

    def close(self):
        pytest.fail("RPC 服务不得关闭借用的引擎")


@contextmanager
def _fake_bridge():
    parent, child = multiprocessing.Pipe(duplex=True)
    stopped = Event()
    engine = _FakeEngine()
    # 同进程测试无需传递重复句柄；单独记录父端 hello 的释放动作。
    released = []
    copied_child_endpoint = SimpleNamespace(close=lambda: released.append(True))
    thread = Thread(
        target=process_compile_module._serve_backend_requests,
        args=(parent, copied_child_endpoint, engine, stopped),
    )
    thread.start()
    try:
        child.send({"operation": "hello"})
        assert child.poll(5)
        assert process_compile_module._receive_control(child)["ok"]
        assert released == [True]
        yield engine, child, thread
    finally:
        child.close()
        stopped.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert parent.closed
        assert engine.active_count == 0


def test_process_proxy_transfers_packed_context_and_releases_only_its_queue():
    with _fake_bridge() as (engine, connection, _thread):
        proxy = process_compile_module._ProcessEngineProxy(connection, "black_mage")
        backend = proxy.create_backend(max_history=None)
        assert backend.queue_id == 1 and engine.active_count == 1
        assert backend.init(initial_timestamp=1.0) == 1.0
        result = backend.apply_external_events([{"timestamp": 1.0, "kind": "movement"}])
        assert vars(result) == {
            "accepted": True, "reason": "", "event_kind": "movement", "timestamp": 1.0,
            "next_scheduled_event_time": None, "token": None, "history_cursor": None,
        }
        frames = backend.replay_policy_sequence(
            [{"timestamp": 1.0, "action": "fire"}], [],
            fight_duration=10.0, policy_action_key="fire",
        )
        assert copysign(1.0, frames[0]["context"]["value"]) == -1.0
        assert frames[0]["history_markers"] == [{
            "token": "<dead/>", "timestamp": 1.0, "after_history_length": 0, "step": 1,
        }]
        replay = engine.backends[0].calls[-1]
        assert replay == (
            "replay", '[{"timestamp":1.0,"action":"fire"}]', '[]',
            {"fight_duration": 10.0, "policy_action_key": "fire"},
        )
        proxy.close()
        proxy.close()
        assert engine.active_count == 0 and engine.backends[0].close_count == 1
        with pytest.raises(RuntimeError, match="backend is closed"):
            backend.init()


@pytest.mark.parametrize("rpc_request,exception,message", [
    ([], TypeError, "must be a mapping"),
    ({"operation": "unknown"}, ValueError, "unsupported simulation bridge operation"),
    ({"operation": "create"}, KeyError, "kwargs"),
    ({"operation": "init", "kwargs": {}}, RuntimeError, "does not own a simulation queue"),
])
def test_invalid_rpc_request_preserves_exception_and_allows_next_queue(rpc_request, exception, message):
    with _fake_bridge() as (engine, connection, _thread):
        connection.send(rpc_request)
        assert connection.poll(5)
        with pytest.raises(exception, match=message):
            process_compile_module._receive_control(connection)
        proxy = process_compile_module._ProcessEngineProxy(connection, "black_mage")
        proxy.create_backend(max_history=None)
        assert engine.active_count == 1
        proxy.close()


@pytest.mark.parametrize("operation,kwargs,exception,message", [
    ("create", {"kwargs": {"max_history": None}}, RuntimeError, "already owns"),
    ("init", {"kwargs": {"initial_timestamp": -1}}, ValueError, "invalid initial timestamp"),
    ("facts", {}, KeyError, "events"),
    ("replay", {}, KeyError, "requests_json"),
])
def test_failed_rpc_releases_existing_queue(operation, kwargs, exception, message):
    with _fake_bridge() as (engine, connection, _thread):
        proxy = process_compile_module._ProcessEngineProxy(connection, "black_mage")
        proxy.create_backend(max_history=None)
        connection.send({"operation": operation, **kwargs})
        assert connection.poll(5)
        with pytest.raises(exception, match=message):
            process_compile_module._receive_control(connection)
        assert engine.active_count == 0 and engine.backends[0].close_count == 1
        proxy.create_backend(max_history=None)
        assert engine.active_count == 1
        proxy.close()


def test_rpc_eof_releases_queue_without_closing_engine():
    with _fake_bridge() as (engine, connection, thread):
        proxy = process_compile_module._ProcessEngineProxy(connection, "black_mage")
        proxy.create_backend(max_history=None)
        connection.close()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert engine.active_count == 0 and engine.backends[0].close_count == 1


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_process_pool_rejects_invalid_process_counts(value):
    with pytest.raises(ValueError, match="positive integer"):
        process_compile_module.ProcessCompilePool(_FakeEngine(), "black_mage", value)


@pytest.mark.parametrize("job_tag,count", [("white_mage", 1), ("black_mage", 5)])
def test_process_pool_rejects_mismatched_engine(job_tag, count):
    with pytest.raises(ValueError, match="shared engine job and capacity"):
        process_compile_module.ProcessCompilePool(_FakeEngine(), job_tag, count)


def _child_diagnostics():
    """在真正的 spawn worker 内检查运行时，避免只验证父进程配置。"""
    import torch

    from scripts.common import inprocess_backend
    from scripts.convert_fflogs.cache import process_compile

    return {
        "pid": os.getpid(), "start_method": multiprocessing.get_start_method(),
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "native_loaded": inprocess_backend._NATIVE_MODULE is not None,
        "engine_type": type(process_compile._WORKER_ENGINE).__name__,
    }


def _child_crash_with_queue():
    from scripts.convert_fflogs.cache import process_compile

    process_compile._WORKER_ENGINE.create_backend(max_history=None)
    os._exit(7)


def _child_wait_with_queue():
    from scripts.convert_fflogs.cache import process_compile

    process_compile._WORKER_ENGINE.create_backend(max_history=None)
    Event().wait(120)


def _assert_pool_closed(pool):
    assert pool._process_exitcodes
    assert all(exitcode is not None for exitcode in pool._process_exitcodes.values())
    assert all(not thread.is_alive() for thread in pool._threads)
    assert all(endpoint.closed for endpoint in (*pool._parent_endpoints, *pool._child_endpoints))
    for process in pool._processes:
        with pytest.raises(ValueError, match="closed"):
            process.is_alive()
    with pytest.raises(RuntimeError, match="not running"):
        pool.submit(())
    # 异常清理后由外层再次释放同一个对象，也应保持幂等。
    pool.__exit__(None, None, None)


def test_spawn_workers_use_one_torch_thread_and_do_not_load_native(cs_backend):
    previous_children = {child.pid for child in multiprocessing.active_children()}
    with InProcessEngine("black_mage", capacity=2) as engine:
        with process_compile_module.ProcessCompilePool(engine, "black_mage", 2) as pool:
            diagnostics = [pool._executor.submit(_child_diagnostics) for _ in range(4)]
            for future in diagnostics:
                result = future.result(timeout=60)
                assert result["pid"] != os.getpid()
                assert result["start_method"] == "spawn"
                assert result["torch_threads"] == result["torch_interop_threads"] == 1
                assert not result["native_loaded"]
                assert result["engine_type"] == "_ProcessEngineProxy"
        assert len(pool._processes) == 2
        _assert_pool_closed(pool)
        assert engine.active_count == 0 and engine._engine is not None
    _assert_no_new_children(previous_children)


def test_crashed_child_releases_queue_and_preserves_borrowed_engine(cs_backend):
    from concurrent.futures.process import BrokenProcessPool

    previous_children = {child.pid for child in multiprocessing.active_children()}
    with InProcessEngine("black_mage", capacity=1) as engine:
        with process_compile_module.ProcessCompilePool(engine, "black_mage", 1) as pool:
            future = pool._executor.submit(_child_crash_with_queue)
            with pytest.raises(BrokenProcessPool):
                future.result(timeout=60)
        _assert_pool_closed(pool)
        assert 7 in pool._process_exitcodes.values()
        assert engine.active_count == 0
        with engine.create_backend(max_history=None) as backend:
            assert backend.queue_id > 0
    _assert_no_new_children(previous_children)


def test_parent_exception_terminates_worker_and_releases_queue(monkeypatch, cs_backend):
    previous_children = {child.pid for child in multiprocessing.active_children()}
    with InProcessEngine("black_mage", capacity=1) as engine:
        created = Event()
        backends = []
        create_backend = engine.create_backend

        def track_backend(**kwargs):
            backend = create_backend(**kwargs)
            backends.append(backend)
            created.set()
            return backend

        monkeypatch.setattr(engine, "create_backend", track_backend)
        with pytest.raises(ValueError, match="parent conversion interrupted"):
            with process_compile_module.ProcessCompilePool(engine, "black_mage", 1) as pool:
                pool._executor.submit(_child_wait_with_queue)
                assert created.wait(timeout=60)
                raise ValueError("parent conversion interrupted")
        _assert_pool_closed(pool)
        assert len(backends) == 1 and backends[0]._closed
        assert engine.active_count == 0 and engine._engine is not None
    _assert_no_new_children(previous_children)


def _write_source(path: Path, *, events=None) -> Path:
    if events is None:
        events = [
            {"type": "cast", "sourceID": 10, "timestamp": 1000 + step * 5000,
             "abilityGameID": 152 if step % 2 == 0 else 154}
            for step in range(6)
        ]
    atomic_write_json(path, {
        "source_id": 10, "report_code": "PROCESS", "fight_id": 1, "events": events,
    })
    return path.resolve()


def _assert_no_new_children(previous_pids: set[int]) -> None:
    assert not [
        process for process in multiprocessing.active_children()
        if process.pid not in previous_pids
    ]


def test_process_failures_match_serial_and_preserve_borrowed_engine(
    tmp_path, monkeypatch, caplog, cs_backend,
):
    """子进程的单文件异常沿用原文案，并释放自己的队列后继续后续源。"""
    torch = pytest.importorskip("torch")
    good = _write_source(tmp_path / "good.json.br")
    empty = _write_source(tmp_path / "empty.json.br", events=[])
    bad_brotli = tmp_path / "bad_brotli.json.br"
    bad_brotli.write_bytes(b"invalid brotli")
    # 此错误发生在申请队列之后，覆盖 raw_source 的 finally 释放路径。
    bad_events = tmp_path / "bad_events.json.br"
    atomic_write_json(bad_events, {"source_id": 10, "events": None})
    sources = [bad_brotli, bad_events, empty, good]
    error_messages = []
    previous_children = {child.pid for child in multiprocessing.active_children()}
    monkeypatch.setattr(cache_compile_module.os, "sched_getaffinity", lambda _pid: {0, 1})

    with InProcessEngine("black_mage", capacity=3) as engine:
        with engine.create_backend(max_history=2) as other_backend:
            for workers in (1, 2):
                caplog.clear()
                cache_dir = tmp_path / f"cache_{workers}"
                with caplog.at_level("ERROR"):
                    result = precompile_raw_training_caches(
                        sources, job_tag="black_mage", normalizer=Normalizer(),
                        int_dtype=torch.int32, float_dtype=torch.float32,
                        cache_dir=cache_dir, shard_size=2, max_workers=workers, engine=engine,
                    )
                assert result == [good]
                assert engine.active_count == 1
                assert other_backend.queue_id > 0
                assert engine._engine is not None
                assert not list(cache_dir.rglob("*.tmp.*"))
                errors = sorted(
                    record.getMessage() for record in caplog.records
                    if "raw JSON 缓存编译失败，跳过并继续:" in record.getMessage()
                )
                assert len(errors) == 2
                assert any("invalid Brotli JSON" in message for message in errors)
                assert any("NoneType" in message for message in errors)
                error_messages.append(errors)
                _assert_no_new_children(previous_children)
        assert engine.active_count == 0
    assert error_messages[0] == error_messages[1]
