"""独立 Python 编译进程通过专属管道复用父进程中的唯一状态机引擎。"""

from __future__ import annotations

import builtins
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import multiprocessing
from multiprocessing.connection import Connection
from multiprocessing.util import Finalize
from queue import Empty
import sys
import threading
from types import SimpleNamespace


_WORKER_ENGINE = None


class _RecordedSpawnContext:
    """记录公开 Process 工厂返回的句柄，兼容没有 executor 终止 API 的 Python 3.12。"""

    def __init__(self, context, processes):
        self._context = context
        self._processes = processes

    def Process(self, *args, **kwargs):
        process = self._context.Process(*args, **kwargs)
        self._processes.append(process)
        return process

    def __getattr__(self, name):
        return getattr(self._context, name)


class ProcessCompilePool:
    """任务和结果由进程池传送；每个进程的状态机 RPC 由独立父线程独占处理。"""

    def __init__(self, engine, job_tag: str, max_processes: int):
        if isinstance(max_processes, bool) or not isinstance(max_processes, int) or max_processes < 1:
            raise ValueError("max_processes must be a positive integer")
        if engine.job_tag != job_tag or max_processes > engine.capacity:
            raise ValueError("process pool must match the shared engine job and capacity")
        self._engine = engine
        self._job_tag = job_tag
        self._max_processes = max_processes
        self._context = multiprocessing.get_context("spawn")
        self._initializer_queue = None
        self._executor = None
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._parent_endpoints: list[Connection] = []
        self._child_endpoints: list[Connection] = []
        self._processes = []
        self._process_exitcodes: dict[int, int | None] = {}
        self._entered = False
        self._closed = False

    def __enter__(self):
        if self._entered:
            raise RuntimeError("process compile pool cannot be entered twice")
        self._entered = True
        self._initializer_queue = self._context.Queue()
        try:
            for index in range(self._max_processes):
                parent, child = self._context.Pipe(duplex=True)
                self._parent_endpoints.append(parent)
                self._child_endpoints.append(child)
                # hello 证明 Queue feeder 已复制并交付句柄，之前不能关闭父端保留的 child 引用。
                self._initializer_queue.put(child)
                thread = threading.Thread(
                    target=_serve_backend_requests,
                    args=(parent, child, self._engine, self._stop_event),
                    name=f"fflogs-bridge-{index}", daemon=True,
                )
                self._threads.append(thread)
                thread.start()
            self._executor = ProcessPoolExecutor(
                max_workers=self._max_processes,
                mp_context=_RecordedSpawnContext(self._context, self._processes),
                initializer=_initialize_worker,
                initargs=(self._initializer_queue, self._job_tag),
            )
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def submit(self, task):
        if self._executor is None:
            raise RuntimeError("process compile pool is not running")
        return self._executor.submit(_run_task, task)

    def __exit__(self, exc_type, exc_value, traceback):
        if self._closed:
            return
        self._closed = True
        executor, self._executor = self._executor, None
        try:
            if exc_type is not None:
                # 这些句柄全部由本池的公开 Process 工厂创建，不访问 executor 的私有进程表。
                for process in self._processes:
                    if process.is_alive():
                        process.terminate()
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=exc_type is not None)
        finally:
            self._stop_event.set()
            queue, self._initializer_queue = self._initializer_queue, None
            if queue is not None:
                # 初始化失败可能留下未领取的 Connection；先排空，避免 feeder 在退出时阻塞。
                try:
                    while True:
                        queue.get(timeout=0.1).close()
                except Empty:
                    pass
                finally:
                    queue.close()
                    queue.join_thread()
            for endpoint in (*self._parent_endpoints, *self._child_endpoints):
                endpoint.close()
            for thread in self._threads:
                if thread.ident is not None:
                    thread.join()
            for process in self._processes:
                if process.pid is not None:
                    process.join()
                    self._process_exitcodes[process.pid] = process.exitcode
                process.close()


def _initialize_worker(connection_queue, job_tag: str) -> None:
    """子进程只初始化 Python 张量运行时，不装载 CLR 或创建第二个 C# 引擎。"""
    global _WORKER_ENGINE
    connection = connection_queue.get(timeout=10)
    Finalize(connection, connection.close, exitpriority=10)
    connection.send({"operation": "hello"})
    if not connection.poll(10):
        raise RuntimeError("parent simulation bridge did not acknowledge worker initialization")
    _receive_control(connection)
    from common.torch_dependencies import import_torch

    torch = import_torch()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _WORKER_ENGINE = _ProcessEngineProxy(connection, job_tag)


def _run_task(task):
    if _WORKER_ENGINE is None:
        raise RuntimeError("process compilation worker is not initialized")
    from .cache_compile import _compile_raw_source_worker

    try:
        return _compile_raw_source_worker(task, engine=_WORKER_ENGINE)
    finally:
        _WORKER_ENGINE.close(suppress_errors=sys.exc_info()[0] is not None)


def _exception_control(error: Exception) -> dict[str, object]:
    supplied_args = getattr(error, "args", ())
    args = supplied_args if all(value is None or type(value) in (str, int, float, bool) for value in supplied_args) else None
    return {"ok": False, "error_type": type(error).__name__, "error_module": type(error).__module__,
            "error_message": str(error), "error_args": args}


def _receive_control(connection: Connection):
    response = connection.recv()
    if not isinstance(response, dict) or "ok" not in response:
        raise RuntimeError("invalid parent simulation bridge response")
    if not response["ok"]:
        error_type = getattr(builtins, str(response.get("error_type")), RuntimeError)
        if response.get("error_module") != "builtins" or not isinstance(error_type, type) or not issubclass(error_type, Exception):
            error_type = RuntimeError
        args = response.get("error_args") if response.get("error_module") == "builtins" else None
        error = error_type(*(args if args is not None else (response.get("error_message", "simulation bridge failed"),)))
        error.remote_exception_type = f"{response.get('error_module')}.{response.get('error_type')}"
        raise error
    return response


def _serve_backend_requests(connection: Connection, child_endpoint: Connection, engine, stopped: threading.Event) -> None:
    """每个管道只在本线程读写；任何失败都释放当前文件的会话。"""
    backend = None

    def close_backend():
        nonlocal backend
        current, backend = backend, None
        if current is not None:
            current.close()

    try:
        while not stopped.is_set():
            if not connection.poll(0.1):
                continue
            request = connection.recv()
            try:
                if not isinstance(request, dict):
                    raise ValueError("simulation bridge request must be a mapping")
                operation = request.get("operation")
                if operation == "hello":
                    child_endpoint.close()
                    value = None
                elif operation == "create":
                    if backend is not None:
                        raise RuntimeError("worker already owns a simulation queue")
                    backend = engine.create_backend(**request["kwargs"])
                    value = getattr(backend, "queue_id", None)
                elif operation == "close":
                    close_backend()
                    value = None
                elif operation in ("init", "facts", "replay"):
                    if backend is None:
                        raise RuntimeError("worker does not own a simulation queue")
                    if operation == "init":
                        value = backend.init(**request["kwargs"])
                    elif operation == "facts":
                        value = asdict(backend.apply_external_events(request["events"]))
                    else:
                        packet = backend.replay_policy_sequence_packet(
                            request["requests_json"], request["facts_json"],
                            fight_duration=request["fight_duration"], policy_action_key=request["policy_action_key"],
                        )
                        connection.send({"ok": True, "packet": True})
                        connection.send_bytes(packet)
                        del packet
                        continue
                else:
                    raise ValueError(f"unsupported simulation bridge operation: {operation!r}")
                connection.send({"ok": True, "value": value})
            except Exception as error:
                try:
                    close_backend()
                except Exception:
                    pass
                connection.send(_exception_control(error))
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        try:
            close_backend()
        finally:
            connection.close()


class _ProcessEngineProxy:
    def __init__(self, connection: Connection, job_tag: str):
        self._connection = connection
        self.job_tag = job_tag
        self._backend = None

    def _request(self, operation: str, **kwargs):
        self._connection.send({"operation": operation, **kwargs})
        return _receive_control(self._connection)

    def create_backend(self, *, max_history, **kwargs):
        if max_history is not None:
            raise ValueError("process compilation requires complete history")
        response = self._request("create", kwargs={"max_history": max_history, **kwargs})
        self._backend = _ProcessBackendProxy(self, response["value"])
        return self._backend

    def close(self, *, suppress_errors=False):
        if self._backend is not None:
            try:
                self._backend.close()
            except Exception:
                if not suppress_errors:
                    raise


class _ProcessBackendProxy:
    def __init__(self, engine: _ProcessEngineProxy, queue_id):
        self._engine = engine
        self.job_tag = engine.job_tag
        self.queue_id = queue_id
        self._closed = False

    def _request(self, operation: str, **kwargs):
        if self._closed:
            raise RuntimeError("process compilation backend is closed")
        return self._engine._request(operation, **kwargs)

    def init(self, **kwargs):
        return self._request("init", kwargs=kwargs)["value"]

    def apply_external_events(self, events):
        return SimpleNamespace(**self._request("facts", events=events)["value"])

    def replay_policy_sequence(self, requests, facts, *, fight_duration, policy_action_key):
        response = self._request(
            "replay", requests_json=json.dumps(requests, allow_nan=False, separators=(",", ":")),
            facts_json=json.dumps(facts, allow_nan=False, separators=(",", ":")),
            fight_duration=float(fight_duration), policy_action_key=policy_action_key,
        )
        if not response.get("packet"):
            raise RuntimeError("parent simulation bridge did not return an observation packet")
        import pickle

        return pickle.loads(self._engine._connection.recv_bytes())["observations"]

    def close(self):
        if self._closed:
            return
        propagating_error = sys.exc_info()[0] is not None
        try:
            self._request("close")
        except Exception:
            if not propagating_error:
                raise
        finally:
            self._closed = True
            self._engine._backend = None
