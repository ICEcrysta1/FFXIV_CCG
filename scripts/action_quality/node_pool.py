"""通过有界 NDJSON 请求复用一个 Node 宿主及其分析线程。"""

from __future__ import annotations

import atexit
from collections import deque
from concurrent.futures import CancelledError, Future, TimeoutError as FutureTimeoutError
import json
import math
import os
from pathlib import Path
import select
import signal
import subprocess
from threading import BoundedSemaphore, Event, Lock, Thread, current_thread
import time

from .config import BridgeConfig, annotation_worker_count

RUNTIME = Path(__file__).parent / "runtime" / "run.cjs"
_STDERR_LIMIT = 64 * 1024
_POLL_SECONDS = 0.05


class NodePool:
    """首次请求才启动宿主；调用方可并发 annotate，最多保留 workers 个在途分析。"""

    def __init__(self, config: BridgeConfig, *, workers: int | None = None, runtime: Path = RUNTIME):
        self.config = config
        self.workers = annotation_worker_count(config, config.max_workers) if workers is None else workers
        if isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers < 1:
            raise ValueError("analysis workers must be a positive integer")
        self._timeout = float(config.timeout)
        if not math.isfinite(self._timeout) or self._timeout <= 0:
            raise ValueError("analysis timeout must be positive and finite")
        self._runtime = Path(runtime).resolve()
        self._command = [config.node, "--expose-gc", str(self._runtime), "--pool"]
        self._lock = Lock()
        self._start_lock = Lock()
        self._write_lock = Lock()
        self._kill_lock = Lock()
        self._close_lock = Lock()
        self._slots = BoundedSemaphore(self.workers)
        self._process: subprocess.Popen | None = None
        self._threads: list[Thread] = []
        self._pending: dict[int, Future] = {}
        self._next_id = 1
        self._ready = False
        self._closed = False
        self._failure: str | None = None
        self._group_stopped = False
        self._stderr_lock = Lock()
        self._stderr_chunks: deque[bytes] = deque()
        self._stderr_size = 0
        atexit.register(self.close)

    @property
    def stderr_tail(self) -> str:
        with self._stderr_lock:
            return b"".join(self._stderr_chunks).decode("utf-8", errors="replace")

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def annotate(self, request: dict, stop: Event | None = None) -> dict:
        """总超时包含首次启动、等待容量和分析；取消后忽略该 ID 的迟到结果。"""
        if not isinstance(request, dict):
            raise TypeError("analysis request must be a mapping")
        deadline = time.monotonic() + self._timeout
        self._ensure_started(deadline, stop)
        self._acquire(self._slots, deadline, stop)
        request_id = None
        try:
            with self._lock:
                self._raise_unavailable_locked()
                request_id = self._next_id
                self._next_id += 1
                future = Future()
                self._pending[request_id] = future
                process = self._process
            self._send(process, {
                "id": request_id, "type": "analyze", "source": request["source"],
                "source_id": request["source_id"],
            }, deadline, stop)
            return self._wait(future, deadline, stop)
        except (CancelledError, subprocess.TimeoutExpired):
            if request_id is not None:
                self._cancel(request_id)
            raise
        finally:
            if request_id is not None:
                with self._lock:
                    abandoned = self._pending.pop(request_id, None)
                if abandoned is not None:
                    abandoned.cancel()
            self._slots.release()

    def _raise_unavailable_locked(self):
        if self._closed:
            raise RuntimeError("analysis pool is closed")
        if self._failure is not None:
            # 每次创建独立异常，避免反复 raise 同一对象积累大批失败任务的 traceback。
            raise RuntimeError(self._failure)

    def _check(self, deadline: float, stop: Event | None):
        if stop is not None and stop.is_set():
            raise CancelledError("标注已取消")
        if time.monotonic() >= deadline:
            raise subprocess.TimeoutExpired(self._command, self._timeout, stderr=self.stderr_tail)

    def _acquire(self, lock, deadline: float, stop: Event | None):
        while True:
            self._check(deadline, stop)
            with self._lock:
                self._raise_unavailable_locked()
            if lock.acquire(timeout=min(_POLL_SECONDS, max(0, deadline - time.monotonic()))):
                return

    def _ensure_started(self, deadline: float, stop: Event | None):
        self._check(deadline, stop)
        with self._lock:
            self._raise_unavailable_locked()
            if self._ready:
                return
        self._acquire(self._start_lock, deadline, stop)
        try:
            with self._lock:
                self._raise_unavailable_locked()
                if self._ready:
                    return
                environment = dict(os.environ)
                for key in ("FFLOGS_V2_CLIENT_ID", "FFLOGS_V2_CLIENT_SECRET"):
                    environment.pop(key, None)
                process = subprocess.Popen(
                    self._command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    cwd=self.config.analyzer_root, env=environment, start_new_session=True,
                )
                self._process = process
                os.set_blocking(process.stdin.fileno(), False)
                ready = Future()
                self._pending[0] = ready
                self._threads = [
                    Thread(target=self._read_stdout, args=(process,), name="action-quality-stdout", daemon=True),
                    Thread(target=self._read_stderr, args=(process,), name="action-quality-stderr", daemon=True),
                ]
            for thread in self._threads:
                thread.start()
            self._send(process, {
                "id": 0, "type": "init", "analyzer_root": str(self.config.analyzer_root),
                "node_modules": str(self.config.node_modules), "workers": self.workers,
                "timeout_seconds": self._timeout,
                "memory_budget_mb": getattr(self.config, "memory_budget_mb", 2048),
            }, deadline, stop)
            self._wait(ready, deadline, stop)
            with self._lock:
                self._raise_unavailable_locked()
                self._ready = True
        except BaseException as error:
            self._fail(f"analysis pool initialization failed: {error}")
            raise
        finally:
            self._start_lock.release()

    def _wait(self, future: Future, deadline: float, stop: Event | None):
        while True:
            self._check(deadline, stop)
            try:
                result = future.result(timeout=min(_POLL_SECONDS, max(0, deadline - time.monotonic())))
                self._check(deadline, stop)
                return result
            except FutureTimeoutError:
                if future.done():
                    raise

    def _send(self, process, message: dict, deadline: float, stop: Event | None = None):
        payload = json.dumps(message, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
        acquired = False
        written = 0
        try:
            while not acquired:
                self._check(deadline, stop)
                acquired = self._write_lock.acquire(timeout=min(_POLL_SECONDS, max(0, deadline - time.monotonic())))
            if process is None or process.poll() is not None:
                raise BrokenPipeError("analysis host is not running")
            descriptor = process.stdin.fileno()
            view = memoryview(payload)
            while written < len(payload):
                self._check(deadline, stop)
                try:
                    written += os.write(descriptor, view[written:])
                except (BlockingIOError, InterruptedError):
                    select.select([], [descriptor], [], min(_POLL_SECONDS, max(0, deadline - time.monotonic())))
        except (OSError, ValueError) as error:
            self._fail(f"analysis pool input failed: {error}")
            raise RuntimeError(f"analysis pool input failed: {error}") from error
        except (CancelledError, subprocess.TimeoutExpired):
            if written:
                # 半行写入无法安全重发 cancel；终止宿主，避免后续 NDJSON 被拼成损坏请求。
                self._fail("analysis pool request was interrupted during transmission")
            raise
        finally:
            if acquired:
                self._write_lock.release()

    def _cancel(self, request_id: int):
        with self._lock:
            future = self._pending.pop(request_id, None)
            process = self._process
        if future is not None:
            future.cancel()
        try:
            self._send(process, {"id": request_id, "type": "cancel"}, time.monotonic() + 0.2)
        except Exception:
            pass

    def _read_stdout(self, process):
        try:
            for line in process.stdout:
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise ValueError("analysis pool response must be an object")
                request_id = message.get("id")
                kind = message.get("type")
                if isinstance(request_id, bool) or not isinstance(request_id, int) or request_id < 0:
                    raise ValueError("analysis pool response requires a non-negative request ID")
                if (kind not in ("ready", "result", "error") or (kind == "ready" and request_id != 0)
                        or (request_id == 0 and kind not in ("ready", "error"))):
                    raise ValueError("unsupported analysis pool response")
                with self._lock:
                    future = self._pending.pop(request_id, None)
                if future is None:
                    continue
                if kind == "ready":
                    future.set_result(None)
                elif kind == "result":
                    analysis = message.get("analysis")
                    if isinstance(analysis, dict):
                        future.set_result(analysis)
                    else:
                        future.set_exception(TypeError("analysis result must be a JSON object"))
                else:
                    error = message.get("error")
                    if not isinstance(error, dict):
                        error = {"message": "analysis worker failed"}
                    detail = str(error.get("stack") or error.get("message") or "analysis worker failed")
                    future.set_exception(RuntimeError(f"analysis process failed:\n{detail}"))
            self._fail(f"analysis pool stdout closed (exit={process.poll()})")
        except (OSError, ValueError, TypeError) as error:
            self._fail(f"analysis pool protocol failed: {error}")

    def _read_stderr(self, process):
        try:
            while chunk := process.stderr.read1(4096):
                with self._stderr_lock:
                    self._stderr_chunks.append(chunk)
                    self._stderr_size += len(chunk)
                    while self._stderr_size > _STDERR_LIMIT:
                        overflow = self._stderr_size - _STDERR_LIMIT
                        first = self._stderr_chunks.popleft()
                        removed = min(overflow, len(first))
                        self._stderr_size -= removed
                        if removed < len(first):
                            self._stderr_chunks.appendleft(first[removed:])
        except (OSError, ValueError):
            pass

    def _fail(self, message: str):
        with self._lock:
            if self._closed or self._failure is not None:
                return
            detail = self.stderr_tail.strip()
            self._failure = f"{message}\n{detail}" if detail else message
            pending = list(self._pending.values())
            self._pending.clear()
            process = self._process
        for future in pending:
            future.set_exception(RuntimeError(self._failure))
        self._stop_process(process)

    def _stop_process(self, process):
        if process is None:
            return
        with self._kill_lock:
            if self._group_stopped:
                return
            # start_new_session 保证 PGID 仅属于本池启动的宿主及其后代，不按进程名或全局 PID 查杀。
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
            self._group_stopped = True

    def close(self):
        with self._close_lock:
            self._close()

    def _close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            pending = list(self._pending.values())
            self._pending.clear()
            process = self._process
        atexit.unregister(self.close)
        for future in pending:
            future.set_exception(CancelledError("标注已取消"))
        if process is not None:
            try:
                self._send(process, {"type": "shutdown"}, time.monotonic() + 0.2)
                process.wait(timeout=0.3)
            except (OSError, RuntimeError, subprocess.TimeoutExpired):
                pass
            self._stop_process(process)
            # 启动方可能刚创建 Popen、尚未启动读线程；等其退出启动段再关闭句柄。
            with self._start_lock:
                pass
            for thread in self._threads:
                if thread is not current_thread() and thread.ident is not None:
                    thread.join()
            with self._write_lock:
                for stream in (process.stdin, process.stdout, process.stderr):
                    stream.close()
