"""启动读取项目根目录 `.env` 端口设置的 TensorBoard 本地网页。"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

from common.policy.config import (
    PROJECT_ROOT,
    resolve_policy_model_config_path,
    resolve_policy_model_variant,
)
from common.project_config import (
    load_root_dotenv,
    resolve_project_path,
    resolve_tensorboard_port,
)
from common.training.tensorboard import resolve_tensorboard_root
from grpo.config import load_grpo_config, load_grpo_run_config
from training.config import load_run_config


def _is_ready(port: int) -> bool:
    """只把真正响应 TensorBoard API 的本机服务视为已启动。"""
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/data/plugins_listing", timeout=0.5
        ) as response:
            return isinstance(json.load(response), dict)
    except (OSError, ValueError, urllib.error.URLError):
        return False


def _record_path(port: int) -> Path:
    return Path(tempfile.gettempdir()) / f"ffxiv-ccg-tensorboard-{port}.json"


def _pid_is_alive(pid: object) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    if sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            return (
                bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)))
                and exit_code.value == 259
            )
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _check_existing_service(port: int, log_dir: Path) -> bool:
    if not _is_ready(port):
        return False
    record = _record_path(port)
    try:
        metadata = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        metadata = None
    if not isinstance(metadata, dict) or not _pid_is_alive(metadata.get("pid")):
        raise RuntimeError(
            f"端口 {port} 已有 TensorBoard，但无法确认它的事件目录；"
            "请关闭旧服务或修改 .env 的 TENSORBOARD_PORT。"
        )
    existing_dir = metadata.get("log_dir")
    if existing_dir != str(log_dir.resolve()):
        raise RuntimeError(
            f"端口 {port} 的 TensorBoard 正在读取 {existing_dir}，"
            f"当前模型需要 {log_dir.resolve()}；请关闭旧服务或修改 TENSORBOARD_PORT。"
        )
    return True


def _write_record(port: int, log_dir: Path, pid: int) -> None:
    record = _record_path(port)
    descriptor, temporary_name = tempfile.mkstemp(prefix=record.name, dir=record.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary:
            json.dump({"pid": pid, "log_dir": str(log_dir.resolve())}, temporary)
        os.replace(temporary_name, record)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _start_background(command: list[str], *, port: int, log_dir: Path) -> None:
    """启动后台服务并等待就绪，失败时保留日志供排查。"""
    if _check_existing_service(port, log_dir):
        print(f"TensorBoard 已在 http://127.0.0.1:{port} 运行。", flush=True)
        return

    descriptor, log_name = tempfile.mkstemp(
        prefix="ffxiv-ccg-tensorboard-", suffix=".log"
    )
    log_path = Path(log_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                ),
                start_new_session=(sys.platform != "win32"),
            )
    except Exception:
        log_path.unlink(missing_ok=True)
        raise

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if _is_ready(port):
            if process.poll() is not None:
                _check_existing_service(port, log_dir)
                raise RuntimeError(f"TensorBoard 进程已退出；请查看日志：{log_path}")
            try:
                _write_record(port, log_dir, process.pid)
            except Exception:
                process.terminate()
                raise
            print(f"TensorBoard 已就绪：http://127.0.0.1:{port}", flush=True)
            return
        if process.poll() is not None:
            detail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"TensorBoard 启动失败（日志：{log_path}）：\n{detail}")
        time.sleep(0.25)
    process.terminate()
    raise RuntimeError(f"TensorBoard 启动超时；请查看日志：{log_path}")


def main() -> int:
    """按所选模型配置启动 TensorBoard，网页仅监听本机回环地址。"""
    parser = argparse.ArgumentParser(
        description="启动项目 TensorBoard Web 视图；端口读取根目录 .env 的 TENSORBOARD_PORT"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="模型配置清单；默认按根目录 .env 选择职业和模型变体",
    )
    parser.add_argument(
        "--logdir",
        type=Path,
        default=None,
        help="覆盖默认事件目录；默认读取所选模型的 <model_variant>_tensorboard",
    )
    parser.add_argument(
        "--background", action="store_true", help="后台启动并等待网页就绪"
    )
    parser.add_argument("--open-browser", action="store_true", help="就绪后打开浏览器")
    parser.add_argument(
        "--if-enabled", action="store_true", help="仅在训练配置启用 TensorBoard 时启动"
    )
    parser.add_argument(
        "--scope",
        choices=("bc", "grpo"),
        default="bc",
        help="自动启动时检查 BC 或 GRPO 的 TensorBoard 开关",
    )
    args = parser.parse_args()

    load_root_dotenv(PROJECT_ROOT)
    port = resolve_tensorboard_port(project_root=PROJECT_ROOT)
    config_path = resolve_policy_model_config_path(args.config)
    config = (
        load_run_config(config_path)
        if args.scope == "bc"
        else load_grpo_run_config(config_path)
    )
    if args.if_enabled:
        tensorboard_config = (
            config.tensorboard
            if args.scope == "bc"
            else load_grpo_config(config_path).tensorboard
        )
        if not tensorboard_config.enabled:
            print(f"当前模型未启用 {args.scope.upper()} TensorBoard，跳过 Web 监控。", flush=True)
            return 0
    model_variant = resolve_policy_model_variant(config_path)
    log_dir = (
        resolve_project_path(args.logdir, project_root=PROJECT_ROOT)
        if args.logdir is not None
        else resolve_tensorboard_root(config.output_dir, model_variant)
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(log_dir),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    print(f"TensorBoard 网页: http://127.0.0.1:{port}", flush=True)
    print(f"事件目录: {log_dir}", flush=True)
    if args.background:
        _start_background(command, port=port, log_dir=log_dir)
        if args.open_browser:
            webbrowser.open(f"http://127.0.0.1:{port}")
        return 0
    return subprocess.call(command, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
