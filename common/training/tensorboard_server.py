"""启动读取项目根目录 `.env` 端口设置的 TensorBoard 本地网页。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common.policy.config import PROJECT_ROOT, resolve_policy_model_config_path
from common.project_config import (
    load_root_dotenv,
    resolve_project_path,
    resolve_tensorboard_port,
)
from training.config import load_run_config


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
        help="覆盖默认事件目录；默认读取所选模型 output_dir/tensorboard",
    )
    args = parser.parse_args()

    load_root_dotenv(PROJECT_ROOT)
    port = resolve_tensorboard_port(project_root=PROJECT_ROOT)
    config_path = resolve_policy_model_config_path(args.config)
    config = load_run_config(config_path)
    log_dir = (
        resolve_project_path(args.logdir, project_root=PROJECT_ROOT)
        if args.logdir is not None
        else config.output_dir / "tensorboard"
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
    return subprocess.call(command, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
