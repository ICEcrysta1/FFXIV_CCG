"""部署目录的原子发布与 Windows 文件占用重试。"""

from __future__ import annotations

import shutil
import time
import uuid
from collections.abc import Callable
from pathlib import Path

PUBLISH_RENAME_ATTEMPTS = 5
PUBLISH_RENAME_DELAY_SECONDS = 0.5


def publish_directory(
    temp_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool,
    rename_with_retry_fn: Callable[..., None],
) -> None:
    if not output_dir.exists():
        temp_dir.rename(output_dir)
        return
    if not overwrite:
        raise FileExistsError(f"output directory already exists: {output_dir}")
    backup = output_dir.parent / f".{output_dir.name}.backup-{uuid.uuid4().hex}"
    rename_with_retry_fn(
        output_dir,
        backup,
        operation="move existing deployment package to backup",
    )
    try:
        rename_with_retry_fn(
            temp_dir,
            output_dir,
            operation="publish new deployment package",
        )
    except BaseException:
        rename_with_retry_fn(
            backup,
            output_dir,
            operation="restore previous deployment package",
        )
        raise
    shutil.rmtree(backup)


def rename_with_retry(
    source: Path,
    destination: Path,
    *,
    operation: str,
    sleep: Callable[[float], None] | None = None,
) -> None:
    """重试 Windows 导出目录的短暂拒绝访问，最终失败时保留原始异常。"""
    sleep_fn = time.sleep if sleep is None else sleep
    for attempt in range(1, PUBLISH_RENAME_ATTEMPTS + 1):
        try:
            source.rename(destination)
            return
        except PermissionError as exc:
            if attempt == PUBLISH_RENAME_ATTEMPTS:
                raise PermissionError(
                    f"{operation} failed after {PUBLISH_RENAME_ATTEMPTS} attempts: "
                    f"{source} -> {destination}; close processes or file viewers "
                    "that use the deployment package and retry"
                ) from exc
            sleep_fn(PUBLISH_RENAME_DELAY_SECONDS)
