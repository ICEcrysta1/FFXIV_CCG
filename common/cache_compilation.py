"""跨脚本复用的 raw JSON 到 compiled cache 调用入口。"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def compile_raw_training_cache(
    *,
    source_path: Path,
    cache_dir: Path,
    cache_shard_size: int,
    job_tag: str,
) -> None:
    """调用正式转换 CLI 编译单个 raw JSON，不在调用方复制转换逻辑。"""
    command = [
        sys.executable,
        "-m",
        "scripts.convert_fflogs.cli",
        str(Path(source_path).resolve()),
        "--job-tag",
        job_tag,
        "--cache-root",
        str(Path(cache_dir).resolve()),
        "--shard-size",
        str(cache_shard_size),
        "--workers",
        "1",
    ]
    try:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"failed to compile cache for raw source: {source_path}"
        ) from exc
