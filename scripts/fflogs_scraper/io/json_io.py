"""下载 JSON 的 UTF-8/LF 原子写入。"""

import json
import os
import tempfile
from pathlib import Path


def _write_download_json(output_path: str, result: dict) -> None:
    """仅在完整序列化成功后替换目标，避免留下被批量任务误跳过的半文件。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n",
            dir=path.parent, suffix=".tmp", delete=False,
        ) as output:
            temporary_path = Path(output.name)
            json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
