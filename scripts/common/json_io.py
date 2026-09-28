"""脚本共用的 UTF-8/LF 原子 JSON 写入。"""

import json
import os
import tempfile
from pathlib import Path


def atomic_write_json(output_path: str | Path, payload: object) -> None:
    """完整序列化成功后才替换目标；失败时保留已有文件并清理临时文件。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n",
            dir=path.parent, suffix=".tmp", delete=False,
        ) as output:
            temporary_path = Path(output.name)
            json.dump(payload, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
