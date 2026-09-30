"""脚本共用的 Brotli JSON 读取与原子写入。"""

import json
import os
import tempfile
from pathlib import Path

import brotli


JSON_BROTLI_SUFFIX = ".json.br"


def is_json_file(path: str | Path) -> bool:
    """识别训练数据使用的 Brotli JSON 文件。"""
    return Path(path).name.lower().endswith(JSON_BROTLI_SUFFIX)


def read_json_bytes(path: str | Path) -> bytes:
    """读取并解压训练数据 JSON 的 UTF-8 字节。"""
    source = Path(path)
    if not is_json_file(source):
        raise ValueError(f"expected {JSON_BROTLI_SUFFIX} dataset file: {source}")
    try:
        return brotli.decompress(source.read_bytes())
    except brotli.error as error:
        raise ValueError(f"invalid Brotli JSON: {source}") from error


def read_json(path: str | Path) -> object:
    """读取 Brotli JSON，保留标准 JSON 解析语义。"""
    return json.loads(read_json_bytes(path))


def atomic_write_json(output_path: str | Path, payload: object) -> None:
    """完整序列化成功后才替换目标；失败时保留已有文件并清理临时文件。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, suffix=".tmp", delete=False,
        ) as output:
            temporary_path = Path(output.name)
            if path.name.lower().endswith(JSON_BROTLI_SUFFIX):
                content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
                output.write(brotli.compress(content, quality=3))
            else:
                content = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
                output.write(content)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
