"""ONNX 部署产物的确定性写入、归一化和文件校验工具。"""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    """流式计算部署产物的 SHA-256。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_payload_sha256(payload: Mapping[str, object]) -> str:
    """按项目确定性 JSON 格式计算载荷哈希。"""
    return hashlib.sha256(_serialize_json(payload).encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """以 UTF-8/LF 原子写入 JSON，避免发布状态留下半文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = _serialize_json(payload)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(serialized)
        stream.flush()
    temporary.replace(path)


def _serialize_json(payload: Mapping[str, object]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def write_deterministic_npz(
    path: Path,
    arrays: Mapping[str, np.ndarray],
) -> None:
    """写固定时间戳和顺序的 NPZ，保证相同输入产生相同哈希。"""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, arrays[name], allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())


def normalize_torch_reports(package_dir: Path) -> None:
    """把 PyTorch 可能生成的多份导出报告归一化为单一审计文件。"""
    reports = sorted(Path(package_dir).glob("onnx_export_*.md"))
    if not reports:
        raise RuntimeError("PyTorch export did not produce a Markdown report")
    target = Path(package_dir) / "torch_export_report.md"
    if len(reports) == 1:
        reports[0].replace(target)
        return
    sections = ["# PyTorch ONNX export reports", ""]
    for report in reports:
        sections.extend(
            [
                f"## {report.name}",
                "",
                report.read_text(encoding="utf-8"),
                "",
            ]
        )
    target.write_text("\n".join(sections), encoding="utf-8", newline="\n")
    for report in reports:
        report.unlink()
