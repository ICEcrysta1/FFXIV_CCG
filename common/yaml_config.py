"""跨顶层模块复用的 YAML mapping 加载工具。"""

from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml_mapping(path: Path, *, description: str = "YAML config") -> dict[str, object]:
    """按 UTF-8 加载 YAML，并验证顶层值为 mapping。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    with open(path, encoding="utf-8", newline="\n") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a mapping: {path}")
    return payload
