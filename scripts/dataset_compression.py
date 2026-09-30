"""将已有 raw/annotated 普通 JSON 一次性迁移到 Brotli 数据集格式。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.common.json_io import atomic_write_json, read_json


def _stage(path: Path) -> Path:
    for parent in path.resolve().parents:
        if parent.name in {"raw", "annotated"}:
            return parent
    raise ValueError(f"输入不在 raw/ 或 annotated/ 阶段目录下: {path}")


def _raw_peer(source: Path, stage: Path) -> dict:
    relative = source.relative_to(stage)
    plain = stage.parent / "raw" / relative
    compressed = plain.with_name(plain.name + ".br")
    candidates = [
        json.loads(plain.read_bytes()) if plain.is_file() else None,
        read_json(compressed) if compressed.is_file() else None,
    ]
    present = [candidate for candidate in candidates if candidate is not None]
    if not present or any(not isinstance(candidate, dict) or candidate != present[0] for candidate in present):
        raise ValueError(f"找不到一致的原始战斗文件: {source}")
    return present[0]


def migrate_file(source: Path, *, remove_original: bool = False) -> tuple[Path, bool]:
    """验证来源并原子写入压缩副本；已有目标仅在内容一致时复用。"""
    source = Path(source).resolve()
    if source.suffix.lower() != ".json" or not source.is_file():
        raise ValueError(f"迁移输入必须是普通 JSON 文件: {source}")
    stage = _stage(source)
    payload = json.loads(source.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError(f"战斗 JSON 顶层必须是对象: {source}")
    if stage.name == "annotated":
        raw = _raw_peer(source, stage)
        analysis = payload.get("analysis")
        if not isinstance(analysis, dict) or analysis.get("schema_version") != 2 or analysis.get("bridge_version") not in {3, 4}:
            raise ValueError(f"标注格式不可迁移: {source}")
        if {key: value for key, value in payload.items() if key != "analysis"} != raw:
            raise ValueError(f"标注内容与原始战斗不一致: {source}")
        identity = analysis.get("source")
        report_code = raw.get("report_code") if raw.get("report_code") is not None else raw.get("code")
        if not isinstance(identity, dict) or identity.get("fight_id") != raw.get("fight_id") or identity.get("report_code") != report_code:
            raise ValueError(f"标注身份与原始战斗不一致: {source}")
        identity.pop("sha256", None)
        analysis["bridge_version"] = 4

    target = source.with_name(source.name + ".br")
    if target.is_file():
        if read_json(target) != payload:
            raise ValueError(f"压缩目标已存在但内容不同: {target}")
        created = False
    else:
        atomic_write_json(target, payload)
        if read_json(target) != payload:
            raise RuntimeError(f"压缩文件校验失败: {target}")
        created = True
    if remove_original:
        source.unlink()
    return target, created


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="一次性将 raw/annotated 普通 JSON 迁移到 .json.br")
    parser.add_argument("inputs", nargs="+", type=Path, help="raw/annotated 文件或上层目录")
    parser.add_argument("--remove-original", action="store_true", help="逐份验证压缩副本后删除对应普通 JSON")
    args = parser.parse_args(argv)
    sources = set()
    for item in args.inputs:
        if item.is_file():
            sources.add(item.resolve())
        elif item.is_dir():
            sources.update(path.resolve() for path in item.rglob("*.json") if any(parent.name in {"raw", "annotated"} for parent in path.parents))
        else:
            parser.error(f"输入不存在: {item}")
    if not sources:
        parser.error("没有找到可迁移的普通 JSON")
    created = reused = 0
    for source in sorted(sources, key=lambda path: ("annotated" in path.parts, str(path).casefold())):
        target, did_create = migrate_file(source, remove_original=args.remove_original)
        created += did_create
        reused += not did_create
        print(f"{'新建' if did_create else '复用'}: {target}")
    print(f"完成: 新建={created} 复用={reused} 原文件{'已逐份删除' if args.remove_original else '保留'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
