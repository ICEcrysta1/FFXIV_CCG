"""数据脚本共用的副本/百分位目录约定与路径映射，不负责读写数据。"""

import math
from pathlib import Path

PERCENTILE_BUCKETS = tuple(f"{lower:02d}-{lower + 10}" for lower in range(90, -1, -10))


def percentile_bucket(percentile: float) -> str:
    """区间左闭右开，最高档包含 100，最低档固定写作 00-10。"""
    if isinstance(percentile, bool) or not isinstance(percentile, (int, float)):
        raise TypeError(f"invalid historical percentile: {percentile!r}")
    if not math.isfinite(percentile) or not 0 <= percentile <= 100:
        raise ValueError(f"invalid historical percentile: {percentile!r}")
    lower = min(int(percentile // 10) * 10, 90)
    return f"{lower:02d}-{lower + 10}"


def percentile_directory(encounter_dir: str | Path, bucket: str) -> Path:
    """在调用方选定的副本目录下构造水平区间目录，不创建目录。"""
    if bucket not in PERCENTILE_BUCKETS:
        raise ValueError(f"invalid percentile bucket: {bucket!r}")
    return Path(encounter_dir) / bucket


def map_dataset_output_path(
    source_path: str | Path, *, source_root: str | Path, output_root: str | Path,
) -> Path:
    """替换数据阶段根目录，保留副本、水平区间和文件名，也兼容旧目录层级。

    例如 raw/FRU/00-10/a.json 映射为 annotated/FRU/00-10/a.json。
    不读取 JSON、不重新评分或分档，也不创建目录；转换产物可再调用 with_suffix。
    """
    source = Path(source_path).resolve()
    relative = source.relative_to(Path(source_root).resolve())
    if relative == Path("."):
        raise ValueError("source_path must be below source_root")
    root = Path(output_root).resolve()
    output = (root / relative).resolve()
    output.relative_to(root)
    return output


def find_dataset_json_files(directory: str | Path) -> list[Path]:
    """递归发现副本和区间下的 JSON，兼容尚未分档的旧数据目录。"""
    return sorted(Path(directory).rglob("*.json"))
