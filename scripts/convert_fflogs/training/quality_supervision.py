"""编译缓存时校验并保存战斗历史排名来源，不参与模型输入。"""

from __future__ import annotations

from pathlib import Path

from common.dataset_layout import PERCENTILE_BUCKETS, percentile_bucket


def source_ranking(source_path: Path, ranking: dict[str, object] | None) -> dict[str, object]:
    """保存真实排名百分位；旧文件可只由目录提供区间。"""
    directory_bucket = source_path.parent.name if source_path.parent.name in PERCENTILE_BUCKETS else None
    if ranking is None:
        return {"percentile": None, "percentile_bucket": directory_bucket}
    if not isinstance(ranking, dict):
        raise TypeError("ranking must be a mapping")
    value = ranking.get("percentile")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("ranking.percentile must be numeric")
    percentile = float(value)
    bucket = percentile_bucket(percentile)
    declared_bucket = ranking.get("percentile_bucket")
    if declared_bucket is not None and declared_bucket != bucket:
        raise ValueError("ranking percentile and bucket disagree")
    if directory_bucket is not None and directory_bucket != bucket:
        raise ValueError("ranking percentile and source directory disagree")
    return {"percentile": percentile, "percentile_bucket": bucket}
