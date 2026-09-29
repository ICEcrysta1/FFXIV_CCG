"""只读选择已编译且仍匹配源 JSON 的训练文件。"""

from __future__ import annotations

from pathlib import Path

from .compiled_cache import (
    DEFAULT_CACHE_MAX_SHARDS,
    DEFAULT_CACHE_SHARD_SIZE,
    DEFAULT_CONVERSION_VERSION,
    CompiledShardCache,
    build_cache_signature,
    load_compiled_cache_for_source,
)
from .normalizer import Normalizer
from .source_selection import select_training_raw_path_groups


def select_prepared_training_sources(
    data_dir: Path,
    *,
    max_files: int | None,
    job_tag: str,
    int_dtype,
    float_dtype,
    cache_dir: Path,
    shard_size: int = DEFAULT_CACHE_SHARD_SIZE,
    max_shards: int = DEFAULT_CACHE_MAX_SHARDS,
) -> list[Path]:
    """按转换时的副本配额和补位顺序选源，但绝不编译或写入缓存。"""
    groups = select_training_raw_path_groups(data_dir, max_files)
    if not groups:
        raise FileNotFoundError(f"没有训练 JSON：{data_dir}")
    normalizer = Normalizer()
    normalizer.ensure_job_resources(job_tag)
    shard_cache = CompiledShardCache(max_shards)
    selected: list[Path] = []
    missing: list[str] = []
    for group in groups:
        accepted = 0
        for source in group.candidates:
            if accepted >= group.target_count:
                break
            signature = build_cache_signature(
                source,
                int_dtype=int_dtype,
                float_dtype=float_dtype,
                normalizer=normalizer,
                shard_size=shard_size,
                conversion_version=DEFAULT_CONVERSION_VERSION,
            )
            cached = load_compiled_cache_for_source(
                cache_dir, source, signature=signature, shard_cache=shard_cache,
            )
            if cached is not None and cached.num_samples > 0 and cached.job_tag == job_tag:
                selected.append(source)
                accepted += 1
        if accepted < group.target_count:
            missing.append(f"{group.directory_name}: {accepted}/{group.target_count}")
    if missing:
        raise FileNotFoundError(
            "训练缓存缺失或已过期（" + ", ".join(missing)
            + "）；请先运行训练文件转换：.\\ffxiv_ccg.ps1 -Action convert"
        )
    return selected
