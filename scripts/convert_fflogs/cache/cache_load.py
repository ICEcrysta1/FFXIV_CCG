"""raw JSON 对应的 compiled cache 校验与读取。"""

from __future__ import annotations

from pathlib import Path

from common.policy.data.compiled_cache import (
    DEFAULT_CACHE_MAX_SHARDS,
    DEFAULT_CACHE_SHARD_SIZE,
    DEFAULT_CONVERSION_VERSION,
    CompiledShardCache,
    build_cache_signature,
    load_compiled_cache_for_source,
)
from common.policy.data.normalizer import Normalizer

RAW_CONVERSION_VERSION = DEFAULT_CONVERSION_VERSION


def load_raw_compiled_cache(
    source_path: Path,
    *,
    cache_dir: Path,
    normalizer: Normalizer,
    int_dtype,
    float_dtype,
    shard_size: int = DEFAULT_CACHE_SHARD_SIZE,
    max_shards: int = DEFAULT_CACHE_MAX_SHARDS,
    shard_cache: CompiledShardCache | None = None,
):
    """只读取 raw JSON 对应的最终 cache，不触发转换。"""
    return _load_cache(
        Path(source_path),
        cache_dir=Path(cache_dir),
        normalizer=normalizer,
        int_dtype=int_dtype,
        float_dtype=float_dtype,
        shard_size=shard_size,
        shard_cache=(
            CompiledShardCache(max_shards)
            if shard_cache is None
            else shard_cache
        ),
    )


def _load_cache(
    source_path: Path,
    *,
    cache_dir: Path,
    normalizer: Normalizer,
    int_dtype,
    float_dtype,
    shard_size: int,
    shard_cache: CompiledShardCache,
):
    signature = build_cache_signature(
        source_path,
        int_dtype=int_dtype,
        float_dtype=float_dtype,
        normalizer=normalizer,
        shard_size=shard_size,
        conversion_version=RAW_CONVERSION_VERSION,
    )
    return load_compiled_cache_for_source(
        Path(cache_dir),
        source_path,
        signature=signature,
        shard_cache=shard_cache,
    )
