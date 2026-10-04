"""分析与回放共用的只读参考场景选择。"""

import logging
import pickle
from pathlib import Path

from common.config import load_precision_config
from common.dataset_layout import PERCENTILE_BUCKETS, find_dataset_json_files
from common.policy.data import ActionSpace, Normalizer
from common.policy.data.compiled_cache import (
    CompiledShardCache,
    build_cache_signature,
    load_compiled_cache_for_source,
)


logger = logging.getLogger(__name__)


def find_prepared_scene_source(
    raw_root: Path,
    *,
    cache_dir: Path,
    job_tag: str,
    cache_shard_size: int,
    expected_action_space: ActionSpace,
    normalizer: Normalizer | None = None,
) -> Path:
    """优先从高百分位选择有效缓存对应的源文件，不触发编译。"""
    bucket_order = {bucket: index for index, bucket in enumerate(PERCENTILE_BUCKETS)}
    candidates = sorted(
        find_dataset_json_files(raw_root),
        key=lambda path: (bucket_order.get(path.parent.name, len(bucket_order)), str(path)),
    )
    if normalizer is None:
        normalizer = Normalizer()
    normalizer.ensure_job_resources(job_tag)
    precision = load_precision_config()
    shard_cache = CompiledShardCache(1)
    for source in candidates:
        signature = build_cache_signature(
            source, normalizer=normalizer,
            int_dtype=precision.resolve_int_dtype(),
            float_dtype=precision.resolve_float_dtype(),
            shard_size=cache_shard_size,
        )
        reader = load_compiled_cache_for_source(
            cache_dir, source, signature=signature, shard_cache=shard_cache,
            expected_action_space=expected_action_space,
        )
        if reader is None or reader.num_samples <= 0 or reader.job_tag != job_tag:
            continue
        try:
            # 分析可能读取整场，回放也可从任意样本开始，因此逐片试读全部样本。
            # LRU 只保留一片，避免校验期间把整场分片同时驻留内存。
            for start in range(0, reader.num_samples, reader.shard_size):
                samples = reader.samples(range(start, min(start + reader.shard_size, reader.num_samples)))
                if any(not isinstance(sample, dict) for sample in samples):
                    raise ValueError("compiled cache shard sample must be a mapping")
        except (OSError, RuntimeError, ValueError, TypeError, IndexError,
                EOFError, pickle.UnpicklingError) as error:
            logger.warning("跳过分片不可读取的参考场景 %s: %s", source, error)
            continue
        finally:
            shard_cache.clear()
        return source
    raise FileNotFoundError(
        f"没有可用的已编译参考场景：{raw_root}（缓存目录：{cache_dir}）；"
        "请先运行训练文件转换：.\\ffxiv_ccg.ps1 -Action convert，"
        "或显式指定场景 JSON"
    )
