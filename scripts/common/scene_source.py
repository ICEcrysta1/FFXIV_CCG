"""分析与回放共用的只读参考场景选择。"""

from pathlib import Path

from common.config import load_precision_config
from common.dataset_layout import PERCENTILE_BUCKETS, find_dataset_json_files
from common.policy.data import Normalizer
from common.policy.data.compiled_cache import CompiledShardCache
from common.policy.data.prepared_sources import cached_candidates_for_group
from common.policy.data.source_selection import RawTrainingPathGroup


def find_prepared_scene_source(
    raw_root: Path,
    *,
    cache_dir: Path,
    job_tag: str,
    cache_shard_size: int,
) -> Path:
    """优先从高百分位选择有效缓存对应的源文件，不触发编译。"""
    bucket_order = {bucket: index for index, bucket in enumerate(PERCENTILE_BUCKETS)}
    candidates = sorted(
        find_dataset_json_files(raw_root),
        key=lambda path: (bucket_order.get(path.parent.name, len(bucket_order)), str(path)),
    )
    normalizer = Normalizer()
    normalizer.ensure_job_resources(job_tag)
    precision = load_precision_config()
    selected = cached_candidates_for_group(
        RawTrainingPathGroup("参考场景", 1, tuple(candidates)),
        job_tag=job_tag,
        normalizer=normalizer,
        int_dtype=precision.resolve_int_dtype(),
        float_dtype=precision.resolve_float_dtype(),
        cache_dir=cache_dir,
        shard_size=cache_shard_size,
        shard_cache=CompiledShardCache(1),
    )
    if not selected:
        raise FileNotFoundError(
            f"没有可用的已编译参考场景：{raw_root}（缓存目录：{cache_dir}）；"
            "请先运行训练文件转换：.\\ffxiv_ccg.ps1 -Action convert，"
            "或显式指定场景 JSON"
        )
    return selected[0]
