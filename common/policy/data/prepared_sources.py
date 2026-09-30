"""只读选择已编译且仍匹配源 JSON 的训练文件。"""

from __future__ import annotations

import logging
import math
import pickle
from pathlib import Path

from common.torch_serialization import safe_torch_load

from .compiled_cache import (
    CACHE_FORMAT,
    DEFAULT_CACHE_MAX_SHARDS,
    DEFAULT_CACHE_SHARD_SIZE,
    DEFAULT_CONVERSION_VERSION,
    CompiledShardCache,
    build_cache_signature,
    cache_path_for_source,
    load_compiled_cache_for_source,
)
from .normalizer import Normalizer
from .schema import SceneWindowSchema, TrainingSchema
from .source_selection import (
    RawTrainingPathGroup,
    select_training_raw_path_groups,
    select_validation_raw_path_groups,
    validation_stage_root,
)

logger = logging.getLogger(__name__)


def cached_candidates_for_group(
    group: RawTrainingPathGroup,
    *,
    job_tag: str,
    normalizer: Normalizer,
    int_dtype,
    float_dtype,
    cache_dir: Path,
    shard_size: int,
    shard_cache: CompiledShardCache,
) -> list[Path]:
    """按候选顺序找足本组配额；只有签名和职业均匹配的 PT 才计数。"""
    cached_paths: list[Path] = []
    for source in group.candidates:
        if len(cached_paths) >= group.target_count:
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
            cached_paths.append(source)
    return cached_paths


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
    data_dir = Path(data_dir).resolve()
    stage_root = next(
        (parent for parent in (data_dir, *data_dir.parents)
         if parent.name in {"raw", "annotated"}),
        None,
    )
    if (stage_root is not None and data_dir != stage_root
            and data_dir.relative_to(stage_root).parts[0] == "VAL"):
        raise ValueError(f"训练输入不能指向验证目录：{data_dir}")
    groups = select_training_raw_path_groups(data_dir, max_files)
    if not groups:
        raise FileNotFoundError(f"没有训练 JSON：{data_dir}")
    normalizer = Normalizer()
    normalizer.ensure_job_resources(job_tag)
    shard_cache = CompiledShardCache(max_shards)
    selected: list[Path] = []
    missing: list[str] = []
    for group in groups:
        cached_paths = cached_candidates_for_group(
            group, job_tag=job_tag, normalizer=normalizer,
            int_dtype=int_dtype, float_dtype=float_dtype,
            cache_dir=cache_dir, shard_size=shard_size, shard_cache=shard_cache,
        )
        selected.extend(cached_paths)
        if len(cached_paths) < group.target_count:
            missing.append(f"{group.directory_name}: {len(cached_paths)}/{group.target_count}")
    if missing:
        raise FileNotFoundError(
            "训练缓存缺失或已过期（" + ", ".join(missing)
            + "）；请先运行训练文件转换：.\\ffxiv_ccg.ps1 -Action convert"
        )
    return selected


def select_prepared_validation_sources(
    cache_dir: Path,
    *,
    data_dir: Path,
    max_files: int,
    job_tag: str,
    int_dtype,
    float_dtype,
    shard_size: int = DEFAULT_CACHE_SHARD_SIZE,
    max_shards: int = DEFAULT_CACHE_MAX_SHARDS,
) -> list[Path]:
    """按各副本固定配额读取 `.cache/VAL/<副本>` 的有效 PT；不足即报错。"""
    groups = select_validation_raw_path_groups(data_dir, max_files)
    source_stage = validation_stage_root(data_dir)
    cache_dir = Path(cache_dir).resolve()
    validation_root = cache_dir / "VAL"
    if not validation_root.is_dir():
        raise FileNotFoundError(f"没有验证缓存目录：{validation_root}")

    normalizer = Normalizer()
    normalizer.ensure_job_resources(job_tag)
    shard_cache = CompiledShardCache(max_shards)
    selected: list[Path] = []
    shortages: list[str] = []
    for group in groups:
        encounter_dir = validation_root / group.directory_name
        candidates: list[Path] = []
        for manifest in sorted(encounter_dir.glob("*.compiled.pt")):
            try:
                payload = safe_torch_load(
                    manifest, mmap=True, map_location="meta",
                    safe_globals=(SceneWindowSchema, TrainingSchema),
                )
                if not isinstance(payload, dict) or payload.get("cache_format") != CACHE_FORMAT:
                    continue
                source_value = payload.get("source_path")
                if not isinstance(source_value, str):
                    continue
                source = Path(source_value)
                if (
                    source.parent.name != encounter_dir.name
                    or source.parent.parent.name != "VAL"
                    or source.parent.parent.parent.resolve() != source_stage
                    or cache_path_for_source(cache_dir, source).resolve() != manifest.resolve()
                    or not source.is_file()
                ):
                    continue
                candidates.append(source)
            except (OSError, RuntimeError, TypeError, ValueError, EOFError,
                    pickle.UnpicklingError) as error:
                logger.warning("跳过不可读取的验证缓存 %s: %s", manifest, error)
        allowed_sources = set(group.candidates)
        candidates = [source for source in candidates if source in allowed_sources]
        candidate_group = RawTrainingPathGroup(group.directory_name, len(candidates), tuple(candidates))
        valid = cached_candidates_for_group(
            candidate_group, job_tag=job_tag, normalizer=normalizer,
            int_dtype=int_dtype, float_dtype=float_dtype,
            cache_dir=cache_dir, shard_size=shard_size, shard_cache=shard_cache,
        )
        if len(valid) < group.target_count:
            shortages.append(f"{group.directory_name}: {len(valid)}/{group.target_count}")
            continue
        selected.extend(
            valid[math.floor((index + 0.5) * len(valid) / group.target_count)]
            for index in range(group.target_count)
        )
    if shortages:
        raise FileNotFoundError(
            f"验证副本缓存不足：{validation_root} " + ", ".join(shortages)
            + "；请先标注并转换对应副本的 VAL 日志"
        )
    logger.info(
        "验证缓存按副本选择：%s",
        ", ".join(f"{group.directory_name}={group.target_count}" for group in groups),
    )
    return selected
