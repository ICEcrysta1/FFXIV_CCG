"""raw JSON 到最终 compiled cache 的编译编排。"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import multiprocessing
from pathlib import Path

from common.torch_dependencies import import_torch
from common.policy.data.compiled_cache import (
    DEFAULT_CACHE_MAX_SHARDS,
    DEFAULT_CACHE_SHARD_SIZE,
    CompiledShardCache,
    build_cache_signature,
    cache_path_for_source,
)
from common.policy.data.normalizer import Normalizer
from common.policy.data.skill_vocab import SkillVocab

from .cache_load import RAW_CONVERSION_VERSION, _load_cache
from .cache_paths import RawTrainingPathGroup, select_training_raw_path_groups
from .cache_writer import write_compiled_cache_stream
from .constants import DEFAULT_DOWNTIME_GAP_SECONDS
from .history_bank import build_history_bank
from .raw_source import convert_raw_file
from .sample_builder import TrainingSampleBuilder
from .source_reader import TrainingSourceReader


logger = logging.getLogger(__name__)


def prepare_training_caches(
    data_dir: Path,
    *,
    max_files: int | None = None,
    job_tag: str,
    int_dtype,
    float_dtype,
    cache_dir: Path | None,
    shard_size: int = DEFAULT_CACHE_SHARD_SIZE,
    max_workers: int = 1,
    max_shards: int = DEFAULT_CACHE_MAX_SHARDS,
    downtime_gap_seconds: float = DEFAULT_DOWNTIME_GAP_SECONDS,
) -> list[Path]:
    """选择 raw JSON、编译 cache，并从同副本候选补齐失败文件。"""
    groups = select_training_raw_path_groups(data_dir, max_files)
    if not groups:
        return []
    normalizer = Normalizer()
    normalizer.ensure_job_resources(job_tag)
    return _compile_training_path_groups(
        groups,
        job_tag=job_tag,
        downtime_gap_seconds=downtime_gap_seconds,
        normalizer=normalizer,
        int_dtype=int_dtype,
        float_dtype=float_dtype,
        cache_dir=cache_dir,
        shard_size=shard_size,
        max_workers=max_workers,
        max_shards=max_shards,
    )


def _compile_training_path_groups(
    groups: tuple[RawTrainingPathGroup, ...],
    *,
    job_tag: str,
    downtime_gap_seconds: float,
    normalizer: Normalizer,
    int_dtype,
    float_dtype,
    cache_dir: Path | None,
    shard_size: int,
    max_workers: int,
    max_shards: int,
) -> list[Path]:
    """逐轮编译副本候选，并从同副本后备文件补齐有效文件配额。"""
    states = [
        {
            "group": group,
            "pending": list(group.candidates),
            "valid": set(),
        }
        for group in groups
    ]
    group_by_path = {
        path.resolve(): state
        for state in states
        for path in state["group"].candidates
    }

    while True:
        round_paths: list[Path] = []
        for state in states:
            group = state["group"]
            needed = group.target_count - len(state["valid"])
            while needed > 0 and state["pending"]:
                candidate = state["pending"].pop(0)
                if candidate.resolve() not in state["valid"]:
                    round_paths.append(candidate)
                    needed -= 1

        if not round_paths:
            break

        compiled_paths = precompile_raw_training_caches(
            round_paths,
            job_tag=job_tag,
            downtime_gap_seconds=downtime_gap_seconds,
            normalizer=normalizer,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            cache_dir=cache_dir,
            shard_size=shard_size,
            max_workers=max_workers,
            max_shards=max_shards,
        )
        for compiled_path in compiled_paths:
            state = group_by_path.get(Path(compiled_path).resolve())
            if state is not None:
                state["valid"].add(Path(compiled_path).resolve())

    valid_paths: list[Path] = []
    shortages: list[str] = []
    for state in states:
        group = state["group"]
        group_valid_paths = [
            path
            for path in group.candidates
            if path.resolve() in state["valid"]
        ][: group.target_count]
        valid_paths.extend(group_valid_paths)
        if len(group_valid_paths) < group.target_count:
            shortage = group.target_count - len(group_valid_paths)
            shortages.append(
                f"{group.directory_name}: required={group.target_count} "
                f"valid={len(group_valid_paths)} missing={shortage}"
            )
            logger.error(
                "副本有效 raw JSON 不足，无法补齐配额: %s",
                shortages[-1],
            )

    if shortages:
        raise ValueError(
            "raw JSON training file quotas could not be filled: "
            + "; ".join(shortages)
        )
    return valid_paths


def precompile_raw_training_caches(
    raw_paths: list[Path],
    *,
    job_tag: str,
    source: int | None = None,
    encounter: str | None = None,
    downtime_gap_seconds: float = DEFAULT_DOWNTIME_GAP_SECONDS,
    normalizer: Normalizer,
    int_dtype,
    float_dtype,
    cache_dir: Path | None,
    shard_size: int = DEFAULT_CACHE_SHARD_SIZE,
    max_workers: int = 1,
    max_shards: int = DEFAULT_CACHE_MAX_SHARDS,
) -> list[Path]:
    """读取 raw JSON，并把结果直接写入最终 compiled cache。"""
    if cache_dir is None or not raw_paths:
        return []
    if max_workers < 1:
        raise ValueError("compiled cache workers must be >= 1")

    normalizer.ensure_job_resources(job_tag)
    cache_dir = Path(cache_dir).resolve()
    shard_cache = CompiledShardCache(max_shards)
    valid_paths: list[Path] = []
    missing: list[Path] = []
    for source_path in raw_paths:
        source_path = Path(source_path)
        cached = _load_cache(
            source_path,
            cache_dir=cache_dir,
            normalizer=normalizer,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            shard_size=shard_size,
            shard_cache=shard_cache,
        )
        if cached is not None and cached.num_samples > 0:
            valid_paths.append(source_path)
        else:
            missing.append(source_path)

    if not missing:
        return sorted(set(valid_paths), key=lambda path: str(path).casefold())

    worker_count = min(int(max_workers), len(missing))
    logger.info("raw JSON 缓存编译: %d 个源文件, %d 个 worker", len(missing), worker_count)
    tasks = [
        (
            source_path,
            job_tag,
            source,
            encounter,
            float(downtime_gap_seconds),
            cache_dir,
            _dtype_name(int_dtype),
            _dtype_name(float_dtype),
            normalizer.normalization_contract,
            int(shard_size),
        )
        for source_path in missing
    ]

    results: list[tuple[str, int, int]] = []
    failed_count = 0
    if worker_count == 1:
        for task in tasks:
            try:
                results.append(_compile_raw_source_worker(task))
            except Exception as exc:
                failed_count += 1
                _log_compile_failure(task[0], exc)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
            future_sources = {
                executor.submit(_compile_raw_source_worker, task): Path(task[0])
                for task in tasks
            }
            for future in as_completed(future_sources):
                try:
                    results.append(future.result())
                except Exception as exc:
                    failed_count += 1
                    _log_compile_failure(future_sources[future], exc)

    if failed_count:
        logger.error(
            "本轮 raw JSON 缓存编译跳过 %d 个失败文件，继续处理其余文件",
            failed_count,
        )

    for source_path, sample_count, shard_count in sorted(results):
        path = Path(source_path)
        if sample_count <= 0:
            logger.info("raw JSON 无可识别动作，跳过 %s", path.name)
            continue
        valid_paths.append(path)
        logger.info(
            "训练缓存已编译: source=%s samples=%d shards=%d",
            path.name,
            sample_count,
            shard_count,
        )
    return sorted(set(valid_paths), key=lambda path: str(path).casefold())


def _compile_raw_source_worker(task) -> tuple[str, int, int]:
    (
        source_path,
        job_tag,
        source,
        encounter,
        downtime_gap_seconds,
        cache_dir,
        int_dtype_name,
        float_dtype_name,
        normalizer_contract,
        shard_size,
    ) = task
    source_path = Path(source_path)
    int_dtype = _dtype_from_name(int_dtype_name)
    float_dtype = _dtype_from_name(float_dtype_name)
    training_payload, _ignored_skill_counts = convert_raw_file(
        source_path,
        job_tag=str(job_tag),
        source=None if source is None else int(source),
        encounter=None if encounter is None else str(encounter),
        downtime_gap=float(downtime_gap_seconds),
    )
    if training_payload is None:
        return str(source_path), 0, 0

    reader = TrainingSourceReader(training_payload)
    worker_normalizer = Normalizer.from_contract(normalizer_contract)
    worker_normalizer.ensure_job_resources(str(job_tag))
    worker_normalizer.register_schema(reader.schema)
    skill_vocab = SkillVocab.build_from_job_tag(reader.job_tag)
    torch = import_torch()
    history_bank = build_history_bank(
        reader,
        torch=torch,
        normalizer=worker_normalizer,
        skill_vocab=skill_vocab,
        skill_feature_names=reader.skill_feature_names,
        int_dtype=int_dtype,
        float_dtype=float_dtype,
    )
    sample_builder = TrainingSampleBuilder(
        torch=torch,
        normalizer=worker_normalizer,
        skill_vocab=skill_vocab,
        skill_feature_names=reader.skill_feature_names,
        int_dtype=int_dtype,
        float_dtype=float_dtype,
        num_candidates=reader.num_candidates,
        history_bank=history_bank,
    )

    def sample_batches():
        for start in range(0, reader.num_samples, shard_size):
            yield [
                sample_builder.build(reader, sample_idx)
                for sample_idx in range(start, min(start + shard_size, reader.num_samples))
            ]

    signature = build_cache_signature(
        source_path,
        int_dtype=int_dtype,
        float_dtype=float_dtype,
        normalizer=worker_normalizer,
        shard_size=shard_size,
        conversion_version=RAW_CONVERSION_VERSION,
    )
    write_compiled_cache_stream(
        cache_path_for_source(Path(cache_dir), source_path),
        source_path,
        signature=signature,
        reader=reader,
        sample_batches=sample_batches(),
        num_samples=reader.num_samples,
        vocab_signature=tuple(skill_vocab),
        shard_size=shard_size,
        history_bank=history_bank,
    )
    shard_count = (reader.num_samples + shard_size - 1) // shard_size
    return str(source_path), reader.num_samples, shard_count


def _log_compile_failure(source_path, exc: Exception) -> None:
    source_path = Path(source_path)
    logger.error(
        "raw JSON 缓存编译失败，跳过并继续: source=%s error=%s",
        source_path,
        exc,
    )
    logger.debug(
        "raw JSON 缓存编译异常详情: source=%s",
        source_path,
        exc_info=(type(exc), exc, exc.__traceback__),
    )


def _dtype_name(dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_from_name(name: str):
    torch = import_torch()
    try:
        return getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"unsupported torch dtype in cache worker: {name!r}") from exc
