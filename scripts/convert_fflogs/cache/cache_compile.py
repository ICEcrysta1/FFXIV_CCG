"""raw JSON 到最终 compiled cache 的编译编排。"""

from __future__ import annotations

import logging
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import replace
from itertools import islice
from pathlib import Path

from common.policy.data.compiled_cache import (
    DEFAULT_CACHE_MAX_SHARDS,
    DEFAULT_CACHE_SHARD_SIZE,
    CompiledShardCache,
    build_cache_signature,
    cache_path_for_source,
)
from common.policy.data.normalizer import Normalizer
from common.policy.data.prepared_sources import cached_candidates_for_group
from common.policy.data.skill_vocab import SkillVocab
from common.policy.data.source_selection import (
    RawTrainingPathGroup,
    select_training_raw_path_groups,
    select_validation_raw_path_groups,
)
from common.torch_dependencies import import_torch
from scripts.common.inprocess_backend import InProcessEngine

from ..config.constants import DEFAULT_DOWNTIME_GAP_SECONDS
from ..source.raw_source import convert_raw_file
from ..source.source_reader import TrainingSourceReader
from ..training.history_bank import build_history_bank
from ..training.quality_supervision import source_ranking
from ..training.sample_builder import TrainingSampleBuilder
from .cache_load import RAW_CONVERSION_VERSION, _load_cache
from .cache_writer import write_compiled_cache_stream

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


def prepare_validation_caches(
    data_dir: Path,
    *,
    max_files: int,
    job_tag: str,
    int_dtype,
    float_dtype,
    cache_dir: Path | None,
    shard_size: int = DEFAULT_CACHE_SHARD_SIZE,
    max_workers: int = 1,
    max_shards: int = DEFAULT_CACHE_MAX_SHARDS,
    downtime_gap_seconds: float = DEFAULT_DOWNTIME_GAP_SECONDS,
) -> list[Path]:
    """按副本固定配额编译 VAL；已有 PT 优先，同副本候选补位，缺额报错。"""
    groups = select_validation_raw_path_groups(data_dir, max_files)
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


def prepare_training_and_validation_caches(
    data_dir: Path,
    *,
    max_files: int | None,
    validation_files: int,
    job_tag: str,
    int_dtype,
    float_dtype,
    cache_dir: Path | None,
    shard_size: int = DEFAULT_CACHE_SHARD_SIZE,
    max_workers: int = 1,
    max_shards: int = DEFAULT_CACHE_MAX_SHARDS,
    downtime_gap_seconds: float = DEFAULT_DOWNTIME_GAP_SECONDS,
) -> tuple[list[Path], list[Path]]:
    """训练档位与 VAL 副本共同编译，所有组补位结束后统一检查缺额。"""
    training_groups = select_training_raw_path_groups(data_dir, max_files)
    try:
        validation_groups = select_validation_raw_path_groups(data_dir, validation_files)
    except FileNotFoundError:
        # 尚无 VAL 副本目录时，也要先完成训练档位的转换与缺额统计。
        validation_groups = ()
    # 只为缺额报告加前缀；候选文件和 VAL 原有的同副本补位范围不变。
    named_validation_groups = tuple(
        replace(group, directory_name=f"VAL/{group.directory_name}")
        for group in validation_groups
    )
    if not validation_groups:
        named_validation_groups = (RawTrainingPathGroup("VAL", validation_files, ()),)
    normalizer = Normalizer()
    normalizer.ensure_job_resources(job_tag)
    valid_paths = _compile_training_path_groups(
        training_groups + named_validation_groups,
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
    validation_candidates = {
        path.resolve()
        for group in validation_groups
        for path in group.candidates
    }
    training_paths = [path for path in valid_paths if path.resolve() not in validation_candidates]
    validation_paths = [path for path in valid_paths if path.resolve() in validation_candidates]
    return training_paths, validation_paths


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
    """先复用整组已有缓存，再逐轮编译缺额并从同副本候选补位。"""
    if cache_dir is None:
        raise ValueError("compiled cache directory is required for training selection")
    shard_cache = CompiledShardCache(max_shards)
    states = [
        {
            "group": group,
            "valid": {
                path.resolve() for path in cached_candidates_for_group(
                    group, job_tag=job_tag, normalizer=normalizer,
                    int_dtype=int_dtype, float_dtype=float_dtype,
                    cache_dir=cache_dir, shard_size=shard_size, shard_cache=shard_cache,
                )
            },
        }
        for group in groups
    ]
    for state in states:
        state["pending"] = [
            path for path in state["group"].candidates
            if path.resolve() not in state["valid"]
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
                "有效转换文件不足，无法补齐配额: %s",
                shortages[-1],
            )

    if shortages:
        raise ValueError(
            "JSON compiled cache quotas could not be filled: "
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
    engine: InProcessEngine | None = None,
) -> list[Path]:
    """读取 raw JSON，并把结果直接写入最终 compiled cache。"""
    if cache_dir is None or not raw_paths:
        return []
    if max_workers < 1:
        raise ValueError("compiled cache workers must be >= 1")
    if engine is not None and engine.job_tag != job_tag:
        raise ValueError("cache compilation job must match the shared engine")

    normalizer.ensure_job_resources(job_tag)
    cache_dir = Path(cache_dir).resolve()
    shard_cache = CompiledShardCache(max_shards)
    valid_paths: list[Path] = []
    missing: list[Path] = []
    for source_path in dict.fromkeys(Path(path).resolve() for path in raw_paths):
        cached = _load_cache(
            source_path,
            cache_dir=cache_dir,
            normalizer=normalizer,
            int_dtype=int_dtype,
            float_dtype=float_dtype,
            shard_size=shard_size,
            shard_cache=shard_cache,
        )
        if cached is not None and cached.num_samples > 0 and cached.job_tag == job_tag:
            valid_paths.append(source_path)
        else:
            missing.append(source_path)

    if not missing:
        return sorted(set(valid_paths), key=lambda path: str(path).casefold())

    # 在启动工作线程前建立公共目录，避免并发首次创建改变 Windows 路径解析结果。
    cache_dir.mkdir(parents=True, exist_ok=True)
    worker_count = min(int(max_workers), len(missing))
    if engine is not None:
        worker_count = min(worker_count, engine.capacity)
    logger.info("raw JSON 缓存编译: %d 个源文件, %d 个线程共享一个状态机引擎", len(missing), worker_count)
    normalizer_contract = normalizer.normalization_contract
    tasks = (
        (
            source_path,
            job_tag,
            source,
            encounter,
            float(downtime_gap_seconds),
            cache_dir,
            _dtype_name(int_dtype),
            _dtype_name(float_dtype),
            normalizer_contract,
            int(shard_size),
        )
        for source_path in missing
    )

    results: list[tuple[str, int, int]] = []
    failed_count = 0
    # 线程数与队列容量一致，在途任务也不超过该上限。完成一个才接收下一个，
    # 避免大量 Future 或异常 traceback 持有已失败文件的完整转换上下文。
    owner = InProcessEngine(job_tag, capacity=worker_count) if engine is None else nullcontext(engine)
    with owner as engine:
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="fflogs-convert") as executor:
            future_sources = {
                executor.submit(_compile_raw_source_worker, task, engine=engine): Path(task[0])
                for task in islice(tasks, worker_count)
            }
            while future_sources:
                completed, _ = wait(future_sources, return_when=FIRST_COMPLETED)
                for future in completed:
                    source_path = future_sources.pop(future)
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        failed_count += 1
                        _log_compile_failure(source_path, exc)
                    task = next(tasks, None)
                    if task is not None:
                        future_sources[executor.submit(
                            _compile_raw_source_worker, task, engine=engine,
                        )] = Path(task[0])

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


def _compile_raw_source_worker(task, *, engine: InProcessEngine) -> tuple[str, int, int]:
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
        engine=engine,
    )
    if training_payload is None:
        return str(source_path), 0, 0

    reader = TrainingSourceReader(training_payload)
    ranking = source_ranking(source_path, reader.ranking)
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
        num_actions=reader.num_actions,
        history_bank=history_bank,
        ranking=ranking,
        annotation_status=reader.annotation_status,
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
