"""把内存训练样本写成最终 compiled cache。"""

from __future__ import annotations

from collections.abc import Iterable
import os
from pathlib import Path

from common.torch_dependencies import import_torch
from common.policy.data.compiled_cache import (
    CACHE_FORMAT,
    CACHE_PICKLE_PROTOCOL,
    CompiledCacheReader,
    CompiledShardCache,
)


def write_compiled_cache_stream(
    cache_path: Path,
    source_path: Path,
    *,
    signature: dict[str, object],
    reader,
    sample_batches: Iterable[list[dict[str, object]]],
    num_samples: int,
    vocab_signature: tuple[tuple[int, int], ...],
    shard_size: int,
    history_bank: dict[str, object],
    shard_cache: CompiledShardCache | None = None,
) -> CompiledCacheReader:
    """流式写入最终 compiled cache，避免保留整场样本列表。"""
    if shard_size < 1:
        raise ValueError("compiled cache shard_size must be >= 1")
    if num_samples < 0:
        raise ValueError("compiled cache num_samples must be >= 0")

    torch = import_torch()
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    shard_files: list[str] = []
    written_samples = 0

    for shard_index, shard_samples in enumerate(sample_batches):
        if not shard_samples:
            raise ValueError("compiled cache sample batches must not be empty")
        if len(shard_samples) > shard_size:
            raise ValueError("compiled cache sample batch exceeds shard_size")
        shard_path = cache_path.with_name(
            f"{cache_path.stem}.shard-{shard_index:05d}.pt"
        )
        temporary_shard_path = shard_path.with_name(
            f"{shard_path.name}.tmp.{os.getpid()}"
        )
        try:
            torch.save(
                {
                    "cache_format": CACHE_FORMAT,
                    "shard_index": shard_index,
                    "samples": shard_samples,
                },
                temporary_shard_path,
                pickle_protocol=CACHE_PICKLE_PROTOCOL,
            )
            temporary_shard_path.replace(shard_path)
        finally:
            if temporary_shard_path.exists():
                temporary_shard_path.unlink()
        shard_files.append(shard_path.name)
        written_samples += len(shard_samples)

    if written_samples != num_samples:
        raise ValueError(
            f"compiled cache sample count mismatch: wrote {written_samples}, expected {num_samples}"
        )

    payload = {
        "cache_format": CACHE_FORMAT,
        "source_path": str(Path(source_path).resolve()),
        "cache_signature": signature,
        "schema": reader.schema,
        "job_tag": reader.job_tag,
        "fight_id": reader.fight_id,
        "num_samples": written_samples,
        "num_candidates": reader.num_candidates,
        "skill_feature_names": reader.skill_feature_names,
        "candidate_action_keys": tuple(reader.candidate_action_keys(0)),
        "vocab_signature": vocab_signature,
        "shard_size": int(shard_size),
        "history_bank_id": str(cache_path.resolve()),
        "shard_files": shard_files,
        # 历史 bank 按 source 只保存一份；样本自身只保留 end/length 引用。
        "history_bank": history_bank,
    }
    temporary_path = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary_path, pickle_protocol=CACHE_PICKLE_PROTOCOL)
        temporary_path.replace(cache_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return CompiledCacheReader(
        payload,
        cache_path=cache_path,
        shard_cache=shard_cache if shard_cache is not None else CompiledShardCache(),
    )
