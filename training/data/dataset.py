"""从最终 compiled cache 读取训练样本的数据集。"""

from __future__ import annotations

from bisect import bisect_right
import logging
from pathlib import Path

try:  # pragma: no cover - 缺依赖时由调用方感知
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover
    class Dataset:  # type: ignore[no-redef]
        pass

from common.policy.data.normalizer import Normalizer
from common.policy.data.skill_vocab import SkillVocab
from common.policy.data.candidate_order import candidate_permutation, load_candidate_order
from common.policy.data.compiled_cache import (
    DEFAULT_CACHE_MAX_SHARDS,
    DEFAULT_CACHE_SHARD_SIZE,
    CompiledCacheReader,
    CompiledShardCache,
    build_cache_signature,
    cache_path_for_source,
    load_compiled_cache,
)


class TrainingDataset(Dataset):
    """按 raw source 对应的最终 compiled cache 读取训练样本。"""

    def __init__(
        self,
        source_paths: list[Path],
        *,
        normalizer: Normalizer | None = None,
        job_tag: str | None = None,
        skill_vocab: SkillVocab | None = None,
        max_history: int | None = 128,
        int_dtype,
        float_dtype,
        cache_dir: Path | None = None,
        compiled_cache_shard_size: int = DEFAULT_CACHE_SHARD_SIZE,
        compiled_cache_max_shards: int = DEFAULT_CACHE_MAX_SHARDS,
        candidate_order_file: Path | None = None,
    ):
        if not source_paths:
            raise ValueError("TrainingDataset requires at least one raw source path")
        if cache_dir is None:
            raise ValueError("TrainingDataset requires a compiled cache directory")

        self._source_paths = tuple(Path(source_path) for source_path in source_paths)
        self._int_dtype = int_dtype
        self._float_dtype = float_dtype
        self._normalizer = normalizer
        if job_tag is not None and normalizer is not None:
            normalizer.ensure_job_resources(job_tag)
        if max_history is not None and max_history < 0:
            raise ValueError(f"max_history must be >= 0, got {max_history}")
        self._max_history = max_history
        if compiled_cache_shard_size < 1:
            raise ValueError("compiled_cache_shard_size must be >= 1")
        if compiled_cache_max_shards < 1:
            raise ValueError("compiled_cache_max_shards must be >= 1")
        self._compiled_cache_shard_size = int(compiled_cache_shard_size)
        self._compiled_cache_max_shards = int(compiled_cache_max_shards)
        self._cache_dir = None if cache_dir is None else Path(cache_dir).resolve()
        self._shard_cache = CompiledShardCache(self._compiled_cache_max_shards)

        self._readers: list[CompiledCacheReader] = []
        reference_reader = None
        vocab_signature: tuple[tuple[int, int], ...] = ()
        for source_path in self._source_paths:
            reader = self._load_reader(source_path)
            if reader is None:
                raise FileNotFoundError(
                    f"compiled cache not found or stale for raw source: {source_path}"
                )
            if reference_reader is None:
                reference_reader = reader
                self._schema = reader.schema
                self._job_tag = reader.job_tag
                self._num_candidates = reader.num_candidates
                self._skill_feature_names = reader.skill_feature_names
                self._cache_candidate_action_keys = tuple(reader.candidate_action_keys(0))
                self._candidate_action_keys = (
                    load_candidate_order(
                        candidate_order_file,
                        expected_action_keys=self._cache_candidate_action_keys,
                    )
                    if candidate_order_file is not None
                    else self._cache_candidate_action_keys
                )
                self._skill_vocab = skill_vocab or SkillVocab.build_from_job_tag(self._job_tag)
                vocab_signature = tuple(self._skill_vocab)

            self._assert_reader_compatible(reader)
            if reader.vocab_signature != vocab_signature:
                raise ValueError(f"compiled cache vocab mismatch: {source_path}")
            self._readers.append(reader)

        if reference_reader is None:
            raise RuntimeError("TrainingDataset could not initialize a compiled reader")

        self._sample_offsets = [0]
        for reader in self._readers:
            self._sample_offsets.append(self._sample_offsets[-1] + reader.num_samples)

    @property
    def schema(self):
        return self._schema

    @property
    def skill_vocab(self) -> SkillVocab:
        return self._skill_vocab

    @property
    def skill_feature_names(self) -> tuple[str, ...]:
        return self._skill_feature_names

    @property
    def job_tag(self) -> str:
        """训练数据中的职业标识。"""
        return self._job_tag

    @property
    def normalizer(self) -> Normalizer | None:
        """返回构建当前数据集时使用的归一化器。"""
        return self._normalizer

    @property
    def num_candidates(self) -> int:
        """训练数据定义的候选数量。"""
        return self._num_candidates

    @property
    def state_dim(self) -> int:
        """训练数据定义的状态向量维度。"""
        return self._schema.state_vector_dim()

    @property
    def scene_dim(self) -> int:
        """训练数据定义的 scene 向量维度。"""
        return self._schema.scene_feature_dim()

    @property
    def num_scene_types(self) -> int:
        """训练数据定义的 scene 类型数量。"""
        if not self._schema.scene_windows:
            return 0
        return max(window.scene_type_id for window in self._schema.scene_windows) + 1

    @property
    def candidate_action_keys(self) -> tuple[str, ...]:
        """训练数据定义的稳定候选顺序。"""
        return self._candidate_action_keys

    def __len__(self) -> int:
        return self._sample_offsets[-1]

    def compiled_shard_indices(self) -> tuple[tuple[int, ...], ...]:
        """返回按源缓存 shard 分组的全局样本索引。"""
        if not all(isinstance(reader, CompiledCacheReader) for reader in self._readers):
            return ()

        groups: list[tuple[int, ...]] = []
        for reader_index, reader in enumerate(self._readers):
            assert isinstance(reader, CompiledCacheReader)
            base_index = self._sample_offsets[reader_index]
            for shard_index in range((reader.num_samples + reader.shard_size - 1) // reader.shard_size):
                start = shard_index * reader.shard_size
                stop = min(start + reader.shard_size, reader.num_samples)
                groups.append(tuple(base_index + index for index in range(start, stop)))
        return tuple(groups)

    def iter_source_index_ranges(self):
        """按 raw source 依次返回连续全局样本索引，供序列级采样规则使用。"""
        for reader_index, reader in enumerate(self._readers):
            start = self._sample_offsets[reader_index]
            yield range(start, start + reader.num_samples)

    def iter_source_readers(self):
        """按 source 返回只读 compiled cache reader，供整场验证回放使用。"""
        return iter(tuple(self._readers))

    def __getstate__(self):
        """Windows worker 只传递 raw source 路径，避免 pickle 整份 cache payload。"""
        state = self.__dict__.copy()
        state["_readers"] = None
        state["_shard_cache"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._shard_cache = CompiledShardCache(self._compiled_cache_max_shards)
        self._readers = [self._load_required_reader(source_path) for source_path in self._source_paths]

    def _load_reader(self, source_path: Path) -> CompiledCacheReader | None:
        signature = build_cache_signature(
            source_path,
            int_dtype=self._int_dtype,
            float_dtype=self._float_dtype,
            normalizer=self._normalizer,
            shard_size=self._compiled_cache_shard_size,
        )
        return load_compiled_cache(
            cache_path_for_source(self._cache_dir, source_path),
            source_path,
            signature=signature,
            shard_cache=self._shard_cache,
        )

    def _load_required_reader(self, source_path: Path) -> CompiledCacheReader:
        reader = self._load_reader(source_path)
        if reader is None:
            raise FileNotFoundError(
                f"compiled cache not found or stale for raw source: {source_path}"
            )
        return reader

    def _assert_reader_compatible(self, reader) -> None:
        self._schema.assert_compatible_with(reader.schema)
        if reader.job_tag != self._job_tag:
            raise ValueError(f"compiled cache job_tag mismatch: {reader.job_tag!r} != {self._job_tag!r}")
        if reader.num_candidates != self._num_candidates:
            raise ValueError(
                f"compiled cache candidate count mismatch: {reader.num_candidates} != {self._num_candidates}"
            )
        if reader.skill_feature_names != self._skill_feature_names:
            raise ValueError("compiled cache skill numeric feature layout mismatch")
        if tuple(reader.candidate_action_keys(0)) != self._cache_candidate_action_keys:
            raise ValueError("compiled cache candidate action order mismatch")

    def __getitem__(self, index: int) -> dict[str, object]:
        reader_index, sample_idx = self._resolve_sample_ref(index)
        reader = self._readers[reader_index]
        return self._reorder_sample(reader.sample(sample_idx), reader)

    def __getitems__(self, indices: list[int]) -> list[dict[str, object]]:
        """批量读取 DataLoader 请求，避免同一 shard 重复执行单样本查找。"""
        if not indices:
            return []
        references = [self._resolve_sample_ref(index) for index in indices]
        grouped: dict[int, list[tuple[int, int]]] = {}
        for output_index, (reader_index, sample_idx) in enumerate(references):
            grouped.setdefault(reader_index, []).append((output_index, sample_idx))

        results: list[dict[str, object] | None] = [None] * len(indices)
        for reader_index, grouped_indices in grouped.items():
            reader = self._readers[reader_index]
            samples = reader.samples([sample_idx for _, sample_idx in grouped_indices])
            for (output_index, _), sample in zip(grouped_indices, samples):
                results[output_index] = self._reorder_sample(sample, reader)
        if any(sample is None for sample in results):
            raise RuntimeError("compiled batch fetch returned an incomplete sample batch")
        return [sample for sample in results if sample is not None]

    def _reorder_sample(
        self,
        sample: dict[str, object],
        reader: CompiledCacheReader | None = None,
    ) -> dict[str, object]:
        if reader is not None:
            sample = self._attach_history_bank(sample, reader)
            sample = self._apply_history_limit(sample)
        source_action_keys = tuple(str(key) for key in sample["candidate_action_keys"])
        if source_action_keys == self._candidate_action_keys:
            return sample
        permutation = candidate_permutation(source_action_keys, self._candidate_action_keys)
        permutation_tensor = sample["candidate_skill_ids"].new_tensor(permutation)
        reordered = dict(sample)
        for key in (
            "candidate_skill_ids",
            "candidate_skill_features",
            "candidate_values",
            "candidate_state_vectors",
            "candidate_state_null_mask",
            "candidate_legal_mask",
        ):
            if key in sample:
                reordered[key] = sample[key].index_select(0, permutation_tensor)
        reordered["candidate_action_keys"] = [
            sample["candidate_action_keys"][index] for index in permutation
        ]
        if "candidate_invalid_reasons" in sample:
            reordered["candidate_invalid_reasons"] = [
                sample["candidate_invalid_reasons"][index] for index in permutation
            ]
        source_label_key = source_action_keys[int(sample["label_index"])]
        reordered["label_index"] = self._candidate_action_keys.index(source_label_key)
        return reordered

    def _apply_history_limit(self, sample: dict[str, object]) -> dict[str, object]:
        """在读取时裁剪完整历史 bank 的引用窗口，不改变磁盘 cache。"""
        if self._max_history is None:
            return sample
        history_length = int(sample["history_length"])
        limited_length = min(history_length, self._max_history)
        if limited_length == history_length:
            return sample
        limited = dict(sample)
        limited["history_length"] = limited_length
        return limited

    def _attach_history_bank(
        self,
        sample: dict[str, object],
        reader: CompiledCacheReader,
    ) -> dict[str, object]:
        """把 source 级 bank 以只读引用挂到样本，避免复制每个历史窗口。"""
        if "history_end" not in sample:
            raise ValueError("compiled sample is missing compact history_end")
        attached = dict(sample)
        bank = reader.history_bank
        for key in (
            "skill_ids",
            "skill_features",
            "state_vectors",
            "state_null_mask",
            "skill_potencies",
            "cumulative_dot_potencies",
        ):
            attached[f"history_bank_{key}"] = bank[key]
        attached["history_bank_action_keys"] = bank["action_keys"]
        attached["history_bank_id"] = reader.history_bank_id
        return attached

    def _resolve_sample_ref(self, index: int) -> tuple[int, int]:
        if not 0 <= index < len(self):
            raise IndexError(f"training sample index out of range: {index}")
        reader_index = bisect_right(self._sample_offsets, index) - 1
        return reader_index, index - self._sample_offsets[reader_index]
