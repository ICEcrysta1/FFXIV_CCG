"""按训练缓存 shard 组织 batch 的采样器。"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler


class ShardBatchSampler(Sampler[list[int]]):
    """让一个 batch 尽量只访问一个已编译缓存 shard。"""

    def __init__(
        self,
        shard_indices: Sequence[Sequence[int]],
        *,
        batch_size: int,
        seed: int = 0,
        shuffle: bool = False,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not shard_indices:
            raise ValueError("shard_indices must not be empty")
        self._shard_indices = tuple(tuple(indices) for indices in shard_indices)
        if any(not indices for indices in self._shard_indices):
            raise ValueError("shard_indices must not contain empty shards")
        self._batch_size = int(batch_size)
        self._seed = int(seed)
        self._shuffle = bool(shuffle)
        self._iteration = 0

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self._seed + self._iteration)
        self._iteration += 1
        shard_order = list(range(len(self._shard_indices)))
        if self._shuffle:
            rng.shuffle(shard_order)

        for shard_index in shard_order:
            indices = list(self._shard_indices[shard_index])
            if self._shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self._batch_size):
                yield indices[start : start + self._batch_size]

    def __len__(self) -> int:
        return sum(
            math.ceil(len(indices) / self._batch_size)
            for indices in self._shard_indices
        )


class WeightedShardBatchSampler(ShardBatchSampler):
    """在保持 batch 不跨 shard 的前提下，按样本权重重复稀有样本。"""

    def __init__(
        self,
        shard_indices: Sequence[Sequence[int]],
        sample_weights: Sequence[int],
        *,
        batch_size: int,
        seed: int = 0,
        shuffle: bool = False,
    ):
        if any(weight < 1 for weight in sample_weights):
            raise ValueError("sample_weights must be >= 1")
        max_index = max(index for shard in shard_indices for index in shard)
        if len(sample_weights) <= max_index:
            raise ValueError("sample_weights must cover every shard sample index")
        self._weighted_shard_indices = tuple(
            tuple(index for index in shard for _ in range(int(sample_weights[index])))
            for shard in shard_indices
        )
        if any(not indices for indices in self._weighted_shard_indices):
            raise ValueError("weighted shard indices must not contain empty shards")
        super().__init__(
            self._weighted_shard_indices,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
        )
