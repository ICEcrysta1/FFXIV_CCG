"""训练数据集、采样器和 DataLoader 构造。"""

from __future__ import annotations

import logging
import random
from pathlib import Path

from torch.utils.data import DataLoader

from common.policy.data import (
    Normalizer,
    SkillVocab,
)
from training.data import (
    ShardBatchSampler,
    TrainingCollator,
    TrainingDataset,
    WeightedShardBatchSampler,
    build_sample_weights,
    load_sequence_oversampler,
)

from common.policy.data import DataSpec
from ..config import RunConfig
from .value_preference import load_skill_values


logger = logging.getLogger(__name__)


def build_dataloaders(
    raw_paths: list[Path],
    config: RunConfig,
    *,
    int_dtype,
    float_dtype,
    cache_dir: Path | None = None,
) -> tuple[DataLoader, DataLoader, TrainingDataset, TrainingDataset]:
    """读取已准备好的 compiled cache，再按战斗文件拆分训练集和验证集。"""
    if not raw_paths:
        raise ValueError("no raw JSON files found")

    normalizer = Normalizer()
    if config.job_tag is not None:
        normalizer.configure_job_resources(config.job_tag)
    valid_raw_paths = [Path(path) for path in raw_paths]

    shuffled = list(valid_raw_paths)
    random.Random(config.seed).shuffle(shuffled)
    if len(shuffled) == 1:
        train_paths = val_paths = shuffled
    else:
        split = min(len(shuffled) - 1, max(1, int(len(shuffled) * config.train_split)))
        train_paths = shuffled[:split]
        val_paths = shuffled[split:]

    train_dataset = _build_dataset(train_paths, config, normalizer, int_dtype, float_dtype, cache_dir)
    val_dataset = _build_dataset(val_paths, config, normalizer, int_dtype, float_dtype, cache_dir)
    DataSpec.from_dataset(train_dataset).assert_compatible_with(DataSpec.from_dataset(val_dataset))

    skill_values = None
    if config.value_preference.enabled and config.value_preference.loss_weight > 0.0:
        skill_values = load_skill_values(train_dataset.job_tag)
        missing_values = sorted(
            set(train_dataset.candidate_action_keys) - set(skill_values)
        )
        if missing_values:
            raise ValueError(
                "job YAML is missing value for candidate actions: "
                + ", ".join(missing_values)
            )

    train_collator = TrainingCollator(
        history_truncation_enabled=config.history_truncation_enabled,
        history_truncation_probability=config.history_truncation_probability,
        history_min_recent=config.history_min_recent,
        candidate_shuffle_enabled=config.candidate_shuffle_enabled,
        candidate_shuffle_probability=config.candidate_shuffle_probability,
        skill_values=skill_values,
    )
    val_collator = TrainingCollator(skill_values=skill_values)
    loader_options = _dataloader_options(config)
    sequence_oversampler = load_sequence_oversampler(config.config_path)
    sample_weights = build_sample_weights(train_dataset, sequence_oversampler)
    if sample_weights is not None:
        logger.info(
            "职业过采样: weighted_samples=%d expanded_samples=%d original_samples=%d",
            sum(weight > 1 for weight in sample_weights),
            sum(sample_weights),
            len(sample_weights),
        )
    train_batch_sampler = _build_batch_sampler(
        train_dataset,
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle=True,
        sample_weights=sample_weights,
    )
    val_batch_sampler = _build_batch_sampler(
        val_dataset,
        batch_size=config.batch_size,
        seed=config.seed,
        shuffle=False,
    )
    if train_batch_sampler is None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=train_collator,
            **loader_options,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=train_collator,
            **loader_options,
        )
    if val_batch_sampler is None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=val_collator,
            **loader_options,
        )
    else:
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_batch_sampler,
            collate_fn=val_collator,
            **loader_options,
        )
    logger.info("训练文件: %d，验证文件: %d", len(train_paths), len(val_paths))
    logger.info("训练样本: %d，验证样本: %d", len(train_dataset), len(val_dataset))
    return train_loader, val_loader, train_dataset, val_dataset


def _build_dataset(paths, config, normalizer, int_dtype, float_dtype, cache_dir):
    return TrainingDataset(
        paths,
        normalizer=normalizer,
        job_tag=config.job_tag,
        max_history=config.model.history_capacity,
        int_dtype=int_dtype,
        float_dtype=float_dtype,
        cache_dir=cache_dir,
        compiled_cache_shard_size=config.compiled_cache_shard_size,
        compiled_cache_max_shards=config.compiled_cache_max_shards,
        candidate_order_file=config.candidate_order_file,
    )


def _build_batch_sampler(
    dataset,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
    sample_weights: tuple[int, ...] | None = None,
):
    shard_indices = dataset.compiled_shard_indices()
    if not shard_indices:
        return None
    if sample_weights is not None:
        return WeightedShardBatchSampler(
            shard_indices,
            sample_weights,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
        )
    return ShardBatchSampler(
        shard_indices,
        batch_size=batch_size,
        seed=seed,
        shuffle=shuffle,
    )


def _dataloader_options(config: RunConfig) -> dict[str, object]:
    options: dict[str, object] = {
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
    }
    if config.num_workers > 0:
        options["prefetch_factor"] = config.prefetch_factor
        options["persistent_workers"] = config.persistent_workers
    return options
