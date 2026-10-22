"""训练数据集、采样器和 DataLoader 构造。"""

from __future__ import annotations

import logging
from pathlib import Path

from torch.utils.data import DataLoader

from common.game.config import load_project_config
from common.game.skills import SkillBook
from common.policy.data import ActionSpace, DataSpec, Normalizer
from common.policy.data.contracts.policy_actions import load_policy_actions
from training.data import (
    ShardBatchSampler,
    TrainingCollator,
    TrainingDataset,
    WeightedShardBatchSampler,
    build_sample_weights,
    load_sequence_oversampler,
)

from ..config import RunConfig

logger = logging.getLogger(__name__)


def build_dataloaders(
    raw_paths: list[Path],
    config: RunConfig,
    *,
    validation_paths: list[Path],
    int_dtype,
    float_dtype,
    cache_dir: Path | None = None,
) -> tuple[DataLoader, DataLoader, TrainingDataset, TrainingDataset]:
    """分别读取训练与 VAL 的 compiled cache，保持两套文件独立。"""
    if not raw_paths:
        raise ValueError("no raw JSON files found")
    if not validation_paths:
        raise ValueError("no VAL compiled caches found")

    normalizer = Normalizer()
    if config.job_tag is not None:
        normalizer.configure_job_resources(config.job_tag)
    train_paths = [Path(path) for path in raw_paths]
    val_paths = [Path(path) for path in validation_paths]
    if set(train_paths) & set(val_paths):
        raise ValueError("training and VAL sources must not overlap")

    train_dataset = _build_dataset(train_paths, config, normalizer, int_dtype, float_dtype, cache_dir)
    val_dataset = _build_dataset(val_paths, config, normalizer, int_dtype, float_dtype, cache_dir)
    DataSpec.from_dataset(train_dataset).assert_compatible_with(DataSpec.from_dataset(val_dataset))

    if config.value_preference.enabled and config.value_preference.loss_weight > 0.0:
        skill_values = _load_skill_values(train_dataset.job_tag)
        missing_values = sorted(
            set(train_dataset.action_keys) - set(skill_values)
        )
        if missing_values:
            raise ValueError(
                "job YAML is missing value for output actions: "
                + ", ".join(missing_values)
            )

    train_collator = TrainingCollator(
        history_truncation_enabled=config.history_truncation_enabled,
        history_truncation_probability=config.history_truncation_probability,
        history_min_recent=config.history_min_recent,
        require_quality_percentile=config.action_quality_loss.enabled,
        int_dtype=int_dtype,
        float_dtype=float_dtype,
    )
    val_collator = TrainingCollator(
        require_quality_percentile=False,
        int_dtype=int_dtype,
        float_dtype=float_dtype,
    )
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
    loaders = []
    for dataset, collator, batch_sampler, shuffle in (
        (train_dataset, train_collator, train_batch_sampler, True),
        (val_dataset, val_collator, val_batch_sampler, False),
    ):
        batch_options = (
            {"batch_size": config.batch_size, "shuffle": shuffle}
            if batch_sampler is None
            else {"batch_sampler": batch_sampler}
        )
        loaders.append(DataLoader(
            dataset,
            collate_fn=collator,
            **batch_options,
            **loader_options,
        ))
    train_loader, val_loader = loaders
    logger.info("训练文件: %d，验证文件: %d", len(train_paths), len(val_paths))
    logger.info("训练样本: %d，验证样本: %d", len(train_dataset), len(val_dataset))
    return train_loader, val_loader, train_dataset, val_dataset


def _load_skill_values(job_tag: str) -> dict[str, float]:
    """读取职业与策略动作的价值定义，供数据加载前完整性校验。"""
    project_config = load_project_config(job_tag=job_tag)
    skill_book = SkillBook.from_project_config(project_config)
    values = {
        skill.key: float(skill.value)
        for skill in skill_book.enabled_skills()
    }
    for action in load_policy_actions():
        if action.key in values:
            raise ValueError(
                f"policy action key conflicts with skill key: {action.key}"
            )
        values[action.key] = float(action.value)
    return values


def _build_dataset(paths, config, normalizer, int_dtype, float_dtype, cache_dir):
    return TrainingDataset(
        paths,
        normalizer=normalizer,
        job_tag=config.job_tag,
        expected_action_space=ActionSpace.from_job_tag(config.job_tag),
        context_capacity=config.model.context_capacity,
        history_reset_keep=config.model.history_reset_keep,
        int_dtype=int_dtype,
        float_dtype=float_dtype,
        cache_dir=cache_dir,
        compiled_cache_shard_size=config.compiled_cache_shard_size,
        compiled_cache_max_shards=config.compiled_cache_max_shards,
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
