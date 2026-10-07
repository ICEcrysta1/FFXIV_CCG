"""预训练数据、训练循环与 checkpoint 运行时组合入口。"""

from __future__ import annotations

from pathlib import Path

import torch
from common.policy.config import PROJECT_ROOT, resolve_policy_cache_dir
from common.policy.model import CausalPolicyModel
from common.project_config import resolve_registered_job_tags


from ..config import RunConfig
from .checkpoint import (
    _best_metric_key,
    _checkpoint_metrics,
    _epoch_checkpoint_name,
    _load_resume_checkpoint,
    _restore_best_state,
    _restore_rng_state,
    _restore_scheduler_state,
    _save_checkpoint,
    _top1_val_ppg_average,
    _validate_resume_checkpoint,
    collect_checkpoint_candidates,
    read_checkpoint_epoch,
)
from .dataloaders import (
    _build_batch_sampler,
    _dataloader_options,
    build_dataloaders,
)
from .training_loop import (
    _debug_stage,
    _lr_lambda,
    _seed_everything,
    run_training as _run_training,
    train_epoch,
    validate,
)


def run_training(
    config: RunConfig,
    *,
    raw_paths: list[Path],
    validation_paths: list[Path],
    output_dir: Path | None = None,
    max_epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    device_name: str = "cuda",
    validation_metrics_callback=None,
    resume_path: Path | None = None,
    force_resume_data_mismatch: bool = False,
) -> dict[str, object]:
    """组合预训练数据、训练循环和 checkpoint 组件。"""
    return _run_training(
        config,
        raw_paths=raw_paths,
        validation_paths=validation_paths,
        output_dir=output_dir,
        max_epochs=max_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device_name=device_name,
        validation_metrics_callback=validation_metrics_callback,
        resume_path=resume_path,
        force_resume_data_mismatch=force_resume_data_mismatch,
        _build_dataloaders=build_dataloaders,
        _train_epoch=train_epoch,
        _validate=validate,
        _save_checkpoint=_save_checkpoint,
        _resolve_cache_dir=resolve_policy_cache_dir,
        _registered_job_tags=lambda: resolve_registered_job_tags(PROJECT_ROOT),
        _model_class=CausalPolicyModel,
    )


__all__ = [
    "build_dataloaders",
    "collect_checkpoint_candidates",
    "read_checkpoint_epoch",
    "run_training",
    "train_epoch",
    "validate",
]
