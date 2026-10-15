"""GRPO checkpoint 保存、优化器快照与贪心保护回滚。"""

from __future__ import annotations

import random
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import torch

from common.policy.data import DataSpec
from common.pytorch.serialization import safe_torch_load, stringify_paths

from .config import GrpoConfig, GrpoRunConfig


def _save_grpo_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scheduler,
    iteration: int,
    config: GrpoRunConfig,
    grpo: GrpoConfig,
    data_spec: DataSpec,
    input_contract,
    precision: str,
    metrics: Mapping[str, float],
) -> None:
    """保存可被现有 replay/ONNX 流程读取的自描述 GRPO checkpoint。"""
    model_variant = getattr(config, "model_variant", None)
    if not isinstance(model_variant, str) or not model_variant.strip():
        raise ValueError("GRPO checkpoint model_variant must be configured before saving")
    model_variant = model_variant.strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(iteration),
        "grpo_iteration": int(iteration),
        "grpo_checkpoint": True,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "model_config": asdict(config.model),
        "data_spec": asdict(data_spec),
        "job_tag": data_spec.job_tag,
        "model_variant": model_variant,
        "input_contract": input_contract.to_dict(),
        "training_precision": precision,
        "run_config": stringify_paths(asdict(config)),
        "grpo_config": asdict(grpo),
        "metrics": dict(metrics),
        "rng_state": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        },
    }
    torch.save(payload, path)


def _snapshot_optimizer_state(optimizer) -> dict:
    """首轮回滚状态放在 CPU，包含主权重时也不额外常驻一份 GPU 副本。"""
    def copy_to_cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().to(device="cpu", copy=True)
        if isinstance(value, Mapping):
            return {key: copy_to_cpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [copy_to_cpu(item) for item in value]
        if isinstance(value, tuple):
            return tuple(copy_to_cpu(item) for item in value)
        return deepcopy(value)

    return copy_to_cpu(optimizer.state_dict())


def _restore_grpo_rollback_state(
    *,
    checkpoint_path: Path | None,
    initial_model_state: Mapping[str, object],
    initial_optimizer_state: Mapping[str, object] | None,
    initial_scheduler_state: Mapping[str, object] | None,
    model,
    optimizer,
    scheduler,
) -> None:
    """从上一轮磁盘 checkpoint 恢复回滚状态，避免常驻一份 GPU 权重副本。"""
    if checkpoint_path is None:
        model.load_state_dict(initial_model_state, strict=True)
        if initial_optimizer_state is not None:
            optimizer.load_state_dict(initial_optimizer_state)
        if scheduler is not None and initial_scheduler_state is not None:
            scheduler.load_state_dict(initial_scheduler_state)
        return

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"GRPO rollback checkpoint not found: {checkpoint_path}"
        )
    payload = safe_torch_load(checkpoint_path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"GRPO rollback checkpoint must be a mapping: {checkpoint_path}")
    model_state = payload.get("model_state_dict")
    optimizer_state = payload.get("optimizer_state_dict")
    if not isinstance(model_state, Mapping) or not isinstance(optimizer_state, Mapping):
        raise ValueError(
            "GRPO rollback checkpoint is missing model/optimizer state: "
            f"{checkpoint_path}"
        )
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    scheduler_state = payload.get("scheduler_state_dict")
    if scheduler is not None and isinstance(scheduler_state, Mapping):
        scheduler.load_state_dict(scheduler_state)
