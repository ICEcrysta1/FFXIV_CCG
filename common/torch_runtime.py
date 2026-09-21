"""跨训练、回放和模型分析复用的 PyTorch 运行时工具。"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext

import torch


def move_batch(
    batch: Mapping[str, object],
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> dict[str, object]:
    """把 batch 中的 tensor 移到目标设备，保留非 tensor 元数据。"""
    return {
        key: value.to(device, non_blocking=non_blocking)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def model_dtype(precision: str) -> torch.dtype:
    """把项目精度配置转换为模型参数和 autocast 使用的 dtype。"""
    if precision == "float32":
        return torch.float32
    if precision == "float16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"unsupported precision: {precision!r}")


def autocast_context(device: torch.device, precision: str):
    """按统一精度配置创建前向 autocast 上下文。"""
    if precision == "float32":
        return nullcontext()
    if precision in {"float16", "bf16"} and device.type != "cuda":
        raise ValueError(f"precision {precision} requires a CUDA device")
    return torch.autocast(device_type=device.type, dtype=model_dtype(precision))
