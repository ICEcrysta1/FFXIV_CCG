"""跨模块复用的 PyTorch 可选依赖导入辅助。"""

from __future__ import annotations


def import_torch():
    """导入 torch，并统一缺失依赖时的错误信息。"""
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - 依赖缺失由调用方感知
        raise RuntimeError("依赖 torch 的训练/导出功能需要先安装 PyTorch") from exc
    return torch
