"""Transformer 编码 trace，供模型分析和调试共用。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModelTrace:
    """包含每层 hidden、最终 hidden 和未平均 attention 的统一分析结果。"""

    encoded: dict[str, torch.Tensor]
    layer_hidden: tuple[torch.Tensor, ...]
    hidden: torch.Tensor
    attentions: tuple[torch.Tensor, ...]


@torch.no_grad()
def trace_encoder(encoder, encoded: dict[str, torch.Tensor]) -> ModelTrace:
    """运行因果编码器并保留每层 hidden 与完整 attention 矩阵。"""
    from ..attention.attention_masks import build_causal_attention_mask
    from .causal_encoder import run_causal_encoder

    hidden, layer_hidden, attentions = run_causal_encoder(
        encoder,
        encoded,
        collect_attention=True,
    )
    traced_encoded = dict(encoded)
    traced_encoded["attention_mask"] = build_causal_attention_mask(
        token_count=hidden.shape[1],
        device=hidden.device,
    )
    return ModelTrace(
        encoded=traced_encoded,
        layer_hidden=layer_hidden,
        hidden=hidden,
        attentions=attentions,
    )
