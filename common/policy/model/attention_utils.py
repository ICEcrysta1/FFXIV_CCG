"""注意力投影、头维度整形与 Transformer 层收尾辅助。"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rotate_qk(
    rotary_position_encoding,
    query_heads: torch.Tensor,
    key_heads: torch.Tensor,
    query_position_ids: torch.Tensor,
    key_position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """在进入 SDPA 前统一旋转 Q/K。"""
    return rotary_position_encoding(
        query_heads,
        key_heads,
        query_position_ids,
        key_position_ids,
    )


def project_qkv(attention, values: torch.Tensor):
    """按 MHA 或 grouped-query attention 的权重投影 Q/K/V。"""
    project = getattr(attention, "project_qkv", None)
    if project is not None:
        return project(values)
    if attention.in_proj_weight is None:
        raise RuntimeError("split attention requires packed self-attention projections")
    weights = attention.in_proj_weight.chunk(3, dim=0)
    if attention.in_proj_bias is None:
        biases = (None, None, None)
    else:
        biases = attention.in_proj_bias.chunk(3, dim=0)
    return tuple(F.linear(values, weight, bias) for weight, bias in zip(weights, biases))


def kv_head_count(attention) -> int:
    """读取 K/V 头数；标准 MHA 未声明时等于 Q 头数。"""
    return int(getattr(attention, "num_kv_heads", attention.num_heads))


def split_heads(values: torch.Tensor, head_count: int) -> torch.Tensor:
    """将 [batch, tokens, model] 变换为 [batch, heads, tokens, head_dim]。"""
    batch_size, sequence_length, model_dim = values.shape
    head_dim = model_dim // head_count
    return values.reshape(batch_size, sequence_length, head_count, head_dim).transpose(1, 2)


def merge_heads(values: torch.Tensor) -> torch.Tensor:
    """将 [batch, heads, tokens, head_dim] 合并回模型维度。"""
    return values.transpose(1, 2).contiguous().reshape(
        values.shape[0],
        values.shape[2],
        values.shape[1] * values.shape[3],
    )


def finish_layer(layer, hidden: torch.Tensor, attended: torch.Tensor) -> torch.Tensor:
    """复用 TransformerEncoderLayer 的残差、归一化和 FFN。"""
    if layer.norm_first:
        hidden = hidden + layer.dropout1(attended)
        return hidden + layer._ff_block(layer.norm2(hidden))
    hidden = layer.norm1(hidden + layer.dropout1(attended))
    return layer.norm2(hidden + layer._ff_block(hidden))
