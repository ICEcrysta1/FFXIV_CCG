"""注意力投影、头维度整形与 Transformer 层收尾辅助。"""

from __future__ import annotations

import torch


def rotate_qk(
    rotary_position_encoding,
    query_heads: torch.Tensor,
    key_heads: torch.Tensor,
    query_position_ids: torch.Tensor,
    key_position_ids: torch.Tensor,
    cosine_sine: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """在进入 SDPA 前统一旋转 Q/K。"""
    return rotary_position_encoding(
        query_heads,
        key_heads,
        query_position_ids,
        key_position_ids,
        cosine_sine,
    )


def project_qkv(attention, values: torch.Tensor):
    """复用低秩 Q 与共享 KV 的唯一投影入口。"""
    return attention.project_qkv(values)


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
