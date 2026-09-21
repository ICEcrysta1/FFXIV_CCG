"""支持独立 Q/K/V 投影的 grouped-query attention。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    """Q 头与 K/V 头数量可不同的 self-attention 投影模块。

    该模块只负责投影和提供与 ``MultiheadAttention`` 兼容的基础属性；
    split encoder 的正式路径会直接调用 ``project_qkv``；两条路径共用按 KV 组
    执行的 SDPA，通过零 stride 广播共享 K/V，减少调用且不复制 K/V 输入。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        *,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError("embed_dim and num_heads must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if num_kv_heads <= 0 or num_kv_heads > num_heads:
            raise ValueError("num_kv_heads must be between 1 and num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        factory_kwargs = {"device": device, "dtype": dtype}
        self.embed_dim = int(embed_dim)
        self.kdim = int(embed_dim)
        self.vdim = int(embed_dim)
        self._qkv_same_embed_dim = False
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = float(dropout)
        self.batch_first = bool(batch_first)

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias, **factory_kwargs)
        kv_dim = self.num_kv_heads * self.head_dim
        self.k_proj = nn.Linear(self.embed_dim, kv_dim, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(self.embed_dim, kv_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias, **factory_kwargs)
        for projection in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            if projection.bias is not None:
                nn.init.zeros_(projection.bias)

        # TransformerEncoderLayer 的 fast path 只接受 packed MHA 权重。
        # 显式置空后会稳定地走其 _sa_block，再调用本模块的 forward。
        self.register_parameter("in_proj_weight", None)
        self.register_parameter("in_proj_bias", None)
        self.bias_k = None
        self.bias_v = None
        self.add_zero_attn = False

    def project_qkv(
        self,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """分别投影 Q、K、V，其中 K/V 只保留配置的头数。"""
        return self.q_proj(values), self.k_proj(values), self.v_proj(values)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """提供 TransformerEncoderLayer 所需的基础 MHA 调用接口。"""
        is_batched = query.ndim == 3
        if query.ndim not in {2, 3} or key.ndim != query.ndim or value.ndim != query.ndim:
            raise ValueError("GroupedQueryAttention expects equally ranked 2D or 3D inputs")

        if not is_batched:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
        elif not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        query_projection, key_projection, value_projection = self.project_qkv(query)
        query_heads = _split_heads(query_projection, self.num_heads)
        key_heads = _split_heads(key_projection, self.num_kv_heads)
        value_heads = _split_heads(value_projection, self.num_kv_heads)
        attention_mask, use_causal = _prepare_attention_mask(
            attn_mask,
            key_padding_mask,
            batch_size=query.shape[0],
            query_count=query.shape[1],
            key_count=key.shape[1],
            num_heads=self.num_heads,
            dtype=query_heads.dtype,
            device=query_heads.device,
            is_causal=is_causal,
        )

        if need_weights:
            attended, weights = _manual_attention(
                query_heads,
                key_heads,
                value_heads,
                attention_mask,
                use_causal=use_causal,
                dropout_p=self.dropout if self.training else 0.0,
            )
            if average_attn_weights:
                weights = weights.mean(dim=1)
        else:
            attended = scaled_dot_product_attention(
                query_heads,
                key_heads,
                value_heads,
                attn_mask=attention_mask,
                is_causal=use_causal,
                dropout_p=self.dropout if self.training else 0.0,
            )
            weights = None

        output = _merge_heads(attended)
        output = self.out_proj(output)
        if not is_batched:
            output = output.squeeze(0)
            if weights is not None:
                weights = weights.squeeze(0)
        elif not self.batch_first:
            output = output.transpose(0, 1)
        return output, weights


def expand_kv_heads(values: torch.Tensor, query_head_count: int) -> torch.Tensor:
    """物化展开 K/V，仅供 trace/显式权重路径使用。"""
    value_head_count = values.shape[1]
    if value_head_count == query_head_count:
        return values
    if value_head_count <= 0 or query_head_count % value_head_count != 0:
        raise ValueError(
            "query head count must be divisible by key/value head count: "
            f"{query_head_count} % {value_head_count} != 0"
        )
    return values.repeat_interleave(query_head_count // value_head_count, dim=1)


def _split_heads(values: torch.Tensor, head_count: int) -> torch.Tensor:
    batch_size, sequence_length, model_dim = values.shape
    head_dim = model_dim // head_count
    return values.reshape(batch_size, sequence_length, head_count, head_dim).transpose(1, 2)


def _merge_heads(values: torch.Tensor) -> torch.Tensor:
    return values.transpose(1, 2).contiguous().reshape(
        values.shape[0],
        values.shape[2],
        values.shape[1] * values.shape[3],
    )


def scaled_dot_product_attention(
    query_heads: torch.Tensor,
    key_heads: torch.Tensor,
    value_heads: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None,
    is_causal: bool,
    dropout_p: float,
) -> torch.Tensor:
    """按 KV 组调用 SDPA；单 KV 的全部 Q 头只需一次调用。"""
    query_head_count = query_heads.shape[1]
    kv_head_count = key_heads.shape[1]
    if kv_head_count == query_head_count or kv_head_count == 1:
        # expand 只创建 head stride 为 0 的视图，保持 K/V 的原始存储。
        # 显式匹配头数，让普通 SDPA 后端处理广播，避免 enable_gqa 的展开路径。
        return F.scaled_dot_product_attention(
            query_heads,
            key_heads.expand(-1, query_head_count, -1, -1),
            value_heads.expand(-1, query_head_count, -1, -1),
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

    # 多 KV 逐组广播；直接展平额外的 group 维度可能隐式复制 K/V。
    group_size = query_head_count // kv_head_count
    outputs = []
    for kv_index in range(kv_head_count):
        group_start = kv_index * group_size
        group_end = group_start + group_size
        group_mask = attn_mask
        if (
            attn_mask is not None
            and attn_mask.ndim == 4
            and attn_mask.shape[1] == query_head_count
        ):
            group_mask = attn_mask[:, group_start:group_end]
        outputs.append(
            F.scaled_dot_product_attention(
                query_heads[:, group_start:group_end],
                key_heads[:, kv_index : kv_index + 1].expand(-1, group_size, -1, -1),
                value_heads[:, kv_index : kv_index + 1].expand(-1, group_size, -1, -1),
                attn_mask=group_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )
        )
    return torch.cat(outputs, dim=1)


def _manual_attention(
    query_heads: torch.Tensor,
    key_heads: torch.Tensor,
    value_heads: torch.Tensor,
    attn_mask: torch.Tensor | None,
    *,
    use_causal: bool,
    dropout_p: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_heads = expand_kv_heads(key_heads, query_heads.shape[1])
    value_heads = expand_kv_heads(value_heads, query_heads.shape[1])
    scores = torch.matmul(query_heads, key_heads.transpose(-2, -1)) * (
        query_heads.shape[-1] ** -0.5
    )
    if attn_mask is not None:
        scores = scores + attn_mask
    elif use_causal:
        causal_mask = torch.ones(
            (query_heads.shape[-2], key_heads.shape[-2]),
            dtype=torch.bool,
            device=query_heads.device,
        ).tril()
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    if dropout_p:
        weights = F.dropout(weights, p=dropout_p, training=True)
    return torch.matmul(weights, value_heads), weights


def _prepare_attention_mask(
    attn_mask: torch.Tensor | None,
    key_padding_mask: torch.Tensor | None,
    *,
    batch_size: int,
    query_count: int,
    key_count: int,
    num_heads: int,
    dtype: torch.dtype,
    device: torch.device,
    is_causal: bool,
) -> tuple[torch.Tensor | None, bool]:
    """把 MHA 的屏蔽语义转换成 SDPA 的加性 mask。"""
    masks: list[torch.Tensor] = []
    if attn_mask is not None:
        masks.append(
            _normalize_attention_mask(
                attn_mask,
                batch_size=batch_size,
                query_count=query_count,
                key_count=key_count,
                num_heads=num_heads,
                dtype=dtype,
                device=device,
            )
        )
    if key_padding_mask is not None:
        if tuple(key_padding_mask.shape) != (batch_size, key_count):
            raise ValueError("key_padding_mask must have shape [batch, key_tokens]")
        if key_padding_mask.dtype == torch.bool:
            padding = torch.zeros_like(key_padding_mask, dtype=dtype).masked_fill(
                key_padding_mask,
                float("-inf"),
            )
        else:
            padding = key_padding_mask.to(dtype=dtype)
        masks.append(padding[:, None, None, :])

    if is_causal and masks:
        causal = torch.zeros(
            (query_count, key_count),
            dtype=dtype,
            device=device,
        )
        causal = causal.masked_fill(
            ~torch.ones((query_count, key_count), dtype=torch.bool, device=device).tril(),
            float("-inf"),
        )
        masks.append(causal[None, None, :, :])
        is_causal = False

    if not masks:
        return None, is_causal
    result = masks[0]
    for mask in masks[1:]:
        result = result + mask
    return result, False


def _normalize_attention_mask(
    mask: torch.Tensor,
    *,
    batch_size: int,
    query_count: int,
    key_count: int,
    num_heads: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    mask = mask.to(device=device)
    if mask.ndim == 2:
        if tuple(mask.shape) != (query_count, key_count):
            raise ValueError("2D attn_mask must have shape [query_tokens, key_tokens]")
        mask = mask[None, None, :, :]
    elif mask.ndim == 3:
        if tuple(mask.shape) == (batch_size, query_count, key_count):
            mask = mask[:, None, :, :]
        elif tuple(mask.shape) == (batch_size * num_heads, query_count, key_count):
            mask = mask.reshape(batch_size, num_heads, query_count, key_count)
        else:
            raise ValueError(
                "3D attn_mask must have shape [batch, query, key] or "
                "[batch*heads, query, key]"
            )
    elif mask.ndim != 4:
        raise ValueError("attn_mask must have 2, 3, or 4 dimensions")
    if mask.dtype == torch.bool:
        return torch.zeros_like(mask, dtype=dtype).masked_fill(mask, float("-inf"))
    return mask.to(dtype=dtype)
