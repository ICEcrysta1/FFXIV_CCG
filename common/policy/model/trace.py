"""Transformer 编码 trace，供模型分析和调试共用。"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .activation import (
    activation_hidden,
    gated_hidden_dim,
    resolve_pointwise_activation,
    uses_gate,
)
from .grouped_attention import GroupedQueryAttention


class TraceableTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """保留标准 TransformerEncoderLayer 行为，同时提供逐 head attention。"""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
        *,
        num_kv_heads: int | None = None,
    ):
        if dim_feedforward < 1:
            raise ValueError("dim_feedforward must be positive")
        use_swiglu = uses_gate(activation)
        # 三个投影取代两个投影；保持同一 ff_dim 配置下矩阵参数量近似相等。
        hidden_dim = gated_hidden_dim(
            activation,
            in_features=d_model,
            out_features=d_model,
            hidden_dim=dim_feedforward,
        )
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            # SiLU 也使父类关闭仅支持 ReLU/GELU 的融合快速路径，避免跳过门控。
            activation=(
                resolve_pointwise_activation(activation) if use_swiglu else activation
            ),
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        if num_kv_heads is None:
            num_kv_heads = int(nhead)
        num_kv_heads = int(num_kv_heads)
        if num_kv_heads <= 0 or num_kv_heads > self.self_attn.num_heads:
            raise ValueError("num_kv_heads must be between 1 and nhead")
        if self.self_attn.num_heads % num_kv_heads != 0:
            raise ValueError("nhead must be divisible by num_kv_heads")
        self.num_kv_heads = num_kv_heads
        if num_kv_heads != self.self_attn.num_heads:
            original_attention = self.self_attn
            self.self_attn = GroupedQueryAttention(
                embed_dim=original_attention.embed_dim,
                num_heads=original_attention.num_heads,
                num_kv_heads=num_kv_heads,
                dropout=original_attention.dropout,
                bias=original_attention.in_proj_bias is not None,
                batch_first=original_attention.batch_first,
                device=original_attention.out_proj.weight.device,
                dtype=original_attention.out_proj.weight.dtype,
            )
        # GELU/ReLU 沿用原有 linear1/linear2 和 state_dict 键，不增加参数。
        self.gate_proj = (
            nn.Linear(d_model, hidden_dim, bias=bias, device=device, dtype=dtype)
            if use_swiglu else None
        )
        self.activation_checkpoint_ffn = False
        self.activation_checkpoint_attention = False
        self._skip_activation_checkpoint = False
        self._runtime_debug = None
        self._runtime_debug_layer_index: int | None = None

    def set_activation_checkpoint_ffn(self, enabled: bool) -> None:
        """控制训练时是否重算 FFN 中间激活。"""
        self.activation_checkpoint_ffn = bool(enabled)

    def set_activation_checkpoint_attention(self, enabled: bool) -> None:
        """控制训练时是否重算 Attention 中间激活。"""
        self.activation_checkpoint_attention = bool(enabled)

    def set_runtime_debug(self, recorder=None, *, layer_index: int | None = None) -> None:
        """接入可选的层级运行时调试记录器。"""
        self._runtime_debug = recorder
        self._runtime_debug_layer_index = layer_index

    def _debug_stage(self, operation: str):
        if self._runtime_debug is None or self._runtime_debug_layer_index is None:
            return nullcontext()
        return self._runtime_debug.stage(
            f"encoder.layer.{self._runtime_debug_layer_index}.{operation}"
        )

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """执行 FFN；训练 checkpoint 模式下只保存输入并在反向时重算。"""
        if (
            self.activation_checkpoint_ffn
            and not self._skip_activation_checkpoint
            and self.training
            and torch.is_grad_enabled()
        ):
            return checkpoint(self._ff_block_without_checkpoint, x, use_reentrant=False)
        return self._ff_block_without_checkpoint(x)

    def _ff_block_without_checkpoint(self, x: torch.Tensor) -> torch.Tensor:
        """共用 FFN 入口；SwiGLU 为 down(SiLU(gate(x)) * up(x))。"""
        with self._debug_stage("ffn"):
            if self.gate_proj is not None:
                hidden = activation_hidden(
                    self.activation,
                    self.linear1(x),
                    gate=self.gate_proj(x),
                )
                # 中间 dropout 放在乘积之后；残差分支 dropout 仍只执行一次。
                return self.dropout2(self.linear2(self.dropout(hidden)))
            return super()._ff_block(x)

@dataclass(frozen=True)
class ModelTrace:
    """包含每层 hidden、最终 hidden 和未平均 attention 的统一分析结果。"""

    encoded: dict[str, torch.Tensor]
    layer_hidden: tuple[torch.Tensor, ...]
    hidden: torch.Tensor
    attentions: tuple[torch.Tensor, ...]


@torch.no_grad()
def trace_encoder(encoder, encoded: dict[str, torch.Tensor]) -> ModelTrace:
    """运行 split encoder 并保留每层 hidden 与完整 attention 矩阵。"""
    from .attention_masks import build_split_attention_mask
    from .split_encoder import run_split_encoder

    prefix_hidden, candidate_hidden, cls_hidden, layer_hidden, attentions = run_split_encoder(
        encoder,
        encoded,
        collect_attention=True,
    )
    hidden = torch.cat((prefix_hidden, candidate_hidden, cls_hidden), dim=1)
    traced_encoded = dict(encoded)
    traced_encoded["attention_mask"] = build_split_attention_mask(
        prefix_length=int(encoded["prefix_length"]),
        candidate_count=int(encoded["candidate_count"]),
        device=hidden.device,
    )
    return ModelTrace(
        encoded=traced_encoded,
        layer_hidden=layer_hidden,
        hidden=hidden,
        attentions=attentions,
    )
