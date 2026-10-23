"""Transformer 层定义、激活重算与运行时诊断。"""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .activation import (
    activation_hidden,
    gated_hidden_dim,
    resolve_pointwise_activation,
    uses_gate,
)

_compiled_activation_hidden = torch.compile(activation_hidden, fullgraph=True, dynamic=True)


class RMSNorm(nn.RMSNorm):
    """在 FP32 中归一化，乘可学习通道尺度后恢复输入精度。"""

    def __init__(self, dim: int, eps: float = 1e-20, *, device=None, dtype=None):
        super().__init__(dim, eps=eps, device=device, dtype=dtype)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output_dtype = values.dtype
        values = values.float()
        variance = values.square().mean(dim=-1, keepdim=True)
        values = values * torch.rsqrt(variance + self.eps)
        return (self.weight * values).to(output_dtype)


class TraceableTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """使用可学习 RMSNorm 的 Transformer 层，同时提供逐 head attention。"""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-20,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
        *,
        num_kv_heads: int | None = None,
        q_lora_rank: int | None = None,
        swiglu_limit: float = 10.0,
    ):
        if dim_feedforward < 1:
            raise ValueError("dim_feedforward must be positive")
        self.swiglu_limit = swiglu_limit
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
        # 父类融合路径固定执行 LayerNorm；所有激活都走模块调用，避免绕过 RMSNorm。
        self.activation_relu_or_gelu = 0
        self.norm1 = RMSNorm(
            d_model, eps=layer_norm_eps,
            device=device, dtype=dtype,
        )
        self.norm2 = RMSNorm(
            d_model, eps=layer_norm_eps,
            device=device, dtype=dtype,
        )
        if num_kv_heads is None:
            num_kv_heads = int(nhead)
        num_kv_heads = int(num_kv_heads)
        if num_kv_heads <= 0 or num_kv_heads > self.self_attn.num_heads:
            raise ValueError("num_kv_heads must be between 1 and nhead")
        if self.self_attn.num_heads % num_kv_heads != 0:
            raise ValueError("nhead must be divisible by num_kv_heads")
        self.num_kv_heads = num_kv_heads
        from ..attention.grouped_attention import GroupedQueryAttention

        self.self_attn = GroupedQueryAttention(
            embed_dim=d_model, num_heads=nhead, num_kv_heads=num_kv_heads,
            q_lora_rank=q_lora_rank, dropout=dropout, bias=bias,
            batch_first=batch_first, device=device, dtype=dtype,
        )
        # GELU/ReLU 沿用原有 linear1/linear2 和 state_dict 键，不增加参数。
        self.gate_proj = (
            nn.Linear(d_model, hidden_dim, bias=bias, device=device, dtype=dtype)
            if use_swiglu else None
        )
        self.activation_checkpoint_ffn = False
        self.activation_checkpoint_attention = False
        self.activation_checkpoint_attention_block = False
        self._skip_activation_checkpoint = False
        self._runtime_debug = None
        self._runtime_debug_layer_index: int | None = None

    def set_activation_checkpoint_ffn(self, enabled: bool) -> None:
        """控制训练时是否重算 FFN 中间激活。"""
        self.activation_checkpoint_ffn = bool(enabled)

    def set_activation_checkpoint_attention(self, enabled: bool) -> None:
        """控制训练时是否重算 Attention 中间激活。"""
        self.activation_checkpoint_attention = bool(enabled)

    def set_activation_checkpoint_attention_block(self, enabled: bool) -> None:
        """控制 Attention 重算粒度：整块（含投影）还是只重算 SDPA。"""
        self.activation_checkpoint_attention_block = bool(enabled)

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
                def activate(up, gate):
                    return activation_hidden(
                        self.activation, up, gate=gate, limit=self.swiglu_limit,
                    )

                up, gate = self.linear1(x), self.gate_proj(x)
                if (
                    self.training
                    and torch.is_grad_enabled()
                    and not self._skip_activation_checkpoint
                    and not torch.onnx.is_in_onnx_export()
                ):
                    if up.is_cuda:
                        # 融合前后向只保存原始 up/gate；CPU 使用局部重算，dropout 均在外部。
                        hidden = _compiled_activation_hidden(
                            self.activation, up, gate=gate, limit=self.swiglu_limit,
                        )
                    elif not self.activation_checkpoint_ffn:
                        hidden = checkpoint(activate, up, gate, use_reentrant=False, preserve_rng_state=False)
                    else:
                        hidden = activate(up, gate)
                else:
                    hidden = activate(up, gate)
                # 中间 dropout 放在乘积之后；残差分支 dropout 仍只执行一次。
                return self.dropout2(self.linear2(self.dropout(hidden)))
            return super()._ff_block(x)
