"""Transformer 注意力使用的旋转位置编码。"""

from __future__ import annotations

import torch
from torch import nn


class RotaryPositionEncoding(nn.Module):
    """对 attention 的 Q/K 应用 RoPE。

    位置编号由输入编码器按样本生成；本模块只负责频率表和 Q/K 旋转，
    不感知场景、历史与当前状态等业务分区。

    仅旋转每个头的末尾通道，相邻两个通道组成一个正旋转 R(+θ) 平面。
    三角函数和旋转乘加均使用 FP32，输出恢复为输入 dtype。
    """

    def __init__(
        self,
        head_dim: int,
        *,
        rope_head_dim: int | None = None,
        theta: float = 10_000.0,
    ):
        super().__init__()
        if isinstance(head_dim, bool) or not isinstance(head_dim, int) or head_dim <= 0:
            raise ValueError("RoPE head_dim must be a positive integer")
        rope_head_dim = head_dim if rope_head_dim is None else rope_head_dim
        if rope_head_dim <= 0 or rope_head_dim % 2 or rope_head_dim > head_dim:
            raise ValueError("RoPE rope_head_dim must be a positive even number not exceeding head_dim")
        if theta <= 1.0:
            raise ValueError("RoPE theta must be greater than 1")

        self.head_dim = int(head_dim)
        self.rope_head_dim = int(rope_head_dim)
        self.theta = float(theta)
        channel_range = torch.arange(0, self.rope_head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (self.theta ** (channel_range / self.rope_head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # 整体转换 dtype 后重新生成频率，避免低精度量化后再转 FP32 丢失精度。
        channel_range = torch.arange(
            0, self.rope_head_dim, 2, device=self.inv_freq.device, dtype=torch.float32,
        )
        self.inv_freq = 1.0 / (self.theta ** (channel_range / self.rope_head_dim))
        return self

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_position_ids: torch.Tensor,
        key_position_ids: torch.Tensor,
        cosine_sine: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回按各自逻辑位置旋转后的 Q/K。"""
        if cosine_sine is None:
            cosine_sine = self._cos_sin(query_position_ids)
        key_cosine_sine = cosine_sine if key_position_ids is query_position_ids else None
        return (
            self.apply(query, query_position_ids, cosine_sine),
            self.apply(key, key_position_ids, key_cosine_sine),
        )

    def _cos_sin(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """同一次编码共享 FP32 三角函数表。"""
        if position_ids.ndim != 2:
            raise ValueError("RoPE position_ids must have shape [batch, tokens]")
        position_values = position_ids.to(dtype=torch.float32).unsqueeze(-1)
        inv_freq = self.inv_freq.to(device=position_ids.device, dtype=torch.float32)
        frequencies = position_values * inv_freq.view(1, 1, -1)
        return frequencies.cos().unsqueeze(1), frequencies.sin().unsqueeze(1)

    def apply(
        self,
        values: torch.Tensor,
        position_ids: torch.Tensor,
        cosine_sine: tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        inverse: bool = False,
        merge_heads: bool = False,
    ) -> torch.Tensor:
        """对 head 张量应用 RoPE；输出聚合通过 inverse 恢复到 query 位置坐标。"""
        if values.ndim != 4:
            raise ValueError("RoPE values must have shape [batch, heads, tokens, head_dim]")
        if values.shape[-1] != self.head_dim:
            raise ValueError(
                "RoPE head dimension mismatch: "
                f"{values.shape[-1]} != {self.head_dim}"
            )
        if position_ids.ndim != 2:
            raise ValueError("RoPE position_ids must have shape [batch, tokens]")
        if position_ids.shape[0] != values.shape[0] or position_ids.shape[1] != values.shape[2]:
            raise ValueError(
                "RoPE position_ids shape must match the batch and token dimensions: "
                f"{tuple(position_ids.shape)} != {(values.shape[0], values.shape[2])}"
            )

        if cosine_sine is None:
            cosine_sine = self._cos_sin(position_ids)
        cosine, sine = (value.to(dtype=torch.float32) for value in cosine_sine)
        if (values.is_cuda and self.training and torch.is_grad_enabled()
                and not torch.onnx.is_in_onnx_export() and not torch.jit.is_tracing()
                and not cosine.requires_grad and not sine.requires_grad):
            from .rotary_cuda import rotary_transform

            return rotary_transform(values, cosine, sine, inverse, merge_heads)
        if inverse:
            sine = -sine
        pairs = values[..., -self.rope_head_dim:].float().unflatten(-1, (-1, 2))
        first, second = pairs[..., 0], pairs[..., 1]
        # 标准正旋转 R(+θ)：out_first = first*cos - second*sin，
        # out_second = first*sin + second*cos；inverse 仅改变 sin 的符号。
        rotated = torch.stack(
            (
                first * cosine - second * sine,
                first * sine + second * cosine,
            ),
            dim=-1,
        ).flatten(-2).to(dtype=values.dtype)
        output = torch.cat((values[..., :-self.rope_head_dim], rotated), dim=-1)
        if merge_heads:
            return output.transpose(1, 2).contiguous().flatten(2)
        return output
