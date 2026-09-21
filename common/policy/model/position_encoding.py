"""Transformer 注意力使用的旋转位置编码。"""

from __future__ import annotations

import torch
import torch.nn as nn


class RotaryPositionEncoding(nn.Module):
    """对 attention 的 Q/K 应用 RoPE。

    位置编号由输入编码器按样本生成；本模块只负责频率表和 Q/K 旋转，
    不感知 scene、history、candidate 等业务分区。

    旋转采用与 LLaMA/Hugging Face/GPT-NeoX 一致的标准正旋转 R(+θ)，
    外部复刻（C# 推理、独立参考实现）可直接套用主流 rotate_half 公式。
    """

    def __init__(self, head_dim: int, *, theta: float = 10_000.0):
        super().__init__()
        if head_dim <= 0 or head_dim % 2:
            raise ValueError("RoPE head_dim must be a positive even number")
        if theta <= 1.0:
            raise ValueError("RoPE theta must be greater than 1")

        self.head_dim = int(head_dim)
        self.theta = float(theta)
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (self.theta ** (channel_range / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_position_ids: torch.Tensor,
        key_position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回按各自逻辑位置旋转后的 Q/K。"""
        return (
            self.apply(query, query_position_ids),
            self.apply(key, key_position_ids),
        )

    def apply(self, values: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """对 ``[batch, heads, tokens, head_dim]`` 张量应用 RoPE。"""
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

        # 模型可以整体转换为 BF16/FP16，但三角函数在 ONNX/ORT 中必须
        # 保持 FP32；只把最终 cos/sin 结果转换到 activation dtype。
        position_values = position_ids.to(dtype=torch.float32).unsqueeze(-1)
        inv_freq = self.inv_freq.to(device=position_ids.device, dtype=torch.float32)
        frequencies = position_values * inv_freq.view(1, 1, -1)
        cosine = frequencies.cos().to(dtype=values.dtype).unsqueeze(1)
        sine = frequencies.sin().to(dtype=values.dtype).unsqueeze(1)
        half = self.head_dim // 2
        first, second = values[..., :half], values[..., half:]
        # 标准正旋转 R(+θ)：out_first = first*cos - second*sin，
        # out_second = first*sin + second*cos（负旋转会让 sin 项反号，
        # 与主流实现形成镜像差异）。
        return torch.cat(
            (
                first * cosine - second * sine,
                first * sine + second * cosine,
            ),
            dim=-1,
        )
