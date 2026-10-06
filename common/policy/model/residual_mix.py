"""普通残差路径中逐层学习残差尺度与初始 token 重注入。"""

from __future__ import annotations

import torch
from torch import nn


class LearnedResidualMix(nn.Module):
    """在每层 attention 前执行 r*x+a*x0，不切断初始 token 的梯度。"""

    def __init__(
        self,
        n_layers: int,
        *,
        r_start: float,
        r_end: float,
        a_start: float,
        a_end: float,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("residual mixing requires at least one layer")
        # 使用确定性的线性端点插值，不消耗 RNG 或改变已有矩阵的初始化。
        self.r = nn.Parameter(torch.tensor([
            r_start + (r_end - r_start) * index / max(n_layers - 1, 1)
            for index in range(n_layers)
        ], dtype=torch.float32))
        self.a = nn.Parameter(torch.tensor([
            a_start + (a_end - a_start) * index / max(n_layers - 1, 1)
            for index in range(n_layers)
        ], dtype=torch.float32))

    def forward(
        self,
        hidden: torch.Tensor,
        initial_tokens: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        return self.r[layer_index] * hidden + self.a[layer_index] * initial_tokens
