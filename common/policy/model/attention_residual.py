"""沿网络深度聚合历史子层输出的 Full Attention Residual。"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class FullAttentionResidual(nn.Module):
    """用深度维 softmax attention 替代固定单位权重的残差累加。

    每个 Transformer block 有 attention 和 FFN 两个子层，因此模型还会为
    最终输出保留一个 query。query 零初始化时，所有已有深度源获得均匀权重，
    使新结构从有界的平均聚合开始训练。
    """

    def __init__(self, d_model: int, num_queries: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if num_queries <= 0:
            raise ValueError("num_queries must be positive")
        self.pseudo_queries = nn.Parameter(torch.zeros(num_queries, d_model))
        self._logit_scale = d_model**-0.5
        self.key_norms = nn.ModuleList(
            nn.RMSNorm(d_model) for _ in range(num_queries)
        )

    @property
    def num_queries(self) -> int:
        """返回可用的深度聚合 query 数量。"""
        return int(self.pseudo_queries.shape[0])

    def reset_parameters(self) -> None:
        """恢复论文建议的零 query / 单位 RMSNorm 初始化。"""
        nn.init.zeros_(self.pseudo_queries)
        for key_norm in self.key_norms:
            nn.init.ones_(key_norm.weight)

    def forward(
        self,
        sources: Sequence[torch.Tensor],
        query_index: int,
    ) -> torch.Tensor:
        """对每个 token 的历史深度源执行一次聚合。

        ``sources`` 中每个 tensor 的形状都是 ``[batch, tokens, d_model]``。
        逻辑上仍沿深度维计算 softmax，但只堆叠 ``[sources, batch, tokens]``
        的 logits，避免为 value/key 额外复制完整的历史激活。
        """
        if not sources:
            raise ValueError("attention residual requires at least one source")
        if not 0 <= query_index < self.num_queries:
            raise IndexError(
                f"attention residual query index {query_index} is out of range "
                f"for {self.num_queries} queries"
            )

        query = self.pseudo_queries[query_index]
        source_values = tuple(sources)
        logits = torch.cat(
            tuple(
                torch.einsum(
                    "d,btd->bt",
                    query,
                    self.key_norms[query_index](source),
                ).unsqueeze(0)
                * self._logit_scale
                for source in source_values
            ),
            dim=0,
        )
        weights = torch.softmax(logits, dim=0)
        output = torch.zeros_like(source_values[0])
        for source_index, source in enumerate(source_values):
            output.add_(weights[source_index].unsqueeze(-1) * source)
        return output
