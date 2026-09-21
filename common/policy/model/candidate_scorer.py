"""候选动作 scorer，仅使用 Transformer 表示映射为 logits。"""

from __future__ import annotations

import torch
import torch.nn as nn


class CandidateScorer(nn.Module):
    """组合 CLS 与候选 Transformer hidden，输出每个候选的分数。"""

    def __init__(
        self,
        *,
        d_model: int,
        dropout: float,
    ):
        super().__init__()
        input_dim = 2 * d_model
        self.network = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        *,
        cls_hidden: torch.Tensor,
        candidate_hidden: torch.Tensor,
    ) -> torch.Tensor:
        candidate_count = candidate_hidden.shape[1]
        cls_for_candidates = cls_hidden.unsqueeze(1).expand(-1, candidate_count, -1)
        return self.network(
            torch.cat((cls_for_candidates, candidate_hidden), dim=-1)
        ).squeeze(-1)
