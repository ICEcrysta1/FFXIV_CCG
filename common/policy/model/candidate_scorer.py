"""候选动作 scorer：把 CLS 与候选 Transformer hidden 映射为 logits。"""

from __future__ import annotations

import torch
import torch.nn as nn

from .activation import (
    activation_hidden,
    gated_hidden_dim,
    resolve_pointwise_activation,
    uses_gate,
)


class CandidateScorer(nn.Module):
    """组合 CLS 与候选 Transformer hidden，输出每个候选的分数。

    激活与主干 FFN 共用 ``model.transformer_activation``：SwiGLU 走门控三投影，
    GELU/ReLU 走单条上行投影。
    """

    def __init__(
        self,
        *,
        d_model: int,
        dropout: float,
        activation: str,
    ):
        super().__init__()
        input_dim = 2 * d_model
        self.gated = uses_gate(activation)
        self.activation = resolve_pointwise_activation(activation)
        hidden_dim = gated_hidden_dim(
            activation,
            in_features=input_dim,
            out_features=1,
            hidden_dim=d_model,
        )
        if self.gated:
            self.gate_proj = nn.Linear(input_dim, hidden_dim)
        self.up_proj = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.down_proj = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        *,
        cls_hidden: torch.Tensor,
        candidate_hidden: torch.Tensor,
    ) -> torch.Tensor:
        candidate_count = candidate_hidden.shape[1]
        cls_for_candidates = cls_hidden.unsqueeze(1).expand(-1, candidate_count, -1)
        paired = torch.cat((cls_for_candidates, candidate_hidden), dim=-1)
        hidden = activation_hidden(
            self.activation,
            self.up_proj(paired),
            gate=self.gate_proj(paired) if self.gated else None,
        )
        return self.down_proj(self.dropout(hidden)).squeeze(-1)
