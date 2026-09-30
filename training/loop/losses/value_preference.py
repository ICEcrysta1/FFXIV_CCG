"""技能价值辅助排序损失。"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from ...config import ValuePreferenceConfig


def compute_value_preference_loss(
    logits: torch.Tensor,
    batch: Mapping[str, object],
    config: ValuePreferenceConfig,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """只强化“人类已选择的高价值技能”相对低价值合法候选的排序。

    价值不是无条件的动作优先级：如果数据中的 label 本身价值更低，
    这里不施加反向约束，避免运行时 value 压过状态、时序和合法性。
    """
    if not config.enabled or config.loss_weight <= 0.0:
        return logits.new_zeros(())

    values = batch.get("candidate_values")
    if values is None:
        raise ValueError(
            "value preference requires runtime candidate values from the job YAML"
        )
    if values.ndim != 2 or values.shape != logits.shape:
        raise ValueError(
            "candidate_values must have the same [batch, candidates] shape as logits"
        )
    values = values.float()
    labels = batch["label_index"]
    chosen_values = values.gather(1, labels.unsqueeze(1)).squeeze(1)
    value_delta = (chosen_values.unsqueeze(1) - values).clamp_min(0.0)
    pair_mask = value_delta > 0.0

    legal_mask = batch.get("candidate_legal_mask")
    if legal_mask is not None:
        pair_mask &= legal_mask.to(dtype=torch.bool)

    if not pair_mask.any():
        return logits.new_zeros(())

    chosen_logits = logits.gather(1, labels.unsqueeze(1))
    logit_delta = chosen_logits - logits
    pair_margin = value_delta * config.margin_scale
    pair_loss = F.softplus(pair_margin - logit_delta.float())
    if sample_weights is not None:
        if sample_weights.shape != labels.shape:
            raise ValueError("value preference sample weights must be [batch]")
        pair_loss = pair_loss * sample_weights.float().unsqueeze(1)
    return pair_loss[pair_mask].mean().to(dtype=logits.dtype)
