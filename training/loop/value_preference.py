"""技能价值辅助排序损失。"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from common.config import load_project_config
from common.skills import SkillBook
from common.policy.data.policy_actions import load_policy_actions

from ..config import ValuePreferenceConfig


def load_skill_values(job_tag: str) -> dict[str, float]:
    """加载技能与 policy action 的 action key -> value 回退映射。"""
    project_config = load_project_config(job_tag=job_tag)
    skill_book = SkillBook.from_project_config(project_config)
    values = {
        skill.key: float(skill.value)
        for skill in skill_book.enabled_skills()
    }
    for action in load_policy_actions():
        if action.key in values:
            raise ValueError(f"policy action key conflicts with skill key: {action.key}")
        values[action.key] = float(action.value)
    return values


def compute_value_preference_loss(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    config: ValuePreferenceConfig,
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
    return pair_loss[pair_mask].mean().to(dtype=logits.dtype)
