"""从职业配置校验固定输出动作的技能价值定义。"""

from __future__ import annotations

from common.config import load_project_config
from common.policy.data.policy_actions import load_policy_actions
from common.skills import SkillBook


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
            raise ValueError(
                f"policy action key conflicts with skill key: {action.key}"
            )
        values[action.key] = float(action.value)
    return values
