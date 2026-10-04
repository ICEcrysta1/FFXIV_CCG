"""配置驱动的固定动作输出空间，与技能输入词表显式对齐。"""

from __future__ import annotations

from dataclasses import dataclass

from common.config import load_project_config
from common.models import ActionKind

from .policy_actions import load_policy_actions
from .skill_vocab import SkillVocab


@dataclass(frozen=True)
class ActionSpace:
    """只包含启用的真实技能和已注册控制动作，不含 padding。"""

    action_keys: tuple[str, ...]
    action_to_vocab_id: tuple[int, ...]
    action_is_gcd: tuple[bool, ...]

    @classmethod
    def from_job_tag(cls, job_tag: str) -> "ActionSpace":
        return cls.from_config(load_project_config(job_tag=job_tag))

    @classmethod
    def from_config(cls, project_config, *, skill_vocab: SkillVocab | None = None) -> "ActionSpace":
        vocab = skill_vocab or SkillVocab.build_from_config(project_config)
        raw_ids: dict[str, int] = {}
        gcd_flags: dict[str, bool] = {}
        for skill in (*project_config.system.skills, *project_config.job.skills):
            if skill.enabled:
                if skill.key in raw_ids:
                    raise ValueError(f"duplicate output action key: {skill.key}")
                raw_ids[skill.key] = skill.game_id
                gcd_flags[skill.key] = skill.kind == ActionKind.GCD
        for action in load_policy_actions():
            if action.key in raw_ids:
                raise ValueError(f"duplicate output action key: {action.key}")
            raw_ids[action.key] = action.raw_id
            gcd_flags[action.key] = action.kind == "gcd"
        # 与 C# StringComparer.Ordinal 一致；现行技能 key 均为 ASCII。
        keys = tuple(sorted(raw_ids, key=lambda value: value.encode("utf-16-be")))
        rows = tuple(vocab.require_lookup(raw_ids[key], context=f"output action {key}") for key in keys)
        if not keys or len(set(rows)) != len(rows):
            raise ValueError("output actions must be nonempty and map to distinct vocabulary rows")
        return cls(keys, rows, tuple(gcd_flags[key] for key in keys))
