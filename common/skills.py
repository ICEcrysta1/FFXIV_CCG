"""技能索引层。

这个文件负责把配置里的技能定义做成便于查询的索引，
让状态机可以按技能 key 或游戏内 id 快速取到对应技能。
"""

from __future__ import annotations

from .config import JobConfig, ProjectConfig, SystemConfig
from .models import ActionKind, SkillDefinition


class SkillBook:
    """当前职业的技能查询表。"""

    def __init__(self, skills: tuple[SkillDefinition, ...]):
        self._ordered = skills
        self._by_key: dict[str, SkillDefinition] = {}
        self._by_game_id: dict[int, SkillDefinition] = {}
        for skill in skills:
            if skill.key in self._by_key:
                raise ValueError(f"duplicate skill key: {skill.key}")
            if skill.game_id in self._by_game_id:
                raise ValueError(f"duplicate skill game_id: {skill.game_id}")
            self._by_key[skill.key] = skill
            self._by_game_id[skill.game_id] = skill

    @classmethod
    def from_job_config(cls, job_config: JobConfig) -> "SkillBook":
        return cls(job_config.skills)

    @classmethod
    def from_system_config(cls, system_config: SystemConfig) -> "SkillBook":
        return cls(system_config.skills)

    @classmethod
    def from_project_config(cls, project_config: ProjectConfig) -> "SkillBook":
        return cls(project_config.system.skills + project_config.job.skills)

    def get(self, ref: str | int) -> SkillDefinition:
        if isinstance(ref, str):
            return self._by_key[ref]
        return self._by_game_id[ref]

    def enabled_skills(self, kind: ActionKind | None = None) -> tuple[SkillDefinition, ...]:
        skills = tuple(skill for skill in self._ordered if skill.enabled)
        if kind is None:
            return skills
        return tuple(skill for skill in skills if skill.kind == kind)

    def keys(self) -> tuple[str, ...]:
        return tuple(skill.key for skill in self._ordered)

    def __contains__(self, ref: str | int) -> bool:
        if isinstance(ref, str):
            return ref in self._by_key
        return ref in self._by_game_id
