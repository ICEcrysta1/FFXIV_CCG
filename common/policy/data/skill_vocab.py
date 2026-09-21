"""SkillVocab: raw_skill_id ↔ skill_vocab_id 映射。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from common.config import load_project_config

from .policy_actions import load_policy_actions


class SkillVocab:
    """技能词表映射，只允许从正式项目配置构建。"""

    def __init__(self, vocab: dict[int, int], reverse: dict[int, int]):
        self._vocab = vocab
        self._reverse = reverse

    @classmethod
    def build_from_job_tag(cls, job_tag: str, config_path: Path | None = None) -> SkillVocab:
        """从职业标签构建 SkillVocab。"""
        project_config = load_project_config(config_path, job_tag=job_tag)
        if project_config.job.key != job_tag:
            raise ValueError(
                f"loaded config job {project_config.job.key!r} != requested {job_tag!r}"
            )
        return cls.build_from_config(project_config)

    @classmethod
    def build_from_config(cls, project_config) -> SkillVocab:
        """从已加载的项目配置构建 SkillVocab。"""
        vocab: dict[int, int] = {}
        next_id = 1

        for skill in sorted(project_config.system.skills, key=lambda item: (item.key, item.game_id)):
            if skill.game_id not in vocab:
                vocab[skill.game_id] = next_id
                next_id += 1

        for skill in sorted(project_config.job.skills, key=lambda item: (item.key, item.game_id)):
            if skill.game_id not in vocab:
                vocab[skill.game_id] = next_id
                next_id += 1

        # policy 动作不属于 SkillBook，但在模型候选和 label 中仍是正式 token。
        # 放在真实技能之后可保持既有真实技能 vocab id 稳定。
        for action in load_policy_actions():
            if action.raw_id in vocab:
                raise ValueError(
                    "policy action raw_id conflicts with game skill: "
                    f"{action.key} ({action.raw_id})"
                )
            vocab[action.raw_id] = next_id
            next_id += 1

        reverse = {vocab_id: raw_skill_id for raw_skill_id, vocab_id in vocab.items()}
        return cls(vocab, reverse)

    def lookup(self, raw_skill_id: int | None) -> int:
        """raw_skill_id → skill_vocab_id。"""
        if raw_skill_id is None:
            return 0
        return self._vocab.get(raw_skill_id, 0)

    def require_lookup(self, raw_skill_id: int | None, *, context: str) -> int:
        """要求 raw_skill_id 必须对应一个已注册动作。"""
        if raw_skill_id is None:
            raise ValueError(f"{context} raw_skill_id must not be None")
        vocab_id = self._vocab.get(raw_skill_id)
        if vocab_id is None:
            raise ValueError(f"{context} raw_skill_id {raw_skill_id} is not registered in SkillVocab")
        return vocab_id

    def reverse_lookup(self, vocab_id: int) -> int | None:
        """skill_vocab_id → raw_skill_id。"""
        return self._reverse.get(vocab_id)

    def size(self) -> int:
        """词表总大小（含 NON_SKILL）。"""
        return len(self._vocab) + 1

    @property
    def num_skills(self) -> int:
        """真实技能数（不含 NON_SKILL）。"""
        return len(self._vocab)

    def __len__(self) -> int:
        return self.size()

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return iter(self._vocab.items())
