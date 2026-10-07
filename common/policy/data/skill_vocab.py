"""SkillVocab: raw_skill_id ↔ skill_vocab_id 映射。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Iterator

from common.config import load_project_config

from .policy_actions import load_policy_actions


class SkillVocab:
    """技能词表映射；新训练从配置构建，已有模型从保存的契约恢复。"""

    def __init__(self, vocab: dict[int, int], reverse: dict[int, int]):
        self._vocab = dict(vocab)
        self._reverse = dict(reverse)

    @classmethod
    def from_entries(cls, entries: Sequence[tuple[int, int]]) -> SkillVocab:
        """恢复完整映射并校验每个 embedding 行，不读取项目配置。"""
        vocab: dict[int, int] = {}
        reverse: dict[int, int] = {}
        for entry in entries:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                raise ValueError("skill vocab entry must contain raw_skill_id and vocab_id")
            raw_id, vocab_id = entry
            if type(raw_id) is not int or type(vocab_id) is not int:
                raise ValueError("skill vocab ids must be integers")
            if vocab_id <= 0:
                raise ValueError("skill vocab entries must not use padding row 0")
            if raw_id in vocab or vocab_id in reverse:
                raise ValueError("skill vocab raw ids and vocabulary rows must be unique")
            vocab[raw_id] = vocab_id
            reverse[vocab_id] = raw_id
        if not vocab or sorted(reverse) != list(range(1, len(vocab) + 1)):
            raise ValueError("skill vocab rows must exactly cover 1..N")
        # 编号顺序是保存映射的唯一顺序；raw_id=0 可以是正式 policy 动作。
        ordered = {reverse[row]: row for row in sorted(reverse)}
        return cls(ordered, reverse)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> SkillVocab:
        """恢复 checkpoint/部署包中完整的技能词表。"""
        if not isinstance(payload, Mapping) or set(payload) != {"size", "padding_vocab_id", "entries"}:
            raise ValueError("skill vocab contract must contain size, padding_vocab_id and entries")
        if type(payload["padding_vocab_id"]) is not int or payload["padding_vocab_id"] != 0:
            raise ValueError("skill vocab padding_vocab_id must be 0")
        entries = payload["entries"]
        if not isinstance(entries, (list, tuple)):
            raise ValueError("skill vocab entries must be a list")
        pairs = []
        for entry in entries:
            if not isinstance(entry, Mapping) or set(entry) != {"raw_skill_id", "vocab_id"}:
                raise ValueError("skill vocab entry must contain raw_skill_id and vocab_id")
            pairs.append((entry["raw_skill_id"], entry["vocab_id"]))
        vocab = cls.from_entries(pairs)
        if type(payload["size"]) is not int or payload["size"] != vocab.size():
            raise ValueError("skill vocab size differs from its complete mapping")
        return vocab

    def to_dict(self) -> dict[str, object]:
        """保存实际使用的完整映射，不从当前 YAML 重建。"""
        return {
            "size": self.size(),
            "padding_vocab_id": 0,
            "entries": [
                {"raw_skill_id": raw_id, "vocab_id": vocab_id}
                for raw_id, vocab_id in sorted(self, key=lambda entry: entry[1])
            ],
        }

    def assert_matches(self, entries: Sequence[tuple[int, int]], *, context: str) -> None:
        """直接比较完整映射，并报告首个缺失或错位的技能。"""
        other = self.from_entries(entries)
        if self._vocab == other._vocab:
            return
        for raw_id in sorted(self._vocab.keys() | other._vocab.keys()):
            expected = self._vocab.get(raw_id)
            actual = other._vocab.get(raw_id)
            if expected != actual:
                raise ValueError(
                    f"{context} skill vocab mismatch: raw_skill_id={raw_id}, "
                    f"expected vocab_id={expected}, actual vocab_id={actual}"
                )

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

        # policy 动作不属于 SkillBook，但在模型输出和 label 中仍是正式 token。
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

    def __len__(self) -> int:
        return self.size()

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return iter(self._vocab.items())
