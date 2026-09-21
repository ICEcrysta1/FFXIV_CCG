"""读取一次性固化的职业 vocab 画像与 scene 容量证据。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re


DEPLOYMENT_PROFILE_VERSION = 1


@dataclass(frozen=True)
class DeploymentProfile:
    """首次数据分析后固化的部署侧画像，不在日常导出时重扫语料。

    scene 容量不在此维护：它以职业模型配置为权威来源（随 checkpoint
    的 model_config 进入部署契约），profile 只保留 vocab 映射与佐证
    该容量的语料统计证据。
    """

    job_tag: str
    vocab_entries: tuple[tuple[int, int], ...]
    evidence: dict[str, object]

    @classmethod
    def load(cls, path: Path) -> "DeploymentProfile":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("deployment profile must be an object")
        expected = {
            "profile_version",
            "job_tag",
            "vocab_entries",
            "evidence",
        }
        if set(payload) != expected:
            raise ValueError("deployment profile fields are incomplete or unsupported")
        if int(payload["profile_version"]) != DEPLOYMENT_PROFILE_VERSION:
            raise ValueError("unsupported deployment profile version")
        entries = payload["vocab_entries"]
        if not isinstance(entries, list) or not entries:
            raise ValueError("deployment profile vocab_entries must be a non-empty list")
        evidence = payload["evidence"]
        if not isinstance(evidence, Mapping):
            raise ValueError("deployment profile evidence must be an object")
        profile = cls(
            job_tag=str(payload["job_tag"]),
            vocab_entries=tuple(
                (int(item["raw_skill_id"]), int(item["vocab_id"]))
                for item in entries
            ),
            evidence=dict(evidence),
        )
        if not profile.job_tag:
            raise ValueError("deployment profile job_tag must not be empty")
        return profile

    @classmethod
    def default_path(cls, job_tag: str) -> Path:
        normalized = str(job_tag)
        if re.fullmatch(r"[a-z][a-z0-9_]*", normalized) is None:
            raise ValueError(f"unsafe deployment profile job_tag: {job_tag!r}")
        return Path(__file__).with_name("profiles") / f"{normalized}.json"

    def to_capacity_report(
        self,
        *,
        scene_capacity: int,
        history_capacity: int,
    ) -> dict[str, object]:
        core = {
            "report_version": 1,
            "job_tag": self.job_tag,
            "scene_capacity": int(scene_capacity),
            "history_capacity": int(history_capacity),
            "vocab_entries": [
                {"raw_skill_id": raw_skill_id, "vocab_id": vocab_id}
                for raw_skill_id, vocab_id in self.vocab_entries
            ],
            "evidence": self.evidence,
        }
        return {**core, "semantic_sha256": stable_sha256(core)}


def stable_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
