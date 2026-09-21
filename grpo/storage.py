"""GRPO 自回归决策的磁盘存储与流式读取。"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import uuid

import torch

from common.torch_serialization import safe_torch_load


GRPO_ROLLOUT_FORMAT = 1


@dataclass(frozen=True)
class GrpoDecision:
    """一条采样轨迹中的决策 token 及其行为策略概率。"""

    batch: dict[str, object]
    candidate_keys: tuple[str, ...]
    action_index: int
    old_logprob: float


@dataclass(frozen=True)
class StoredGrpoTrajectory:
    """一条已落盘 GRPO 轨迹的轻量索引，不持有决策 tensor。"""

    path: Path
    scene_json_path: Path
    ppg: float
    greedy_ppg: float
    reward: float
    decision_count: int
    advantage: float | None = None


def _decision_to_payload(decision: GrpoDecision) -> dict[str, object]:
    """将决策转换成仅含 tensor/基础类型的安全序列化结构。"""
    return {
        "batch": decision.batch,
        "candidate_keys": list(decision.candidate_keys),
        "action_index": int(decision.action_index),
        "old_logprob": float(decision.old_logprob),
    }


def _decision_from_payload(payload: object) -> GrpoDecision:
    """从磁盘载荷恢复一条决策，并拒绝不完整的旧/损坏记录。"""
    if not isinstance(payload, Mapping):
        raise ValueError("GRPO decision payload must be a mapping")
    batch = payload.get("batch")
    candidate_keys = payload.get("candidate_keys")
    if not isinstance(batch, Mapping) or not isinstance(candidate_keys, (list, tuple)):
        raise ValueError("GRPO decision payload is missing batch or candidate_keys")
    return GrpoDecision(
        batch=dict(batch),
        candidate_keys=tuple(str(key) for key in candidate_keys),
        action_index=int(payload["action_index"]),
        old_logprob=float(payload["old_logprob"]),
    )


class GrpoRolloutStore:
    """按轨迹落盘并按 minibatch 流式恢复 GRPO 决策。

    轨迹文件只保存 CPU tensor；索引和奖励信息保持为很小的 Python 对象。
    训练更新时按轨迹顺序随机化，每次最多将一个轨迹和一个 minibatch 放入内存。
    """

    def __init__(
        self,
        root: Path,
        *,
        iteration: int,
        run_id: str | None = None,
    ) -> None:
        if iteration < 1:
            raise ValueError("GRPO iteration must be >= 1")
        normalized_root = Path(root).resolve()
        resolved_run_id = run_id or f"run-{uuid.uuid4().hex}"
        if not resolved_run_id.strip() or Path(resolved_run_id).name != resolved_run_id:
            raise ValueError("GRPO run_id must be a non-empty file-name component")
        self.root = normalized_root
        self.run_dir = normalized_root / resolved_run_id
        self.iteration_dir = self.run_dir / f"iteration_{iteration:03d}"
        self.iteration_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.iteration_dir / "manifest.json"
        self._entries: list[StoredGrpoTrajectory] = []
        self._write_manifest()

    @property
    def entries(self) -> tuple[StoredGrpoTrajectory, ...]:
        """返回不含决策 tensor 的轨迹索引。"""
        return tuple(self._entries)

    @property
    def total_decisions(self) -> int:
        """返回所有已落盘轨迹的决策数量。"""
        return sum(entry.decision_count for entry in self._entries)

    def write_trajectory(
        self,
        *,
        scene_json_path: Path,
        decisions: Sequence[GrpoDecision],
        ppg: float,
        greedy_ppg: float,
        reward: float,
    ) -> StoredGrpoTrajectory:
        """原子写入一条完整轨迹，并只将轻量索引留在内存。"""
        decision_list = tuple(decisions)
        if not decision_list:
            raise ValueError("cannot persist an empty GRPO trajectory")
        entry_index = len(self._entries)
        trajectory_path = self.iteration_dir / f"trajectory_{entry_index:05d}.pt"
        payload = {
            "format": GRPO_ROLLOUT_FORMAT,
            "scene_json_path": str(Path(scene_json_path).resolve()),
            "ppg": float(ppg),
            "greedy_ppg": float(greedy_ppg),
            "reward": float(reward),
            "decisions": [_decision_to_payload(decision) for decision in decision_list],
        }
        _atomic_torch_save(payload, trajectory_path)
        entry = StoredGrpoTrajectory(
            path=trajectory_path,
            scene_json_path=Path(scene_json_path).resolve(),
            ppg=float(ppg),
            greedy_ppg=float(greedy_ppg),
            reward=float(reward),
            decision_count=len(decision_list),
        )
        self._entries.append(entry)
        self._write_manifest()
        return entry

    def set_advantages(self, advantages: Sequence[float]) -> None:
        """为已落盘轨迹写入组相对 advantage，并刷新索引。"""
        if len(advantages) != len(self._entries):
            raise ValueError("GRPO trajectory/advantage count mismatch")
        self._entries = [
            StoredGrpoTrajectory(
                path=entry.path,
                scene_json_path=entry.scene_json_path,
                ppg=entry.ppg,
                greedy_ppg=entry.greedy_ppg,
                reward=entry.reward,
                decision_count=entry.decision_count,
                advantage=float(advantage),
            )
            for entry, advantage in zip(self._entries, advantages, strict=True)
        ]
        self._write_manifest()

    def iter_minibatches(
        self,
        minibatch_size: int,
    ) -> Iterator[tuple[list[GrpoDecision], torch.Tensor]]:
        """按随机化的轨迹顺序从磁盘产生一个个 CPU minibatch。"""
        if minibatch_size < 1:
            raise ValueError("GRPO minibatch_size must be >= 1")
        if not self._entries:
            raise ValueError("GRPO rollout store is empty")
        if any(entry.advantage is None for entry in self._entries):
            raise ValueError("GRPO rollout advantages have not been persisted")

        pending_decisions: list[GrpoDecision] = []
        pending_advantages: list[float] = []
        for entry_index in torch.randperm(len(self._entries)).tolist():
            entry = self._entries[entry_index]
            decisions = self._load_decisions(entry.path)
            for decision_index in torch.randperm(len(decisions)).tolist():
                decision = decisions[decision_index]
                pending_decisions.append(decision)
                pending_advantages.append(float(entry.advantage))
                if len(pending_decisions) == minibatch_size:
                    yield pending_decisions, torch.tensor(
                        pending_advantages,
                        dtype=torch.float32,
                    )
                    pending_decisions = []
                    pending_advantages = []
        if pending_decisions:
            yield pending_decisions, torch.tensor(
                pending_advantages,
                dtype=torch.float32,
            )

    def _load_decisions(self, path: Path) -> tuple[GrpoDecision, ...]:
        payload = safe_torch_load(path)
        if not isinstance(payload, Mapping):
            raise ValueError(f"GRPO trajectory payload must be a mapping: {path}")
        if int(payload.get("format", -1)) != GRPO_ROLLOUT_FORMAT:
            raise ValueError(f"unsupported GRPO trajectory format: {path}")
        raw_decisions = payload.get("decisions")
        if not isinstance(raw_decisions, list) or not raw_decisions:
            raise ValueError(f"GRPO trajectory has no decisions: {path}")
        decisions = tuple(_decision_from_payload(item) for item in raw_decisions)
        expected_count = self._entry_for_path(path).decision_count
        if len(decisions) != expected_count:
            raise ValueError(
                f"GRPO trajectory decision count mismatch: {path} "
                f"{len(decisions)} != {expected_count}"
            )
        return decisions

    def _entry_for_path(self, path: Path) -> StoredGrpoTrajectory:
        for entry in self._entries:
            if entry.path == path:
                return entry
        raise ValueError(f"GRPO trajectory is not indexed: {path}")

    def _write_manifest(self) -> None:
        payload = {
            "format": GRPO_ROLLOUT_FORMAT,
            "trajectories": [
                {
                    "path": str(entry.path.relative_to(self.iteration_dir)),
                    "scene_json_path": str(entry.scene_json_path),
                    "ppg": entry.ppg,
                    "greedy_ppg": entry.greedy_ppg,
                    "reward": entry.reward,
                    "decision_count": entry.decision_count,
                    "advantage": entry.advantage,
                }
                for entry in self._entries
            ],
        }
        temporary_path = self.manifest_path.with_name(
            f"{self.manifest_path.name}.tmp.{os.getpid()}"
        )
        try:
            temporary_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary_path.replace(self.manifest_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()


def _atomic_torch_save(payload: Mapping[str, object], path: Path) -> None:
    """按项目 compiled cache 的方式原子写入 tensor 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
