"""公共连续重复动作软惩罚策略。"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from functools import lru_cache
from typing import Any


@dataclass(frozen=True)
class RepetitionConfig:
    """控制连续重复候选动作的 logits 软惩罚。"""

    mode: str = "none"
    skills: tuple[str, ...] = ()
    penalty: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.mode in {"whitelist", "blacklist"} and self.penalty > 0.0


def parse_repetition_config(raw: object) -> RepetitionConfig:
    """从 YAML 或 checkpoint 的 mapping 解析重复动作策略。"""
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("training.repetition must be a mapping")
    mode = str(raw.get("mode", "none")).strip().lower()
    if mode not in {"none", "whitelist", "blacklist"}:
        raise ValueError("training.repetition.mode must be none, whitelist, or blacklist")
    skills_raw = raw.get("skills", ()) or ()
    if not isinstance(skills_raw, (list, tuple)):
        raise ValueError("training.repetition.skills must be a list")
    skills = tuple(str(skill).strip() for skill in skills_raw if str(skill).strip())
    penalty = float(raw.get("penalty", 0.0))
    if penalty < 0.0:
        raise ValueError("training.repetition.penalty must be >= 0")
    if mode in {"whitelist", "blacklist"} and not skills:
        raise ValueError(
            "training.repetition.skills must not be empty for whitelist or blacklist mode"
        )
    return RepetitionConfig(mode=mode, skills=skills, penalty=penalty)


def repetition_config_from_checkpoint(checkpoint: Mapping[str, object]) -> RepetitionConfig:
    """从 checkpoint 的运行配置恢复重复动作策略；旧 checkpoint 默认关闭。"""
    run_config = checkpoint.get("run_config", {})
    if not isinstance(run_config, Mapping):
        return RepetitionConfig()
    return parse_repetition_config(run_config.get("repetition", {}))


@lru_cache(maxsize=128)
def _candidate_penalty_indices(
    candidates: tuple[str, ...],
    mode: str,
    skills: tuple[str, ...],
):
    """缓存动作到候选索引的静态映射；重排导致缓存未命中时也不分配 tensor。"""
    listed_skills = frozenset(skills)
    positions: dict[str, list[int]] = {}
    for index, key in enumerate(candidates):
        penalized = (
            key not in listed_skills if mode == "whitelist" else key in listed_skills
        )
        if key != "ogcd_wait" and penalized:
            positions.setdefault(key, []).append(index)
    return {key: tuple(indices) for key, indices in positions.items()}


def _build_repetition_mask_cpu(batch: dict[str, Any], config: RepetitionConfig):
    """在输入准备阶段解析最近非 wait 动作，候选匹配复用静态查找表。"""
    import torch

    shape = batch["candidate_skill_ids"].shape[:2]
    candidates = batch.get("candidate_action_keys")
    histories = batch.get("history_action_keys")
    if (
        not config.enabled
        or not isinstance(candidates, (list, tuple))
        or not isinstance(histories, (list, tuple))
    ):
        return torch.zeros(shape, dtype=torch.bool, device="cpu")
    if len(histories) != len(candidates):
        raise ValueError("history_action_keys and candidate_action_keys batch lengths differ")
    if len(candidates) != shape[0] or any(len(row) != shape[1] for row in candidates):
        raise ValueError("candidate_action_keys shape must match candidate_skill_ids")
    row_indices = []
    column_indices = []
    for row_index, (history, candidate_keys) in enumerate(zip(histories, candidates)):
        last_action = next(
            (str(key) for key in reversed(history) if str(key) != "ogcd_wait"), None,
        )
        if last_action is None:
            continue
        by_action = _candidate_penalty_indices(
            tuple(str(key) for key in candidate_keys), config.mode, config.skills,
        )
        indices = by_action.get(last_action, ())
        row_indices.extend([row_index] * len(indices))
        column_indices.extend(indices)
    mask = torch.zeros(shape, dtype=torch.bool, device="cpu")
    if column_indices:
        mask[row_indices, column_indices] = True
    return mask


def prepare_repetition_penalty(
    batch: dict[str, Any],
    config: RepetitionConfig,
) -> dict[str, Any]:
    """为完成截断/重排的 batch 缓存惩罚 mask，随后与其他输入一起搬运。

    mask 是当前动作元数据和策略的快照；改变输入后应重新调用本函数。
    不修改调用方 batch，也不将运行期 mask 写入 checkpoint 或数据缓存。
    """
    if not config.enabled:
        return batch
    return {
        **batch,
        "repetition_penalty_mask": _build_repetition_mask_cpu(batch, config),
        "repetition_penalty_config": config,
    }


def build_repetition_penalty_mask(
    batch: dict[str, Any],
    config: RepetitionConfig,
    *,
    device,
):
    """返回 `[batch, candidate]` 的连续重复候选掩码。"""
    import torch

    mask = batch.get("repetition_penalty_mask")
    if isinstance(mask, torch.Tensor) and batch.get("repetition_penalty_config") == config:
        if mask.shape != batch["candidate_skill_ids"].shape or mask.dtype != torch.bool:
            raise ValueError(
                "repetition_penalty_mask must be boolean and match candidate_skill_ids"
            )
        return mask.to(device=device)
    # 独立推理/旧调用方仍可直接传字符串元数据，共用同一策略实现。
    return _build_repetition_mask_cpu(batch, config).to(device=device)


def apply_repetition_penalty(logits, batch: dict[str, Any], config: RepetitionConfig):
    """对不允许连续重复的候选动作降低 logits，不改变其合法性。"""
    if not config.enabled:
        return logits
    mask = build_repetition_penalty_mask(batch, config, device=logits.device)
    return logits - mask.to(dtype=logits.dtype) * config.penalty
