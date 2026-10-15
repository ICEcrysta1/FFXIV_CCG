"""GRPO 自回归采样与训练决策记录。"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import torch

from common.policy.model.repetition import apply_repetition_penalty
from common.policy.replay import AutoregressiveReplayConfig
from scripts.autoregressive_replay.replay import (
    AutoregressiveReplay,
    AutoregressiveReplaySession,
    NoLegalActionError,
    _apply_top_p,
)

from .constants import LOG_PROB_EPSILON
from .storage import GrpoDecision


class _TrainableRolloutReplay(AutoregressiveReplay):
    """复用生产回放时序，只把策略采样换成共享训练模型。"""

    def __init__(
        self,
        config,
        *,
        record_decisions: bool,
        session: AutoregressiveReplaySession,
    ):
        self.decisions: list[GrpoDecision] = []
        self._record_decisions = bool(record_decisions)
        super().__init__(config, session=session)

    def _reset_for_trajectory(self, *, initial_timestamp: float = 0.0):
        """重置共享 replay session，并丢弃上一条轨迹的决策记录。"""
        self.decisions.clear()
        return super()._reset_for_trajectory(initial_timestamp=initial_timestamp)

    def _score_row(
        self,
        batch,
        action_keys: list[str],
        *,
        gcd_step: int,
    ):
        """按生产 replay 的合法性规则采样，并保留训练所需行为概率。"""
        raw_logits = self.backend.raw_logits(batch, action_keys)
        logits = apply_repetition_penalty(
            raw_logits,
            batch,
            self.backend.repetition,
        )[0].float()

        legal_mask = batch["action_legal_mask"][0].bool()
        legal_indices = legal_mask.nonzero(as_tuple=True)[0]
        if legal_indices.numel() == 0:
            raise NoLegalActionError("live state has no legal action")

        legal_logits = logits.masked_fill(~legal_mask, float("-inf"))
        order = torch.argsort(legal_logits, descending=True)
        if self.config.temperature == 0.0:
            probabilities = torch.softmax(legal_logits, dim=-1)
            selected_index = int(order[0].item())
        else:
            probabilities = torch.softmax(
                legal_logits / self.config.temperature,
                dim=-1,
            )
            probabilities = _apply_top_p(
                probabilities,
                order,
                self.config.top_p,
            )
            selected_index = int(torch.multinomial(
                probabilities, 1, generator=getattr(self, "_sampling_generator", None),
            ).item())

        selected_probability = probabilities[selected_index].clamp_min(
            LOG_PROB_EPSILON
        )
        if self._record_decisions:
            recorded_batch = _detach_batch_to_cpu(batch)
            recorded_batch["action_legal_mask"] = legal_mask.unsqueeze(0).cpu()
            self.decisions.append(
                GrpoDecision(
                    batch=recorded_batch,
                    context_metadata=self.batcher.context_metadata,
                    action_keys=tuple(str(key) for key in action_keys),
                    action_index=selected_index,
                    old_logprob=float(torch.log(selected_probability).item()),
                )
            )

        return self._make_replay_row(
            gcd_step=gcd_step,
            action_keys=action_keys,
            logits=logits,
            probabilities=probabilities,
            legal_mask=legal_mask,
            order=order,
            selected_index=selected_index,
        )

    def _make_replay_row(
        self,
        *,
        gcd_step: int,
        action_keys: list[str],
        logits: torch.Tensor,
        probabilities: torch.Tensor,
        legal_mask: torch.Tensor,
        order: torch.Tensor,
        selected_index: int,
    ):
        # 延迟导入避免复制 ReplayRow 的生产数据结构定义。
        from scripts.autoregressive_replay.replay import ReplayRow

        return ReplayRow(
            gcd_step=gcd_step,
            action_key=action_keys[selected_index],
            probability=float(probabilities[selected_index].item()),
            top_actions=tuple(
                (
                    action_keys[index],
                    float(logits[index].item()),
                    float(probabilities[index].item()),
                    bool(legal_mask[index].item()),
                )
                for index in order[: self.config.top_k].tolist()
            ),
        )


def _detach_batch_to_cpu(batch: Mapping[str, object]) -> dict[str, object]:
    """把 rollout 输入脱离 GPU/推理 tensor，等待后续 minibatch 重组。

    ``torch.inference_mode`` 产生的 tensor 即使调用 ``clone`` 仍会保留
    inference 标记；这里显式关闭该模式，保证后续 GRPO 训练前向可以为
    collated batch 构建正常的 autograd 图。
    """
    with torch.inference_mode(False):
        return {
            key: value.detach().cpu().clone()
            if isinstance(value, torch.Tensor)
            else deepcopy(value)
            for key, value in batch.items()
        }


def _ppg_from_result(result) -> float:
    """读取状态机执行轨迹的 PPG，禁止退回动作预演或单步威力。"""
    if result.ppg is None:
        raise RuntimeError("GRPO rollout did not produce PPG")
    return float(result.ppg)


def _run_scene_rollout(
    replay_config: AutoregressiveReplayConfig,
    *,
    scene_json_path: Path,
    temperature: float,
    record_decisions: bool,
    session: AutoregressiveReplaySession,
    sampling_seed: int | None = None,
):
    config = replace(
        replay_config,
        scene_json_path=Path(scene_json_path),
        temperature=float(temperature),
    )
    replay = _TrainableRolloutReplay(
        config,
        record_decisions=record_decisions,
        session=session,
    )
    if sampling_seed is not None:
        # 每条轨迹的随机流由 YAML seed 与稳定任务编号决定，不依赖线程调度。
        replay._sampling_generator = torch.Generator(device=session.device)
        replay._sampling_generator.manual_seed(sampling_seed)
    # 自回归阶段只负责执行状态机、采样和记录标注；禁止构建任何反向图。
    # backend.raw_logits 本身也使用 no_grad，这里再用 inference_mode
    # 覆盖整条轨迹，避免 16 条样本把激活留在显存中。
    with torch.inference_mode():
        result = replay.run()
    decisions = tuple(replay.decisions)
    return result, decisions
