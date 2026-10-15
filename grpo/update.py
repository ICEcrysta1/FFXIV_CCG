"""GRPO 优势、策略损失与磁盘轨迹的流式参数更新。"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from common.policy.model.repetition import apply_repetition_penalty, prepare_repetition_penalty
from common.pytorch.runtime import autocast_context, move_batch
from common.training.metrics import MetricAccumulator
from common.training.tensorboard import TensorBoardMonitor, write_scalar_metrics
from scripts.autoregressive_replay.parallel import collate_live_batches

from .config import GrpoConfig
from .constants import ADVANTAGE_EPSILON, LOG_PROB_EPSILON
from .storage import GrpoDecision, GrpoRolloutStore


def compute_baseline_relative_advantages(
    reward_deltas: Sequence[Sequence[float]],
    *,
    scale_floor_ratio: float = 0.1,
    advantage_clip: float = 5.0,
) -> tuple[torch.Tensor, ...]:
    """按贪心零点计算优势，保留高于/低于基准的正负号。

    组内标准差只负责反映相对离散程度；当采样结果几乎相同时，使用
    奖励绝对量级的相对下限并限制最终优势，避免小方差把梯度放大到失真。
    """
    if not reward_deltas:
        raise ValueError("GRPO requires at least one prompt group")
    if scale_floor_ratio < 0.0:
        raise ValueError("scale_floor_ratio must be >= 0")
    if advantage_clip <= 0.0:
        raise ValueError("advantage_clip must be > 0")
    result: list[torch.Tensor] = []
    for group in reward_deltas:
        if not group:
            raise ValueError("GRPO reward group must not be empty")
        values = torch.tensor(tuple(float(value) for value in group), dtype=torch.float32)
        std_scale = values.std(unbiased=False)
        relative_scale_floor = values.abs().mean() * float(scale_floor_ratio)
        scale = torch.maximum(std_scale, relative_scale_floor).clamp_min(
            ADVANTAGE_EPSILON
        )
        result.append((values / scale).clamp(-float(advantage_clip), float(advantage_clip)))
    return tuple(result)


def collate_grpo_decisions(
    decisions: Sequence[GrpoDecision],
) -> dict[str, object]:
    return collate_live_batches([decision.batch for decision in decisions])


def _top_p_probabilities_for_actions(
    probabilities: torch.Tensor,
    order: torch.Tensor,
    selected_indices: torch.Tensor,
    top_p: float,
) -> torch.Tensor:
    """应用 rollout 的 top-p，并强制保留已采样 action 以定义 ratio。"""
    if top_p >= 1.0:
        return probabilities
    sorted_probabilities = probabilities.gather(1, order)
    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
    keep_sorted = torch.ones_like(sorted_probabilities, dtype=torch.bool)
    keep_sorted[:, 1:] = cumulative_probabilities[:, :-1] < top_p
    keep = torch.zeros_like(keep_sorted)
    keep.scatter_(1, order, keep_sorted)
    keep.scatter_(1, selected_indices.unsqueeze(1), True)
    filtered = probabilities * keep
    return filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(LOG_PROB_EPSILON)


def compute_grpo_loss(
    model,
    batch: dict[str, object],
    *,
    action_indices: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    grpo: GrpoConfig,
    repetition,
    device: torch.device,
    precision: str,
    diagnostics: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """计算 token-level clipped GRPO loss，指标保留为脱离计算图的设备 tensor。"""
    with autocast_context(device, precision):
        model_logits = model(batch)["logits"]
    logits = apply_repetition_penalty(model_logits, batch, repetition).float()
    legal_mask = batch["action_legal_mask"].bool()
    masked_logits = logits.masked_fill(~legal_mask, float("-inf"))
    scaled_logits = masked_logits / grpo.temperature
    probabilities = torch.softmax(scaled_logits, dim=-1)
    order = torch.argsort(masked_logits, descending=True)
    probabilities = _top_p_probabilities_for_actions(
        probabilities,
        order,
        action_indices,
        grpo.top_p,
    )
    selected = probabilities.gather(1, action_indices.unsqueeze(1)).squeeze(1)
    new_logprobs = torch.log(selected.clamp_min(LOG_PROB_EPSILON))

    ratio = torch.exp(new_logprobs - old_logprobs)
    clipped_ratio = ratio.clamp(1.0 - grpo.clip_low, 1.0 + grpo.clip_high)
    unclipped = ratio * advantages
    clipped = clipped_ratio * advantages
    surrogate = torch.minimum(unclipped, clipped)
    policy_loss = -surrogate.mean()

    log_ratio_ref_to_policy = old_logprobs - new_logprobs
    kl = torch.exp(log_ratio_ref_to_policy) - log_ratio_ref_to_policy - 1.0
    loss = policy_loss + grpo.kl_coefficient * kl.mean()

    positive_probabilities = probabilities.clamp_min(LOG_PROB_EPSILON)
    entropy = -(
        probabilities * torch.log(positive_probabilities)
    ).sum(dim=-1)
    clipped_fraction = (
        (ratio < 1.0 - grpo.clip_low) | (ratio > 1.0 + grpo.clip_high)
    ).float()
    if diagnostics is not None:
        # 复用损失的同一次前向；行为分布只保留合法动作，所有数据均脱离计算图。
        diagnostics.update({
            "model_logits": model_logits.detach(),
            "policy/legal_logits": logits.detach()[legal_mask],
            "policy/legal_probabilities": probabilities.detach()[legal_mask],
            "policy/selected_probability": selected.detach(),
            "policy/ratio": ratio.detach(),
            "policy/kl": kl.detach(),
            "policy/entropy": entropy.detach(),
            "rollout/advantages": advantages.detach(),
            "rollout/old_logprob": old_logprobs.detach(),
            "rollout/action_index": action_indices.detach(),
        })
    return loss, {
        "loss": loss.detach(),
        "policy_loss": policy_loss.detach(),
        "kl": kl.detach().mean(),
        "entropy": entropy.detach().mean(),
        "clip_fraction": clipped_fraction.detach().mean(),
        "mean_ratio": ratio.detach().mean(),
    }


def _update_policy(
    model,
    optimizer,
    scheduler,
    store: GrpoRolloutStore,
    *,
    grpo: GrpoConfig,
    repetition,
    device: torch.device,
    precision: str,
    tensorboard_writer=None,
    tensorboard_monitor: TensorBoardMonitor | None = None,
    tensorboard_log_every_steps: int = 50,
    global_step_offset: int = 0,
) -> dict[str, float]:
    """从磁盘轨迹逐批读取决策并执行 GRPO 的 inner updates。"""
    if store.total_decisions < 1:
        raise ValueError("GRPO rollout produced no trainable decisions")

    totals = MetricAccumulator(
        ("loss", "policy_loss", "kl", "entropy", "clip_fraction", "mean_ratio"),
        device=device,
    )
    update_count = 0
    interval_metrics = (
        MetricAccumulator((*totals.names, "grad_norm"), device=device)
        if tensorboard_writer is not None else None
    )
    interval_learning_rate_total = 0.0

    def flush_interval(global_step: int) -> None:
        nonlocal interval_metrics, interval_learning_rate_total
        if interval_metrics is None or interval_metrics.weight == 0:
            return
        write_scalar_metrics(
            tensorboard_writer,
            {
                **interval_metrics.mean(),
                "learning_rate": interval_learning_rate_total / interval_metrics.weight,
                "next_learning_rate": optimizer.param_groups[0]["lr"],
            },
            prefix="grpo/update",
            global_step=global_step,
        )
        interval_metrics = MetricAccumulator((*totals.names, "grad_norm"), device=device)
        interval_learning_rate_total = 0.0

    # rollout 行为策略和训练前向必须使用同一确定性分布；GRPO 不需要
    # BC 阶段的 dropout 噪声，否则零优势组会被 KL 项的随机 dropout 误更新。
    model.eval()
    def update_minibatch(
        selected_decisions: list[GrpoDecision],
        minibatch_advantages: torch.Tensor,
    ) -> None:
        nonlocal update_count, interval_learning_rate_total
        global_step = global_step_offset + update_count + 1
        batch = move_batch(
            prepare_repetition_penalty(
                collate_grpo_decisions(selected_decisions), repetition,
            ),
            device,
            non_blocking=device.type == "cuda",
        )
        action_indices = torch.tensor(
            [decision.action_index for decision in selected_decisions],
            dtype=torch.long,
            device=device,
        )
        old_logprobs = torch.tensor(
            [decision.old_logprob for decision in selected_decisions],
            dtype=torch.float32,
            device=device,
        )
        minibatch_advantages = minibatch_advantages.to(device=device)
        diagnostics = (
            {} if tensorboard_monitor is not None
            and tensorboard_monitor.should_log_histograms(global_step) else None
        )
        if tensorboard_monitor is not None:
            tensorboard_monitor.log_graph(batch)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = compute_grpo_loss(
            model,
            batch,
            action_indices=action_indices,
            old_logprobs=old_logprobs,
            advantages=minibatch_advantages,
            grpo=grpo,
            repetition=repetition,
            device=device,
            precision=precision,
            **({"diagnostics": diagnostics} if diagnostics is not None else {}),
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if tensorboard_monitor is not None:
            tensorboard_monitor.log_gradients(global_step, grad_norm=grad_norm)
        if diagnostics is not None and tensorboard_monitor is not None:
            model_logits = diagnostics.pop("model_logits", None)
            if model_logits is not None:
                tensorboard_monitor.log_step(
                    model_logits, global_step=global_step, prefix="grpo/model",
                )
            tensorboard_monitor.log_distributions(
                diagnostics, global_step=global_step, prefix="grpo/update",
            )
        # 当前更新使用的是 scheduler 推进之前的学习率。
        learning_rate = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        minibatch_size = len(selected_decisions)
        totals.update(metrics, weight=minibatch_size)
        if interval_metrics is not None:
            interval_metrics.update({**metrics, "grad_norm": grad_norm}, weight=minibatch_size)
            interval_learning_rate_total += learning_rate * minibatch_size
        update_count += 1
        if global_step == 1 or global_step % tensorboard_log_every_steps == 0:
            flush_interval(global_step)

    for _ in range(grpo.inner_updates):
        for selected_decisions, minibatch_advantages in store.iter_minibatches(grpo.minibatch_size):
            update_minibatch(selected_decisions, minibatch_advantages)
    flush_interval(global_step_offset + update_count)
    return totals.mean() | {"optimizer_updates": float(update_count)}


def _weighted_advantage_mean(
    store: GrpoRolloutStore,
    advantages: Sequence[float],
) -> float:
    """按决策数量计算与旧内存实现一致的 advantage 均值。"""
    if len(advantages) != len(store.entries) or store.total_decisions < 1:
        raise ValueError("GRPO advantage summary is not aligned with rollout store")
    weighted_total = sum(
        float(advantage) * entry.decision_count
        for entry, advantage in zip(store.entries, advantages, strict=True)
    )
    return weighted_total / store.total_decisions


def _weighted_advantage_std(
    store: GrpoRolloutStore,
    advantages: Sequence[float],
) -> float:
    """按决策数量计算磁盘流式 rollout 的 advantage 标准差。"""
    mean = _weighted_advantage_mean(store, advantages)
    weighted_squared = sum(
        (float(advantage) - mean) ** 2 * entry.decision_count
        for entry, advantage in zip(store.entries, advantages, strict=True)
    )
    return (weighted_squared / store.total_decisions) ** 0.5
