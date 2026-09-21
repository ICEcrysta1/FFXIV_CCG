"""基于真实状态机 rollout 的 GRPO 后训练。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import logging
from pathlib import Path
import random

import torch

from common.torch_serialization import safe_torch_load
from common.torch_runtime import autocast_context, move_batch
from common.policy.config import (
    resolve_policy_cache_dir,
    resolve_policy_grpo_dir,
    validate_policy_model_variant,
)
from common.policy.data import DataSpec
from common.policy.model.repetition import (
    apply_repetition_penalty,
    prepare_repetition_penalty,
)
from common.policy.replay import AutoregressiveReplayConfig
from common.training.metrics import MetricAccumulator
from scripts.autoregressive_replay.backends import PyTorchPolicyBackend
from scripts.autoregressive_replay.replay import (
    NoLegalCandidateError,
    AutoregressiveReplay,
    AutoregressiveReplaySession,
    ReplayCacheStore,
    _apply_top_p,
)
from .config import GrpoConfig, GrpoRunConfig
from .storage import GrpoDecision, GrpoRolloutStore


logger = logging.getLogger(__name__)
ADVANTAGE_EPSILON = 1e-4
LOG_PROB_EPSILON = 1e-12


@dataclass(frozen=True)
class GrpoTrajectory:
    """一条已落盘状态机执行轨迹及其相对贪心基准的奖励。"""

    scene_json_path: Path
    decision_path: Path
    decision_count: int
    ppg: float
    greedy_ppg: float
    reward: float


class _TrainableRolloutReplay(AutoregressiveReplay):
    """复用生产回放时序，只把策略采样换成共享训练模型。"""

    def __init__(
        self,
        config,
        *,
        backend,
        record_decisions: bool,
        session: AutoregressiveReplaySession | None = None,
    ):
        self.decisions: list[GrpoDecision] = []
        self._record_decisions = bool(record_decisions)
        super().__init__(config, backend=backend, session=session)

    def _reset_for_trajectory(self, *, initial_timestamp: float = 0.0):
        """重置共享 replay session，并丢弃上一条轨迹的决策记录。"""
        self.decisions.clear()
        return super()._reset_for_trajectory(initial_timestamp=initial_timestamp)

    def _score_row(
        self,
        batch,
        candidate_keys: list[str],
        *,
        gcd_step: int,
    ):
        """按生产 replay 的合法性规则采样，并保留训练所需行为概率。"""
        raw_logits = self.backend.raw_logits(batch, candidate_keys)
        logits = apply_repetition_penalty(
            raw_logits,
            batch,
            self.backend.repetition,
        )[0].float()

        legal_mask = batch["candidate_legal_mask"][0].bool()
        legal_indices = legal_mask.nonzero(as_tuple=True)[0]
        if legal_indices.numel() == 0:
            raise NoLegalCandidateError("live state has no legal candidate")

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
            selected_index = int(torch.multinomial(probabilities, 1).item())

        selected_probability = probabilities[selected_index].clamp_min(
            LOG_PROB_EPSILON
        )
        if self._record_decisions:
            recorded_batch = _detach_batch_to_cpu(batch)
            recorded_batch["candidate_legal_mask"] = legal_mask.unsqueeze(0).cpu()
            self.decisions.append(
                GrpoDecision(
                    batch=recorded_batch,
                    candidate_keys=tuple(str(key) for key in candidate_keys),
                    action_index=selected_index,
                    old_logprob=float(torch.log(selected_probability).item()),
                )
            )

        return self._make_replay_row(
            gcd_step=gcd_step,
            candidate_keys=candidate_keys,
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
        candidate_keys: list[str],
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
            action_key=candidate_keys[selected_index],
            probability=float(probabilities[selected_index].item()),
            top_candidates=tuple(
                (
                    candidate_keys[index],
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


def _pad_sequence_field(
    samples: Sequence[dict[str, object]],
    key: str,
    *,
    pad_value=0,
) -> torch.Tensor:
    """把不同历史/scene 长度的单样本 live batch 进行右侧 padding。"""
    values = [sample[key][0] for sample in samples]
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise TypeError(f"GRPO batch field is not tensor: {key}")
    tensors = [value for value in values if isinstance(value, torch.Tensor)]
    if not tensors:
        return torch.empty((0, 0), dtype=torch.float32)
    return torch.nn.utils.rnn.pad_sequence(
        tensors,
        batch_first=True,
        padding_value=pad_value,
    )


def collate_grpo_decisions(
    decisions: Sequence[GrpoDecision],
) -> dict[str, object]:
    """组合变长 live 输入；候选维度必须保持 checkpoint 契约的固定顺序。"""
    if not decisions:
        raise ValueError("cannot collate an empty GRPO decision batch")
    samples = [decision.batch for decision in decisions]
    batch: dict[str, object] = {}
    for key in (
        "scene_vectors",
        "scene_types",
        "scene_mask",
        "history_skill_ids",
        "history_skill_features",
        "history_state_vectors",
        "history_mask",
    ):
        batch[key] = _pad_sequence_field(samples, key)
    batch["history_state_null_mask"] = _pad_sequence_field(
        samples,
        "history_state_null_mask",
        pad_value=True,
    )
    for key in (
        "candidate_skill_ids",
        "candidate_skill_features",
        "candidate_state_vectors",
        "candidate_state_null_mask",
        "candidate_legal_mask",
    ):
        values = [sample[key] for sample in samples]
        if not all(isinstance(value, torch.Tensor) for value in values):
            raise TypeError(f"GRPO candidate field is not tensor: {key}")
        batch[key] = torch.cat(
            [value for value in values if isinstance(value, torch.Tensor)],
            dim=0,
        )

    batch["history_action_keys"] = [
        deepcopy(sample["history_action_keys"][0]) for sample in samples
    ]
    batch["candidate_action_keys"] = [
        deepcopy(sample["candidate_action_keys"][0]) for sample in samples
    ]
    return batch


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
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """计算 token-level clipped GRPO loss，指标保留为脱离计算图的设备 tensor。"""
    with autocast_context(device, precision):
        logits = model(batch)["logits"]
    logits = apply_repetition_penalty(logits, batch, repetition).float()
    legal_mask = batch["candidate_legal_mask"].bool()
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
    return loss, {
        "loss": loss.detach(),
        "policy_loss": policy_loss.detach(),
        "kl": kl.detach().mean(),
        "entropy": entropy.detach().mean(),
        "clip_fraction": clipped_fraction.detach().mean(),
        "mean_ratio": ratio.detach().mean(),
    }


def _ppg_from_result(result) -> float:
    """读取状态机执行轨迹的 PPG，禁止退回候选预演或单步威力。"""
    if result.ppg is None:
        raise RuntimeError("GRPO rollout did not produce PPG")
    return float(result.ppg)


def _run_scene_rollout(
    replay_config: AutoregressiveReplayConfig,
    *,
    scene_json_path: Path,
    backend,
    temperature: float,
    record_decisions: bool,
    session: AutoregressiveReplaySession | None = None,
):
    config = replace(
        replay_config,
        scene_json_path=Path(scene_json_path),
        temperature=float(temperature),
    )
    replay: _TrainableRolloutReplay | None = None
    try:
        replay = _TrainableRolloutReplay(
            config,
            backend=backend,
            record_decisions=record_decisions,
            session=session,
        )
        # 自回归阶段只负责执行状态机、采样和记录标注；禁止构建任何反向图。
        # backend.raw_logits 本身也使用 no_grad，这里再用 inference_mode
        # 覆盖整条轨迹，避免 16 条样本把激活留在显存中。
        with torch.inference_mode():
            result = replay.run()
        decisions = tuple(replay.decisions)
    except BaseException:
        if session is not None:
            session.close()
        raise
    finally:
        if replay is not None:
            replay.close()
    return result, decisions


def _update_policy(
    model,
    optimizer,
    scheduler,
    decisions: Sequence[GrpoDecision] | GrpoRolloutStore,
    advantages: torch.Tensor | None,
    *,
    grpo: GrpoConfig,
    repetition,
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    """对当前 rollout buffer 执行 GRPO 的 inner updates。"""
    if isinstance(decisions, GrpoRolloutStore):
        if advantages is not None:
            raise ValueError("disk-backed GRPO rollouts must provide advantages on disk")
        decision_count = decisions.total_decisions
        if decision_count < 1:
            raise ValueError("GRPO rollout produced no trainable decisions")
    else:
        if not decisions:
            raise ValueError("GRPO rollout produced no trainable decisions")
        if advantages is None:
            raise ValueError("in-memory GRPO decisions require advantages")
        if len(decisions) != int(advantages.shape[0]):
            raise ValueError("GRPO decisions and advantages length mismatch")
        decision_count = len(decisions)

    totals = MetricAccumulator(
        ("loss", "policy_loss", "kl", "entropy", "clip_fraction", "mean_ratio"),
        device=device,
    )
    update_count = 0
    # rollout 行为策略和训练前向必须使用同一确定性分布；GRPO 不需要
    # BC 阶段的 dropout 噪声，否则零优势组会被 KL 项的随机 dropout 误更新。
    model.eval()
    def update_minibatch(
        selected_decisions: list[GrpoDecision],
        minibatch_advantages: torch.Tensor,
    ) -> None:
        nonlocal update_count
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
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        totals.update(metrics)
        update_count += 1

    for _ in range(grpo.inner_updates):
        if isinstance(decisions, GrpoRolloutStore):
            minibatches = decisions.iter_minibatches(grpo.minibatch_size)
            for selected_decisions, minibatch_advantages in minibatches:
                update_minibatch(selected_decisions, minibatch_advantages)
            continue

        permutation = torch.randperm(decision_count)
        for start in range(0, decision_count, grpo.minibatch_size):
            indices = permutation[start : start + grpo.minibatch_size]
            selected_decisions = [decisions[int(index)] for index in indices.tolist()]
            assert advantages is not None
            minibatch_advantages = advantages[indices]
            update_minibatch(selected_decisions, minibatch_advantages)
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


def _save_grpo_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scheduler,
    iteration: int,
    config: GrpoRunConfig,
    grpo: GrpoConfig,
    data_spec: DataSpec,
    input_contract,
    precision: str,
    metrics: Mapping[str, float],
) -> None:
    """保存可被现有 replay/ONNX 流程读取的自描述 GRPO checkpoint。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(iteration),
        "grpo_iteration": int(iteration),
        "grpo_checkpoint": True,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "model_config": asdict(config.model),
        "data_spec": asdict(data_spec),
        "job_tag": data_spec.job_tag,
        "model_variant": config.model_variant,
        "input_contract": input_contract.to_dict(),
        "training_precision": precision,
        "run_config": asdict(config),
        "grpo_config": asdict(grpo),
        "metrics": dict(metrics),
        "rng_state": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        },
    }
    torch.save(payload, path)


def _restore_grpo_rollback_state(
    *,
    checkpoint_path: Path | None,
    initial_model_state: Mapping[str, object],
    initial_optimizer_state: Mapping[str, object] | None,
    initial_scheduler_state: Mapping[str, object] | None,
    model,
    optimizer,
    scheduler,
) -> None:
    """从上一轮磁盘 checkpoint 恢复回滚状态，避免常驻一份 GPU 权重副本。"""
    if checkpoint_path is None:
        model.load_state_dict(initial_model_state, strict=True)
        if initial_optimizer_state is not None:
            optimizer.load_state_dict(initial_optimizer_state)
        if scheduler is not None and initial_scheduler_state is not None:
            scheduler.load_state_dict(initial_scheduler_state)
        return

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"GRPO rollback checkpoint not found: {checkpoint_path}"
        )
    payload = safe_torch_load(checkpoint_path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"GRPO rollback checkpoint must be a mapping: {checkpoint_path}")
    model_state = payload.get("model_state_dict")
    optimizer_state = payload.get("optimizer_state_dict")
    if not isinstance(model_state, Mapping) or not isinstance(optimizer_state, Mapping):
        raise ValueError(
            "GRPO rollback checkpoint is missing model/optimizer state: "
            f"{checkpoint_path}"
        )
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    scheduler_state = payload.get("scheduler_state_dict")
    if scheduler is not None and isinstance(scheduler_state, Mapping):
        scheduler.load_state_dict(scheduler_state)


def run_grpo_training(
    config: GrpoRunConfig,
    *,
    grpo: GrpoConfig,
    checkpoint_path: Path,
    raw_paths: Sequence[Path],
    output_dir: Path | None = None,
    device_name: str = "cuda",
    precision: str | None = None,
) -> dict[str, object]:
    """从 BC checkpoint 进入独立 GRPO 后训练。"""
    if not raw_paths:
        raise FileNotFoundError("GRPO requires at least one real training scene")
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"GRPO checkpoint not found: {checkpoint_path}")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by GRPO, but torch.cuda.is_available() is false")
    device = torch.device(device_name)
    resolved_precision = str(precision or config.precision).strip().lower()
    output_path = (
        Path(output_dir)
        if output_dir is not None
        else config.output_dir.parent / f"{config.output_dir.name}_grpo"
    ).resolve()

    # backend 与 replay session 只创建一次：模型、SidecarHost 和 compiled cache
    # 均跨场景/轨迹复用，单条轨迹开始时由 session.reset() 恢复初始状态。
    backend = PyTorchPolicyBackend(
        checkpoint_path,
        device=device_name,
        use_kv_cache=False,
        precision=resolved_precision,
    )
    data_spec = backend.data_spec
    input_contract = backend.input_contract
    model = backend.model
    if config.model_variant is not None:
        validate_policy_model_variant(
            backend.checkpoint,
            config.model_variant,
            artifact_name="GRPO checkpoint",
        )
    model_config = model.config
    if asdict(model_config) != asdict(config.model):
        raise ValueError(
            "GRPO config model does not match checkpoint model; "
            "use the same training YAML used to produce the checkpoint"
        )

    scenes = [Path(path).resolve() for path in raw_paths]
    if any(not path.is_file() for path in scenes):
        missing = [str(path) for path in scenes if not path.is_file()]
        raise FileNotFoundError("GRPO scene files not found: " + ", ".join(missing))
    random.Random(config.seed).shuffle(scenes)

    replay_config = AutoregressiveReplayConfig(
        checkpoint_path=checkpoint_path,
        output_path=output_path / "rollout_unused.md",
        scene_json_path=scenes[0],
        cache_dir=resolve_policy_cache_dir(data_spec.job_tag),
        model_history_capacity=config.model.history_capacity,
        cache_shard_size=config.compiled_cache_shard_size,
        cache_max_shards=config.compiled_cache_max_shards,
        scene_mode="cache",
        scene_sample_index=0,
        max_steps=None,
        max_gcds=None,
        max_duration_seconds=grpo.max_duration_seconds,
        top_k=data_spec.num_candidates,
        top_p=grpo.top_p,
        temperature=grpo.temperature,
        max_history=config.model.history_capacity,
        device=device_name,
        job_tag=data_spec.job_tag,
        model_variant=config.model_variant,
        initial_action=None,
        use_kv_cache=False,
        backend="pytorch",
        policy_precision=resolved_precision,
    )
    replay_cache_store = ReplayCacheStore(max_shards=config.compiled_cache_max_shards)
    replay_session = AutoregressiveReplaySession(
        replay_config,
        backend=backend,
        cache_store=replay_cache_store,
    )
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=grpo.learning_rate,
            weight_decay=grpo.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=(
                lambda step: 1.0
                if grpo.warmup_steps == 0
                else min(1.0, float(step + 1) / float(grpo.warmup_steps))
            ),
        )

        output_path.mkdir(parents=True, exist_ok=True)
        best_greedy_ppg = float("-inf")
        best_metrics: dict[str, float] = {}
        iteration_metrics: list[dict[str, float]] = []
        rollout_root = resolve_policy_grpo_dir(data_spec.job_tag)

        for iteration in range(1, grpo.max_iterations + 1):
            selected_scenes = [
                scenes[
                    ((iteration - 1) * grpo.prompt_batch_size + offset) % len(scenes)
                ]
                for offset in range(grpo.prompt_batch_size)
            ]
            model.eval()
            trajectories: list[GrpoTrajectory] = []
            group_trajectories: list[list[GrpoTrajectory]] = []
            reward_groups: list[list[float]] = []
            baseline_ppgs: list[float] = []
            rollout_store = GrpoRolloutStore(rollout_root, iteration=iteration)
            # 阶段一：完整自回归采样并用 C# 状态机完成 PPG/奖励标注。
            # 这一步完全 inference-only；所有 decisions 在 _score_row 中已脱离到 CPU。
            with torch.inference_mode():
                for scene_path in selected_scenes:
                    greedy_result, _ = _run_scene_rollout(
                        replay_config,
                        scene_json_path=scene_path,
                        backend=backend,
                        temperature=0.0,
                        record_decisions=False,
                        session=replay_session,
                    )
                    greedy_ppg = _ppg_from_result(greedy_result)
                    del greedy_result
                    baseline_ppgs.append(greedy_ppg)

                    scene_trajectories: list[GrpoTrajectory] = []
                    scene_deltas: list[float] = []
                    for _ in range(grpo.group_size):
                        sampled_result, decisions = _run_scene_rollout(
                            replay_config,
                            scene_json_path=scene_path,
                            backend=backend,
                            temperature=grpo.temperature,
                            record_decisions=True,
                            session=replay_session,
                        )
                        sampled_ppg = _ppg_from_result(sampled_result)
                        delta = sampled_ppg - greedy_ppg
                        scene_deltas.append(delta)
                        stored_trajectory = rollout_store.write_trajectory(
                            scene_json_path=scene_path,
                            decisions=decisions,
                            ppg=sampled_ppg,
                            greedy_ppg=greedy_ppg,
                            reward=delta,
                        )
                        del decisions, sampled_result
                        scene_trajectories.append(
                            GrpoTrajectory(
                                scene_json_path=scene_path,
                                decision_path=stored_trajectory.path,
                                decision_count=stored_trajectory.decision_count,
                                ppg=sampled_ppg,
                                greedy_ppg=greedy_ppg,
                                reward=delta,
                            )
                        )
                    trajectories.extend(scene_trajectories)
                    group_trajectories.append(scene_trajectories)
                    reward_groups.append(scene_deltas)

            # 阶段二：所有场景/样本均已标注后，才组装优势并执行真实 GRPO 反向更新。
            group_advantages = compute_baseline_relative_advantages(
                reward_groups,
                scale_floor_ratio=grpo.advantage_scale_floor_ratio,
                advantage_clip=grpo.advantage_clip,
            )
            trajectory_advantages: list[float] = []
            for group_index, scene_trajectories in enumerate(group_trajectories):
                for sample_index, _trajectory in enumerate(scene_trajectories):
                    trajectory_advantages.append(
                        float(group_advantages[group_index][sample_index].item())
                    )
            rollout_store.set_advantages(trajectory_advantages)
            if rollout_store.total_decisions < 1:
                raise RuntimeError("GRPO produced no decisions; inspect the scene time window")

            # 第 1 轮尚未有上一轮 GRPO checkpoint，只需保留小型优化器状态；模型权重
            # 直接复用 backend 已加载的 CPU checkpoint。后续轮次直接读取上一轮 latest.pt，
            # 不再 deepcopy 一份完整模型到 GPU。
            rollback_checkpoint = (
                output_path / "latest.pt" if iteration > 1 else None
            )
            initial_optimizer_state = (
                deepcopy(optimizer.state_dict()) if rollback_checkpoint is None else None
            )
            initial_scheduler_state = (
                deepcopy(scheduler.state_dict()) if rollback_checkpoint is None else None
            )
            update_metrics = _update_policy(
                model,
                optimizer,
                scheduler,
                rollout_store,
                None,
                grpo=grpo,
                repetition=backend.repetition,
                device=device,
                precision=resolved_precision,
            )

            model.eval()
            post_update_greedy = []
            for scene_path in selected_scenes:
                result, _ = _run_scene_rollout(
                    replay_config,
                    scene_json_path=scene_path,
                    backend=backend,
                    temperature=0.0,
                    record_decisions=False,
                    session=replay_session,
                )
                post_update_greedy.append(_ppg_from_result(result))
            pre_mean = sum(baseline_ppgs) / len(baseline_ppgs)
            post_mean = sum(post_update_greedy) / len(post_update_greedy)
            rolled_back = post_mean < pre_mean - grpo.greedy_guard_tolerance
            if rolled_back:
                _restore_grpo_rollback_state(
                    checkpoint_path=rollback_checkpoint,
                    initial_model_state=backend.checkpoint["model_state_dict"],
                    initial_optimizer_state=initial_optimizer_state,
                    initial_scheduler_state=initial_scheduler_state,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                )
                logger.warning(
                    "GRPO iteration %d 因贪心基准下降回滚: before=%.4f after=%.4f",
                    iteration,
                    pre_mean,
                    post_mean,
                )
                post_mean = pre_mean

            mean_reward = sum(trajectory.reward for trajectory in trajectories) / len(trajectories)
            metrics = {
                "iteration": float(iteration),
                "greedy_ppg_before": float(pre_mean),
                "greedy_ppg_after": float(post_mean),
                "sample_ppg_mean": float(
                    sum(trajectory.ppg for trajectory in trajectories) / len(trajectories)
                ),
                "reward_delta_mean": float(mean_reward),
                "reward_delta_std": float(
                    torch.tensor(
                        [trajectory.reward for trajectory in trajectories],
                        dtype=torch.float32,
                    ).std(unbiased=False).item()
                ),
                "advantage_mean": _weighted_advantage_mean(
                    rollout_store,
                    trajectory_advantages,
                ),
                "advantage_std": _weighted_advantage_std(
                    rollout_store,
                    trajectory_advantages,
                ),
                "advantage_abs_max": float(max(abs(value) for value in trajectory_advantages)),
                "actions": float(rollout_store.total_decisions),
                "rollout_max_duration_seconds": (
                    float(grpo.max_duration_seconds)
                    if grpo.max_duration_seconds is not None
                    else -1.0
                ),
                "greedy_guard_rollback": float(rolled_back),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **update_metrics,
            }
            iteration_metrics.append(metrics)
            logger.info(
                "GRPO iteration %d | greedy %.2f -> %.2f | sample delta %.4f | "
                "loss %.5f kl %.5f rollback=%s",
                iteration,
                metrics["greedy_ppg_before"],
                metrics["greedy_ppg_after"],
                metrics["reward_delta_mean"],
                metrics["loss"],
                metrics["kl"],
                rolled_back,
            )

            checkpoint_metrics = dict(metrics)
            iteration_checkpoint = output_path / f"iteration_{iteration:03d}.pt"
            _save_grpo_checkpoint(
                iteration_checkpoint,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                iteration=iteration,
                config=config,
                grpo=grpo,
                data_spec=data_spec,
                input_contract=input_contract,
                precision=resolved_precision,
                metrics=checkpoint_metrics,
            )
            _save_grpo_checkpoint(
                output_path / "latest.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                iteration=iteration,
                config=config,
                grpo=grpo,
                data_spec=data_spec,
                input_contract=input_contract,
                precision=resolved_precision,
                metrics=checkpoint_metrics,
            )
            if metrics["greedy_ppg_after"] >= best_greedy_ppg:
                best_greedy_ppg = metrics["greedy_ppg_after"]
                best_metrics = checkpoint_metrics
                _save_grpo_checkpoint(
                    output_path / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    iteration=iteration,
                    config=config,
                    grpo=grpo,
                    data_spec=data_spec,
                    input_contract=input_contract,
                    precision=resolved_precision,
                    metrics=checkpoint_metrics,
                )

        final_metrics = iteration_metrics[-1]
        _save_grpo_checkpoint(
            output_path / "final.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            iteration=grpo.max_iterations,
            config=config,
            grpo=grpo,
            data_spec=data_spec,
            input_contract=input_contract,
            precision=resolved_precision,
            metrics=final_metrics,
        )
        return {
            "output_dir": output_path,
            "final_checkpoint": output_path / "final.pt",
            "best_checkpoint": output_path / "best.pt",
            "data_spec": data_spec,
            "best_greedy_ppg": best_greedy_ppg,
            "best_metrics": best_metrics,
            "last_metrics": final_metrics,
        }

    finally:
        replay_session.close()
