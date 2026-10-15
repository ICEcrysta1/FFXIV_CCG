"""GRPO 后训练编排：采样、策略更新、贪心保护与 checkpoint 生命周期。"""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import torch

from common.policy.config import (
    resolve_policy_cache_dir,
    resolve_policy_grpo_dir,
    validate_policy_model_variant,
)
from common.policy.data import ActionSpace
from common.policy.replay import AutoregressiveReplayConfig
from common.training.optimizer import build_optimizer
from common.training.tensorboard import (
    TensorBoardMonitor,
    close_tensorboard_writer,
    create_tensorboard_writer,
    write_scalar_metrics,
)
from scripts.autoregressive_replay.backends import PyTorchPolicyBackend
from scripts.autoregressive_replay.parallel import ParallelRollouts
from scripts.autoregressive_replay.replay import AutoregressiveReplaySession, ReplayCacheStore

from .checkpoint import (
    _restore_grpo_rollback_state,
    _save_grpo_checkpoint,
    _snapshot_optimizer_state,
)
from .config import GrpoConfig, GrpoRunConfig
from .rollout import _ppg_from_result, _run_scene_rollout
from .storage import GrpoRolloutStore, StoredGrpoTrajectory
from .update import (
    _update_policy,
    _weighted_advantage_mean,
    _weighted_advantage_std,
    compute_baseline_relative_advantages,
)

logger = logging.getLogger(__name__)


def _unique_grpo_output_path(base_path: Path, suffix: str) -> Path:
    """为从已有 GRPO checkpoint 热启动选择不覆盖历史的输出目录。"""
    base_path = Path(base_path).resolve()
    candidate = base_path.parent / f"{base_path.name}{suffix}"
    if not candidate.exists():
        return candidate
    index = 1
    while True:
        candidate = base_path.parent / f"{base_path.name}{suffix}_{index:03d}"
        if not candidate.exists():
            return candidate
        index += 1


def run_grpo_training(
    config: GrpoRunConfig,
    *,
    grpo: GrpoConfig,
    checkpoint_path: Path,
    raw_paths: Sequence[Path],
    output_dir: Path | None = None,
    tensorboard_output_dir: Path | None = None,
    device_name: str = "cuda",
    precision: str | None = None,
) -> dict[str, object]:
    """从 checkpoint 进入独立 GRPO 后训练。"""
    if not raw_paths:
        raise FileNotFoundError("GRPO requires at least one real training scene")
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"GRPO checkpoint not found: {checkpoint_path}")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by GRPO, but torch.cuda.is_available() is false")
    device = torch.device(device_name)
    resolved_precision = str(precision or config.precision).strip().lower()
    # 只加载一份模型；各轨迹持有独立 session，共用有界引擎和 compiled cache。
    # 所有采样完成并释放队列后才更新参数，避免推理与反向传播交错。
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

    if output_dir is not None:
        output_path = Path(output_dir).resolve()
    elif isinstance(backend.checkpoint, Mapping) and backend.checkpoint.get(
        "grpo_checkpoint"
    ):
        # --checkpoint 对 GRPO 只做权重热启动，且不能默认覆盖来源实验；输出目录
        # 始终从模型 YAML 的 output_dir 派生，避免 checkpoint 在项目外时把新 run
        # 写到项目外，也避免重复热启动时叠加多段 `_hotstart` 后缀。
        base_output = config.output_dir.parent / f"{config.output_dir.name}_grpo"
        output_path = _unique_grpo_output_path(base_output, "_hotstart")
    else:
        output_path = config.output_dir.parent / f"{config.output_dir.name}_grpo"
    output_path = output_path.resolve()

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
        top_k=data_spec.num_actions,
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
    rollouts = ParallelRollouts(backend, job_tag=data_spec.job_tag)
    replay_cache_store = ReplayCacheStore(
        max_shards=config.compiled_cache_max_shards, max_readers=rollouts.workers,
    )

    def run_rollout_task(task, policy, engine):
        scene_path, temperature, record_decisions, sampling_seed = task
        session = AutoregressiveReplaySession(
            replay_config, backend=policy, cache_store=replay_cache_store, engine=engine,
        )
        try:
            return _run_scene_rollout(
                replay_config, scene_json_path=scene_path,
                temperature=temperature, record_decisions=record_decisions,
                session=session, sampling_seed=sampling_seed,
            )
        finally:
            session.close()

    def prepare_rollouts(tasks, engine):
        replay_cache_store.prepare(
            [replace(replay_config, scene_json_path=task[0]) for task in tasks],
            job_tag=data_spec.job_tag, normalizer=input_contract.create_normalizer(),
            expected_action_space=ActionSpace.from_data_spec(data_spec),
            expected_skill_vocab=input_contract.create_skill_vocab(),
            engine=engine, workers=rollouts.workers,
        )

    tensorboard_writer = None
    tensorboard_monitor = None
    best_greedy_ppg = float("-inf")
    iteration_metrics: list[dict[str, float]] = []
    total_optimizer_updates = 0
    try:
        tensorboard_writer = create_tensorboard_writer(
            grpo.tensorboard,
            (
                tensorboard_output_dir
                if tensorboard_output_dir is not None
                else config.output_dir
            ),
            run_name="grpo",
            model_variant=config.model_variant,
        )
        if tensorboard_writer is not None:
            logger.info("TensorBoard events: %s", tensorboard_writer.log_dir)
            tensorboard_monitor = TensorBoardMonitor(
                tensorboard_writer,
                grpo.tensorboard,
                model=model,
                vocab=input_contract.create_skill_vocab(),
                metadata={
                    "stage": "grpo",
                    "precision": resolved_precision,
                    "checkpoint": str(checkpoint_path),
                    "scene_count": len(scenes),
                    "run_config": asdict(config),
                    "grpo_config": asdict(grpo),
                    "iteration_step": "累计优化器更新次数",
                    "greedy_ppg_after_update": "贪心保护回滚前的候选模型 PPG",
                },
            )
        optimizer = build_optimizer(
            model,
            grpo.optimizer,
            learning_rate=grpo.learning_rate,
            weight_decay=grpo.weight_decay,
        )
        logger.info(
            "GRPO 优化器: %s learning_rate=%g weight_decay=%g",
            grpo.optimizer.name,
            grpo.learning_rate,
            grpo.weight_decay,
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
        best_metrics: dict[str, float] = {}
        rollout_root = resolve_policy_grpo_dir(data_spec.job_tag)

        for iteration in range(1, grpo.max_iterations + 1):
            selected_scenes = [
                scenes[
                    ((iteration - 1) * grpo.prompt_batch_size + offset) % len(scenes)
                ]
                for offset in range(grpo.prompt_batch_size)
            ]
            model.eval()
            trajectories: list[StoredGrpoTrajectory] = []
            reward_groups: list[list[float]] = []
            baseline_ppgs: list[float] = []
            rollout_store = GrpoRolloutStore(rollout_root, iteration=iteration, input_contract=input_contract)
            # 阶段一：完整自回归采样并用 C# 状态机完成 PPG/奖励标注。
            # 这一步完全 inference-only；所有 decisions 在 _score_row 中已脱离到 CPU。
            baseline_tasks = ((path, 0.0, False, None) for path in selected_scenes)
            for result, _ in rollouts.map(run_rollout_task, baseline_tasks, prepare=prepare_rollouts):
                baseline_ppgs.append(_ppg_from_result(result))
            reward_groups = [[] for _ in selected_scenes]
            sampling_tasks = (
                (path, grpo.temperature, True,
                 config.seed + (((iteration - 1) * grpo.prompt_batch_size + scene_index)
                                * grpo.group_size + sample_index))
                for scene_index, path in enumerate(selected_scenes)
                for sample_index in range(grpo.group_size)
            )
            for task_index, (sampled_result, decisions) in enumerate(
                rollouts.map(run_rollout_task, sampling_tasks, prepare=prepare_rollouts)
            ):
                scene_index = task_index // grpo.group_size
                scene_path = selected_scenes[scene_index]
                greedy_ppg = baseline_ppgs[scene_index]
                sampled_ppg = _ppg_from_result(sampled_result)
                delta = sampled_ppg - greedy_ppg
                reward_groups[scene_index].append(delta)
                stored_trajectory = rollout_store.write_trajectory(
                    scene_json_path=scene_path, decisions=decisions,
                    ppg=sampled_ppg, greedy_ppg=greedy_ppg, reward=delta,
                )
                del decisions, sampled_result
                trajectories.append(stored_trajectory)

            # 阶段二：所有场景/样本均已标注后，才组装优势并执行真实 GRPO 反向更新。
            group_advantages = compute_baseline_relative_advantages(
                reward_groups,
                scale_floor_ratio=grpo.advantage_scale_floor_ratio,
                advantage_clip=grpo.advantage_clip,
            )
            trajectory_advantages = [
                float(advantage.item())
                for group in group_advantages
                for advantage in group
            ]
            rollout_store.set_advantages(trajectory_advantages)
            if rollout_store.total_decisions < 1:
                raise RuntimeError("GRPO produced no decisions; inspect the scene time window")

            # 第 1 轮的优化器状态（含 FP32 主权重）只在 CPU 留回滚快照；模型权重
            # 复用 backend 已加载的 CPU checkpoint。后续轮次读取上一轮 latest.pt。
            rollback_checkpoint = (
                output_path / "latest.pt" if iteration > 1 else None
            )
            initial_optimizer_state = (
                _snapshot_optimizer_state(optimizer) if rollback_checkpoint is None else None
            )
            initial_scheduler_state = (
                deepcopy(scheduler.state_dict()) if rollback_checkpoint is None else None
            )
            update_metrics = _update_policy(
                model,
                optimizer,
                scheduler,
                rollout_store,
                grpo=grpo,
                repetition=backend.repetition,
                device=device,
                precision=resolved_precision,
                **(
                    {
                        "tensorboard_writer": tensorboard_writer,
                        "tensorboard_monitor": tensorboard_monitor,
                        "tensorboard_log_every_steps": grpo.tensorboard.log_every_steps,
                        "global_step_offset": total_optimizer_updates,
                    }
                    if tensorboard_writer is not None
                    else {}
                ),
            )
            total_optimizer_updates += int(update_metrics["optimizer_updates"])

            model.eval()
            post_update_greedy = []
            for result, _ in rollouts.map(
                run_rollout_task, ((path, 0.0, False, None) for path in selected_scenes),
                prepare=prepare_rollouts,
            ):
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
            if tensorboard_writer is not None:
                write_scalar_metrics(
                    tensorboard_writer,
                    metrics,
                    prefix="grpo/iteration",
                    global_step=total_optimizer_updates,
                )
            if tensorboard_monitor is not None:
                tensorboard_monitor.log_distributions(
                    {
                        "greedy_ppg_before": baseline_ppgs,
                        "greedy_ppg_after_update": post_update_greedy,
                        "sample_ppg": [trajectory.ppg for trajectory in trajectories],
                        "reward_delta": [trajectory.reward for trajectory in trajectories],
                        "trajectory_advantage": trajectory_advantages,
                        "decision_count": [trajectory.decision_count for trajectory in trajectories],
                    },
                    prefix="grpo/rollout",
                    global_step=total_optimizer_updates,
                )
                # 回滚完成后再记录权重与 embedding，展示实际保留的策略。
                tensorboard_monitor.log_epoch(global_step=total_optimizer_updates)
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
        try:
            if tensorboard_monitor is not None and iteration_metrics:
                tensorboard_monitor.finish(
                    {
                        **{
                            f"grpo/iteration/{name}": value
                            for name, value in iteration_metrics[-1].items()
                        },
                        "grpo/best_greedy_ppg": best_greedy_ppg,
                    },
                    global_step=total_optimizer_updates,
                )
        finally:
            try:
                close_tensorboard_writer(tensorboard_writer)
            finally:
                rollouts.close()
