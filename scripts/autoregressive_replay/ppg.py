"""训练期间的模型自回归 PPG 评估（C# 状态机后端版）。"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math

import torch

from common.torch_runtime import autocast_context
from scripts.common.inprocess_backend import InProcessBackend
from common.policy.data import Normalizer

from .context import LiveBatchBuilder, SceneTemplateProvider
from .replay import observe_replay_state, read_replay_cumulative_potency
from .scheduler import DecisionScheduler, gcd_request_delay, is_gcd_decision

PPG_TIME_EPSILON = 1e-6
OGCD_WAIT_ACTION_KEY = "ogcd_wait"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PpgResult:
    """一次固定 GCD 长度木桩回放的威力指标。"""

    output_gcds: int
    cumulative_potency: float
    cumulative_dot_potency: float
    ppg: float
    normalized_ppg: float


class EmptySceneProvider:
    """提供没有副本机制 token、始终单目标的木桩场景。"""

    def __init__(self, scene_dim: int):
        if scene_dim <= 0:
            raise ValueError("PPG requires a positive scene dimension")
        self._scene_vectors = torch.zeros((0, scene_dim), dtype=torch.float32)
        self._scene_types = torch.zeros((0,), dtype=torch.int32)

    def at_time(self, time_seconds: float):
        del time_seconds
        return self._scene_vectors, self._scene_types

    def target_count_at(self, time_seconds: float) -> int:
        del time_seconds
        return 1


def evaluate_none_ppg(
    model,
    *,
    dataset,
    data_spec,
    vocab,
    config,
    device: torch.device,
) -> dict[str, float]:
    """从空历史初始状态开始，执行模型 Top-1 合法动作并计算 none_ppg。"""
    ppg_config = config.ppg
    if not ppg_config.enabled:
        return {}

    normalizer = Normalizer()
    normalizer.configure_job_resources(data_spec.job_tag)
    normalizer.register_schema(dataset.schema)
    with InProcessBackend(
        job_tag=data_spec.job_tag,
        max_history=config.model.history_capacity,
    ) as backend:
        batcher = LiveBatchBuilder(
            backend=backend,
            vocab=vocab,
            normalizer=normalizer,
            schema=dataset.schema,
            skill_feature_names=dataset.skill_feature_names,
            scene_provider=EmptySceneProvider(data_spec.scene_dim),
            device=device,
            max_history=config.model.history_capacity,
            candidate_action_keys=data_spec.candidate_action_keys,
        )
        result = _run_rollout(
            model,
            backend,
            batcher,
            gcd_count=ppg_config.gcd_count,
            normalization=ppg_config.normalization,
            precision=config.precision,
            device=device,
            expected_candidate_keys=data_spec.candidate_action_keys,
        )
    return {
        "none_ppg": result.ppg,
        "none_ppg_normalized": result.normalized_ppg,
        "none_ppg_output_gcds": float(result.output_gcds),
        "none_ppg_cumulative_potency": result.cumulative_potency,
        "none_ppg_cumulative_dot_potency": result.cumulative_dot_potency,
    }


@torch.no_grad()
def evaluate_validation_ppg(
    model,
    *,
    dataset,
    data_spec,
    vocab,
    config,
    device: torch.device,
) -> dict[str, float]:
    """按验证副本执行模型自回归回放，并平均每个副本的最终 PPG。"""
    ppg_config = config.ppg
    if not ppg_config.enabled:
        return {}

    iter_source_readers = getattr(dataset, "iter_source_readers", None)
    if not callable(iter_source_readers):
        raise TypeError("validation PPG requires a TrainingDataset source reader iterator")

    normalizer = Normalizer()
    normalizer.configure_job_resources(data_spec.job_tag)
    normalizer.register_schema(dataset.schema)
    source_results: list[PpgResult] = []
    model.eval()
    enable_kv_cache = getattr(model, "enable_kv_cache", None)
    reset_kv_cache = getattr(model, "reset_kv_cache", None)
    cache_was_enabled = bool(getattr(model, "_kv_cache_enabled", False))
    if callable(enable_kv_cache):
        # 当前基准中 KV 路径更慢，只有显式配置才启用。
        enable_kv_cache(ppg_config.use_kv_cache)
    try:
        with InProcessBackend(
            job_tag=data_spec.job_tag,
            max_history=config.model.history_capacity,
        ) as backend:
            for source_index, reader in enumerate(iter_source_readers()):
                if callable(reset_kv_cache):
                    reset_kv_cache()
                initial_metadata = reader.step_metadata(0)
                initial_time = float(initial_metadata["time_offset"])
                scene_provider = SceneTemplateProvider(
                    reader,
                    normalizer=normalizer,
                    initial_sample_index=0,
                    backend=backend,
                )
                fight_end_time = scene_provider.last_targetable_end()
                if fight_end_time <= initial_time + PPG_TIME_EPSILON:
                    raise ValueError(
                        "validation PPG source must have a positive replay window: "
                        f"fight={reader.fight_id!r} initial={initial_time} end={fight_end_time}"
                    )
                actual_base_gcd = _infer_initial_base_gcd(
                    reader,
                    normalizer=normalizer,
                    skill_feature_names=dataset.skill_feature_names,
                )

                backend.init(
                    actual_base_gcd=float(actual_base_gcd),
                    fight_remaining=fight_end_time - initial_time,
                    initial_timestamp=initial_time,
                )
                state = observe_replay_state(backend, initial_time)
                scene_provider.sync_state(state)
                batcher = LiveBatchBuilder(
                    backend=backend,
                    vocab=vocab,
                    normalizer=normalizer,
                    schema=dataset.schema,
                    skill_feature_names=dataset.skill_feature_names,
                    scene_provider=scene_provider,
                    device=device,
                    max_history=config.model.history_capacity,
                    candidate_action_keys=data_spec.candidate_action_keys,
                )
                source_results.append(
                    _run_rollout_until_time(
                        model,
                        backend,
                        batcher,
                        scene_provider=scene_provider,
                        end_time=fight_end_time,
                        normalization=ppg_config.normalization,
                        precision=config.precision,
                        device=device,
                        expected_candidate_keys=data_spec.candidate_action_keys,
                        source_label=f"{source_index}:{reader.fight_id}",
                    )
                )
    finally:
        if callable(reset_kv_cache):
            reset_kv_cache()
        if callable(enable_kv_cache):
            enable_kv_cache(cache_was_enabled)

    if not source_results:
        raise ValueError("validation PPG found no source to replay")

    average_ppg = sum(result.ppg for result in source_results) / len(source_results)
    return {
        "val_ppg": average_ppg,
        "val_ppg_normalized": average_ppg / ppg_config.normalization,
        "val_ppg_fights": float(len(source_results)),
        "val_ppg_output_gcds": float(sum(result.output_gcds for result in source_results)),
        "val_ppg_cumulative_potency": sum(
            result.cumulative_potency for result in source_results
        ),
        "val_ppg_cumulative_dot_potency": sum(
            result.cumulative_dot_potency for result in source_results
        ),
    }


def _infer_initial_base_gcd(
    reader,
    *,
    normalizer: Normalizer,
    skill_feature_names,
) -> float:
    """从首个缓存样本的合法 GCD 技能 token 恢复状态机基础 GCD。

    首个样本没有历史，正是转换器开始回放时的初始状态。`gcd_window.seconds`
    已按 remaining_seconds_max 归一化，缓存本身不需要额外保存一份重复的 GCD 元数据。
    """
    feature_names = tuple(str(name) for name in skill_feature_names)
    try:
        kind_index = feature_names.index("kind")
        gcd_window_index = feature_names.index("gcd_window.seconds")
        legal_index = feature_names.index("is_legal")
    except ValueError as exc:
        raise ValueError(
            "validation PPG cache is missing numeric skill kind/GCD legality features"
        ) from exc

    sample = reader.sample(0)
    candidate_features = sample.get("candidate_skill_features")
    candidate_legal_mask = sample.get("candidate_legal_mask")
    if candidate_features is None or candidate_legal_mask is None:
        raise ValueError(
            "validation PPG cache first sample is missing candidate skill features"
        )

    gcd_values: list[float] = []
    for row, legal in zip(candidate_features, candidate_legal_mask):
        if float(legal) < 0.5 or float(row[legal_index]) < 0.5:
            continue
        if float(row[kind_index]) < 0.5:
            continue
        normalized_gcd = float(row[gcd_window_index])
        if normalized_gcd <= PPG_TIME_EPSILON:
            continue
        gcd_values.append(normalized_gcd * normalizer.remaining_seconds_max)

    if not gcd_values:
        raise ValueError(
            "validation PPG cache first sample has no legal GCD candidate to recover base GCD"
        )
    base_gcd = min(gcd_values)
    if not math.isfinite(base_gcd) or base_gcd <= PPG_TIME_EPSILON:
        raise ValueError(f"invalid initial base GCD recovered from cache: {base_gcd!r}")
    return base_gcd


@torch.no_grad()
def _run_rollout(
    model,
    backend: InProcessBackend,
    batcher,
    *,
    gcd_count: int,
    normalization: float,
    precision: str,
    device: torch.device,
    expected_candidate_keys,
) -> PpgResult:
    """执行固定输出 GCD 数量的贪心自回归木桩回放。"""
    if gcd_count < 1:
        raise ValueError("PPG gcd_count must be >= 1")
    if normalization <= 0.0:
        raise ValueError("PPG normalization must be > 0")

    model.eval()
    state = observe_replay_state(backend, 0.0)
    output_gcds = 0
    action_count = 0
    max_actions = max(gcd_count * 16, gcd_count + 32)
    expected_candidate_keys = tuple(str(key) for key in expected_candidate_keys)

    while output_gcds < gcd_count:
        action_count += 1
        if action_count > max_actions:
            raise RuntimeError(
                "PPG autoregressive rollout exceeded its action safety limit "
                f"before reaching {gcd_count} GCDs"
            )

        batch, candidate_keys = batcher.build(state)
        candidate_keys = tuple(str(key) for key in candidate_keys)
        if candidate_keys != expected_candidate_keys:
            raise ValueError(
                "PPG candidate action order mismatch: "
                f"live={candidate_keys!r} cache={expected_candidate_keys!r}"
            )
        with autocast_context(device, precision):
            logits = model(batch)["logits"][0].float()
        legal_mask = batch["candidate_legal_mask"][0].bool()
        selected_index, has_legal_candidate = _select_legal_candidate(logits, legal_mask)
        if not has_legal_candidate:
            advanced = DecisionScheduler(
                backend, lambda timestamp: observe_replay_state(backend, timestamp),
            ).advance_to_next_decision(state)
            if advanced is not None:
                state = advanced
                continue
            raise RuntimeError("PPG live state has no legal candidate")
        action_key = candidate_keys[selected_index]
        if action_key == OGCD_WAIT_ACTION_KEY:
            wait_seconds = gcd_request_delay(state)
            if wait_seconds <= PPG_TIME_EPSILON:
                raise RuntimeError("PPG selected ogcd_wait while GCD is already ready")
            backend.record_policy_action(
                float(state.time),
                OGCD_WAIT_ACTION_KEY,
                float(state.time) + wait_seconds,
            )
            state = _advance_event_time(backend, state, wait_seconds)
            continue
        action_kind = "gcd" if is_gcd_decision(state.gcd_remaining) else "ogcd"
        previous_gcd_index = int(getattr(state, "gcd_index", 0))
        submission = backend.submit_action(float(state.time), action_key)
        if not submission.accepted:
            raise RuntimeError(f"PPG action rejected: {action_key} ({submission.reason})")
        state = observe_replay_state(backend, float(state.time))
        state = _advance_after_action(
            backend,
            state,
            submission,
            action_kind=action_kind,
        )
        if int(getattr(state, "gcd_index", 0)) > previous_gcd_index:
            output_gcds += 1

    cumulative_potency, cumulative_dot_potency = read_replay_cumulative_potency(
        backend, float(state.time)
    )
    total_potency = cumulative_potency + cumulative_dot_potency
    ppg = total_potency / float(output_gcds)
    return PpgResult(
        output_gcds=output_gcds,
        cumulative_potency=cumulative_potency,
        cumulative_dot_potency=cumulative_dot_potency,
        ppg=ppg,
        normalized_ppg=ppg / normalization,
    )


@torch.no_grad()
def _run_rollout_until_time(
    model,
    backend: InProcessBackend,
    batcher,
    *,
    scene_provider,
    end_time: float,
    normalization: float,
    precision: str,
    device: torch.device,
    expected_candidate_keys,
    source_label: str,
) -> PpgResult:
    """从验证副本首个决策点自回归执行到最后一个人类动作时刻。"""
    if normalization <= 0.0:
        raise ValueError("PPG normalization must be > 0")

    model.eval()
    state = observe_replay_state(backend, 0.0)
    initial_time = float(state.time)
    output_gcds = 0
    action_count = 0
    max_actions = max(1024, int(max(1.0, end_time - initial_time) * 32.0))
    expected_candidate_keys = tuple(str(key) for key in expected_candidate_keys)

    while float(state.time) < end_time - PPG_TIME_EPSILON:
        action_count += 1
        if action_count > max_actions:
            raise RuntimeError(
                "validation PPG autoregressive rollout exceeded its action safety limit "
                f"for {source_label}: {max_actions}"
            )

        batch, candidate_keys = batcher.build(state)
        candidate_keys = tuple(str(key) for key in candidate_keys)
        if candidate_keys != expected_candidate_keys:
            raise ValueError(
                "PPG candidate action order mismatch: "
                f"live={candidate_keys!r} cache={expected_candidate_keys!r}"
            )
        with autocast_context(device, precision):
            logits = model(batch)["logits"][0].float()
        legal_mask = batch["candidate_legal_mask"][0].bool()
        selected_index, has_legal_candidate = _select_legal_candidate(logits, legal_mask)
        if not has_legal_candidate:
            advanced = DecisionScheduler(
                backend, lambda timestamp: observe_replay_state(backend, timestamp), scene_provider,
            ).advance_to_next_decision(state, end_time=end_time)
            if advanced is not None:
                state = advanced
                continue
            logger.warning(
                "验证 PPG 副本候选全非法，跳过本副本并记 PPG=0: "
                "source=%s time=%.4f executed_gcds=%d",
                source_label,
                float(state.time),
                output_gcds,
            )
            return PpgResult(
                output_gcds=0,
                cumulative_potency=0.0,
                cumulative_dot_potency=0.0,
                ppg=0.0,
                normalized_ppg=0.0,
            )
        action_key = candidate_keys[selected_index]
        if action_key == OGCD_WAIT_ACTION_KEY:
            wait_seconds = gcd_request_delay(state)
            if wait_seconds <= PPG_TIME_EPSILON:
                raise RuntimeError("validation PPG selected ogcd_wait while GCD is already ready")
            backend.record_policy_action(
                float(state.time),
                OGCD_WAIT_ACTION_KEY,
                float(state.time) + wait_seconds,
            )
            state = _advance_event_time(
                backend,
                state,
                wait_seconds,
                scene_provider=scene_provider,
                end_time=end_time,
            )
            continue
        action_kind = "gcd" if is_gcd_decision(state.gcd_remaining) else "ogcd"
        previous_gcd_index = int(getattr(state, "gcd_index", 0))
        submission = backend.submit_action(float(state.time), action_key)
        if not submission.accepted:
            raise RuntimeError(f"validation PPG action rejected: {action_key} ({submission.reason})")
        state = observe_replay_state(backend, float(state.time))
        state = _advance_after_action(
            backend,
            state,
            submission,
            action_kind=action_kind,
            scene_provider=scene_provider,
            end_time=end_time,
        )
        if int(getattr(state, "gcd_index", 0)) > previous_gcd_index:
            output_gcds += 1

    cumulative_potency, cumulative_dot_potency = read_replay_cumulative_potency(
        backend, float(state.time)
    )
    total_potency = cumulative_potency + cumulative_dot_potency
    if output_gcds < 1:
        raise RuntimeError(
            "validation PPG rollout reached its end without executing a GCD "
            f"for {source_label}"
        )
    ppg = total_potency / float(output_gcds)
    return PpgResult(
        output_gcds=output_gcds,
        cumulative_potency=cumulative_potency,
        cumulative_dot_potency=cumulative_dot_potency,
        ppg=ppg,
        normalized_ppg=ppg / normalization,
    )


def _advance_after_action(
    backend: InProcessBackend,
    state,
    submission,
    *,
    action_kind: str,
    scene_provider=None,
    end_time: float | None = None,
):
    """推进真实动作占用时间，并在 scene 边界或验证结束时停下。"""
    return DecisionScheduler(
        backend,
        lambda timestamp: observe_replay_state(backend, timestamp),
        scene_provider,
    ).advance_submitted_action(
        state,
        submission,
        action_kind=action_kind,
        end_time=end_time,
    )


def _select_legal_candidate(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
) -> tuple[int, bool]:
    """合并动作索引与合法性读取，避免两次 CUDA 到 CPU 同步。"""
    selected_index = logits.masked_fill(~legal_mask, float("-inf")).argmax()
    has_legal_candidate = legal_mask.any()
    decision = torch.stack(
        (
            selected_index.to(dtype=torch.int64),
            has_legal_candidate.to(dtype=torch.int64),
        )
    )
    selected_value, has_legal_value = decision.cpu().tolist()
    return int(selected_value), bool(has_legal_value)


def _advance_event_time(
    backend: InProcessBackend,
    state,
    seconds: float,
    *,
    scene_provider=None,
    interrupt_on_scene_event: bool = False,
    end_time: float | None = None,
):
    return DecisionScheduler(
        backend,
        lambda timestamp: observe_replay_state(backend, timestamp),
        scene_provider,
    ).advance_by(
        state,
        seconds,
        interrupt_on_scene_event=interrupt_on_scene_event,
        end_time=end_time,
    )
