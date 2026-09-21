"""基于 C# 状态机后端（SidecarHost）的模型自回归回放。"""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import torch

from common.config import load_project_config
from common.skills import SkillBook
from scripts.common.cs_backend import SidecarBackend
from scripts.convert_fflogs.cache import (
    load_raw_compiled_cache,
    precompile_raw_training_caches,
)
from common.policy.data import Normalizer, SkillVocab
from common.policy.data.compiled_cache import (
    CompiledCacheReader,
    CompiledShardCache,
    cache_path_for_source,
)
from common.policy.model.repetition import apply_repetition_penalty

from .backends import (
    OrtPolicyBackend,
    PolicyBackend,
    PyTorchPolicyBackend,
    validate_backend_vocab,
)
from .config import AutoregressiveReplayConfig
from .context import LiveBatchBuilder, SceneTemplateProvider
from .scheduler import DecisionScheduler, decision_timing, gcd_request_delay


GCD_ACTION_KIND = "gcd"
OGCD_WAIT_ACTION_KEY = "ogcd_wait"
EVENT_TIME_EPSILON = 1e-6
EVENT_SAFETY_MULTIPLIER = 32
EVENT_SAFETY_LIMIT_WITHOUT_ACTION_CAP = 1_000_000
logger = logging.getLogger(__name__)


def _load_scene_duration_seconds(scene_json_path: Path) -> float | None:
    """从 raw scene 的 fights 元数据读取以 0 为起点的战斗时长。"""
    try:
        with Path(scene_json_path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    fights = payload.get("fights")
    if not isinstance(fights, list):
        return None
    requested_fight_id = payload.get("fight_id")
    candidates = [
        fight
        for fight in fights
        if isinstance(fight, dict)
        and (
            requested_fight_id is None
            or fight.get("id") == requested_fight_id
        )
    ]
    if not candidates:
        candidates = [fight for fight in fights if isinstance(fight, dict)]
    durations = []
    for fight in candidates:
        try:
            duration = (float(fight["end_time"]) - float(fight["start_time"])) / 1000.0
        except (KeyError, TypeError, ValueError):
            continue
        if duration > 0.0:
            durations.append(duration)
    return max(durations) if durations else None


def observe_replay_state(backend: SidecarBackend, timestamp: float):
    """读取绝对时间秒制快照并提供回放所需的属性视图。"""
    observation = backend.observe_at(float(timestamp), format="seconds")
    payload = dict(observation.context)
    payload["time"] = float(observation.timestamp)
    payload["next_scheduled_event_time"] = observation.next_scheduled_event_time
    aliases = {
        "gcd_remaining_seconds": "gcd_remaining",
        "cast_remaining_seconds": "cast_remaining",
        "downtime_remaining_seconds": "downtime_remaining",
        "fight_remaining_seconds": "fight_remaining",
        "current_gcd_seconds": "current_gcd",
    }
    for source, target in aliases.items():
        payload[target] = float(payload.get(source, 0.0) or 0.0)
    return SimpleNamespace(**payload)


def read_replay_cumulative_potency(backend: SidecarBackend, timestamp: float) -> tuple[float, float]:
    """从最终 vector 观测读取状态机实际累计威力。"""
    canonical = backend.observe_at(
        float(timestamp),
        format="vector",
        next_observation_timestamp=float(timestamp),
    ).context
    history = canonical.get("state_history_context", {})
    tokens = history.get("tokens", []) if isinstance(history, dict) else []
    if not tokens:
        return 0.0, 0.0
    keys = history.get("target_buff_state_feature_keys", [])
    target = tokens[-1].get("target_buff_state", [])
    try:
        potency_index = keys.index("after.target.cumulative_potency")
        dot_index = keys.index("after.target.cumulative_dot_potency")
        return float(target[potency_index]), float(target[dot_index])
    except (AttributeError, IndexError, ValueError, TypeError):
        raise RuntimeError("C# vector observation lacks cumulative potency fields") from None


class NoLegalCandidateError(RuntimeError):
    """当前时刻没有合法动作，但时间推进后可能恢复。"""


@dataclass(frozen=True)
class ReplayRow:
    """一次 live 决策的可读记录。"""

    gcd_step: int
    action_key: str
    probability: float
    top_candidates: tuple[tuple[str, float, float, bool], ...]
    forced: bool = False
    reference_action_key: str | None = None


@dataclass(frozen=True)
class ReplaySnapshot:
    """完整自回归轨迹中的一个固定决策状态（决策时刻的上下文原料）。

    历史消融直接复用缓存的 canonical 上下文做截断重算，不需要恢复
    后端状态；scene 向量来自 scene provider（模板固定整场）。
    """

    canonical: dict[str, object]
    reference_row: ReplayRow
    # 该标记来自 C# 状态机快照，避免 replay 自己推导 wait 边界生命周期。


@dataclass(frozen=True)
class ReplayResult:
    """回放结果和运行元数据。"""

    rows: tuple[ReplayRow, ...]
    checkpoint_path: Path | None
    scene_json_path: Path
    device: str
    history_limit: int | None = None
    is_history_ablation: bool = False
    backend_name: str = "pytorch"
    execution_provider: str | None = None
    model_source_path: Path | None = None
    backend_metrics: dict[str, object] | None = None
    output_gcds: int | None = None
    cumulative_potency: float | None = None
    cumulative_dot_potency: float | None = None
    ppg: float | None = None


def _load_replay_cache(
    config: AutoregressiveReplayConfig,
    job_tag: str,
    normalizer: Normalizer,
    *,
    shard_cache: CompiledShardCache | None = None,
) -> CompiledCacheReader:
    """读取 replay cache；缺失或过期时用 checkpoint 契约重建后重试。"""
    normalizer.ensure_job_resources(job_tag)
    load_kwargs = {
        "cache_dir": config.cache_dir,
        "normalizer": normalizer,
        "int_dtype": torch.int32,
        "float_dtype": torch.float32,
        "shard_size": config.cache_shard_size,
        "max_shards": config.cache_max_shards,
    }
    if shard_cache is not None:
        load_kwargs["shard_cache"] = shard_cache
    reader = load_raw_compiled_cache(config.scene_json_path, **load_kwargs)
    if reader is not None:
        return reader

    logger.info("回放 cache 缺失或过期，调用转换脚本: %s", config.scene_json_path)
    precompile_raw_training_caches(
        [config.scene_json_path],
        job_tag=job_tag,
        normalizer=normalizer,
        int_dtype=torch.int32,
        float_dtype=torch.float32,
        cache_dir=config.cache_dir,
        shard_size=config.cache_shard_size,
        max_workers=1,
        max_shards=config.cache_max_shards,
    )
    reader = load_raw_compiled_cache(config.scene_json_path, **load_kwargs)
    if reader is None:
        raise FileNotFoundError(
            "compiled cache not found or stale for replay raw JSON after compilation: "
            f"{config.scene_json_path}"
        )
    return reader


def _cache_file_fingerprint(path: Path) -> tuple[int, int] | None:
    """返回 cache manifest 的大小和修改时间，用于避免复用过期 reader。"""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return int(stat.st_size), int(stat.st_mtime_ns)


class ReplayCacheStore:
    """跨多条 replay 轨迹复用 compiled cache reader 与 shard LRU。"""

    def __init__(self, *, max_shards: int):
        self._shard_cache = CompiledShardCache(max_shards)
        self._readers: dict[tuple[object, ...], CompiledCacheReader] = {}
        self._cache_fingerprints: dict[
            str,
            tuple[tuple[int, int], tuple[int, int] | None],
        ] = {}

    def load(
        self,
        config: AutoregressiveReplayConfig,
        *,
        job_tag: str,
        normalizer: Normalizer,
    ) -> CompiledCacheReader:
        """按 source/cache 契约读取 reader，并在 cache 更新后失效旧 shard。"""
        source_path = Path(config.scene_json_path).resolve()
        source_stat = source_path.stat()
        cache_path = cache_path_for_source(config.cache_dir, source_path).resolve()
        cache_key = str(cache_path).lower()
        cache_fingerprint = _cache_file_fingerprint(cache_path)
        source_fingerprint = (int(source_stat.st_size), int(source_stat.st_mtime_ns))
        generation = (source_fingerprint, cache_fingerprint)
        previous_fingerprint = self._cache_fingerprints.get(cache_key)
        if (
            cache_key in self._cache_fingerprints
            and previous_fingerprint != generation
        ):
            # shard key 只包含 cache path 和 shard index；manifest 被原子替换
            # 后必须清空旧 shard，避免新旧 cache 同名时读到旧 tensor。
            self._shard_cache.clear()
            self._readers.clear()
        self._cache_fingerprints[cache_key] = generation
        normalizer_signature = getattr(normalizer, "cache_signature", None)
        reader_key = (
            cache_key,
            job_tag,
            int(source_stat.st_size),
            int(source_stat.st_mtime_ns),
            config.cache_shard_size,
            config.cache_max_shards,
            repr(normalizer_signature),
            cache_fingerprint,
        )
        reader = self._readers.get(reader_key)
        if reader is None:
            reader = _load_replay_cache(
                config,
                job_tag,
                normalizer,
                shard_cache=self._shard_cache,
            )
            # 缺失 cache 时本次调用可能触发编译，重新记录 manifest 指纹。
            cache_fingerprint = _cache_file_fingerprint(cache_path)
            self._cache_fingerprints[cache_key] = (source_fingerprint, cache_fingerprint)
            reader_key = (*reader_key[:-1], cache_fingerprint)
            self._readers[reader_key] = reader
        return reader


class AutoregressiveReplaySession:
    """可复用的 replay 运行时资源。

    session 顺序服务多条轨迹：SidecarHost、compiled cache reader/shard LRU
    和静态职业资源只初始化一次；每条轨迹由 :meth:`reset` 恢复到空战斗状态。
    session 不是线程安全对象，调用方必须串行运行轨迹。
    """

    def __init__(
        self,
        config: AutoregressiveReplayConfig,
        *,
        backend=None,
        cache_store: ReplayCacheStore | None = None,
    ):
        self.backend = _create_backend(config) if backend is None else backend
        self.device = self.backend.input_device
        self.data_spec = self.backend.data_spec
        self.input_contract = self.backend.input_contract
        if config.job_tag and self.data_spec.job_tag != config.job_tag:
            raise ValueError(
                f"replay job_tag mismatch: checkpoint={self.data_spec.job_tag!r} "
                f"configured={config.job_tag!r}"
            )
        self.vocab = SkillVocab.build_from_job_tag(self.data_spec.job_tag)
        validate_backend_vocab(self.backend, self.vocab)
        self.normalizer = self.input_contract.create_normalizer()
        self.cache_store = (
            ReplayCacheStore(max_shards=config.cache_max_shards)
            if cache_store is None
            else cache_store
        )
        project_config = load_project_config(job_tag=self.data_spec.job_tag)
        self.skill_book = SkillBook.from_project_config(project_config)
        mp_recovery = getattr(project_config.system, "mp_recovery", None)
        self.mp_tick_interval_seconds = (
            float(mp_recovery.tick_interval_seconds) if mp_recovery is not None else None
        )
        self.sidecar = SidecarBackend(
            job_tag=self.data_spec.job_tag,
            actual_base_gcd=config.base_gcd,
            max_history=config.max_history,
        )
        self._closed = False

    def load_reader(self, config: AutoregressiveReplayConfig):
        """读取指定场景的 reader，场景之间共享 shard LRU。"""
        if self._closed:
            raise RuntimeError("replay session is already closed")
        return self.cache_store.load(
            config,
            job_tag=self.data_spec.job_tag,
            normalizer=self.normalizer,
        )

    def reset(
        self,
        config: AutoregressiveReplayConfig,
        *,
        initial_timestamp: float = 0.0,
    ) -> float:
        """清空 Sidecar 与模型运行时缓存，开始一条独立轨迹。"""
        if self._closed:
            raise RuntimeError("replay session is already closed")
        if config.job_tag and config.job_tag != self.data_spec.job_tag:
            raise ValueError(
                f"replay session job_tag mismatch: {self.data_spec.job_tag!r} "
                f"configured={config.job_tag!r}"
            )
        self.backend.configure_cache(bool(getattr(config, "use_kv_cache", True)))
        fight_remaining = None
        scene_duration = getattr(config, "scene_duration_seconds", None)
        max_duration = getattr(config, "max_duration_seconds", None)
        if scene_duration is not None:
            fight_end = float(scene_duration)
            if max_duration is not None:
                fight_end = min(
                    fight_end,
                    float(initial_timestamp) + float(max_duration),
                )
            fight_remaining = max(0.0, fight_end - float(initial_timestamp))
        init_kwargs = {
            "actual_base_gcd": config.base_gcd,
            "max_history": config.max_history,
            "initial_timestamp": initial_timestamp,
        }
        if fight_remaining is not None:
            init_kwargs["fight_remaining"] = fight_remaining
        self.sidecar.init(**init_kwargs)
        return float(initial_timestamp)

    def close(self) -> None:
        """关闭长驻 SidecarHost；重复调用安全。"""
        if self._closed:
            return
        self._closed = True
        self.sidecar.close()

    def __enter__(self) -> "AutoregressiveReplaySession":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class AutoregressiveReplay:
    """加载 checkpoint，在 C# 状态机后端上逐步执行模型 Top-1 合法动作。"""

    def __init__(
        self,
        config: AutoregressiveReplayConfig,
        *,
        backend=None,
        session: AutoregressiveReplaySession | None = None,
    ):
        scene_duration = getattr(config, "scene_duration_seconds", None)
        scene_path = getattr(config, "scene_json_path", None)
        if scene_duration is None and scene_path is not None:
            scene_duration = _load_scene_duration_seconds(scene_path)
        if is_dataclass(config):
            self.config = replace(
                config,
                scene_duration_seconds=scene_duration,
            )
        else:
            # 测试中的轻量 SimpleNamespace 注入不具备 dataclass replace 契约。
            setattr(config, "scene_duration_seconds", scene_duration)
            self.config = config
        # 训练侧 rollout 可以注入一个共享 backend，避免同一轮 GRPO 为每条
        # 轨迹重复加载 checkpoint；普通回放仍由 config 创建 backend。
        if session is None:
            self._session = AutoregressiveReplaySession(config, backend=backend)
            self._owns_session = True
        else:
            if backend is not None and backend is not session.backend:
                raise ValueError("replay backend must match the supplied session")
            self._session = session
            self._owns_session = False
        try:
            self.backend = self._session.backend
            self.device = self._session.device
            self.data_spec = self._session.data_spec
            self.input_contract = self._session.input_contract
            self.vocab = self._session.vocab
            self._configure_kv_cache(bool(getattr(config, "use_kv_cache", True)))

            reader = self._session.load_reader(config)
            if reader.job_tag != self.data_spec.job_tag:
                raise ValueError(
                    f"scene compiled cache job_tag mismatch: {reader.job_tag!r} != {self.data_spec.job_tag!r}"
                )
            self.input_contract.schema.assert_compatible_with(reader.schema)
            if tuple(reader.skill_feature_names) != tuple(self.data_spec.skill_feature_names):
                raise ValueError("scene compiled cache skill feature layout mismatch")
            self.skill_book = self._session.skill_book
            self._mp_tick_interval_seconds = self._session.mp_tick_interval_seconds
            self._sidecar = self._session.sidecar
            scene_provider = SceneTemplateProvider(
                reader,
                normalizer=self._session.normalizer,
                initial_sample_index=config.scene_sample_index,
                enabled=config.scene_mode == "cache",
                backend=self._sidecar,
            )
            self.scene_provider = scene_provider
            self.batcher = LiveBatchBuilder(
                backend=self._sidecar,
                vocab=self.vocab,
                normalizer=self._session.normalizer,
                schema=reader.schema,
                skill_feature_names=reader.skill_feature_names,
                scene_provider=scene_provider,
                device=self.device,
                max_history=config.max_history,
                candidate_action_keys=self.data_spec.candidate_action_keys,
            )
        except BaseException:
            if self._owns_session:
                self._session.close()
            raise

    def close(self) -> None:
        """回收自有 replay session；共享 session 由其所有者关闭。"""
        if self._owns_session:
            self._session.close()

    def __enter__(self) -> "AutoregressiveReplay":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _sync_scene_state(self, state) -> object:
        sync_state = getattr(self.scene_provider, "sync_state", None)
        if callable(sync_state):
            return sync_state(state)
        return state

    def _observe_state(self, timestamp: float):
        return observe_replay_state(self._sidecar, timestamp)

    def _reset_for_trajectory(self, *, initial_timestamp: float = 0.0):
        """恢复独立轨迹的状态机与模型运行时缓存。"""
        scene_provider = getattr(self, "scene_provider", None)
        if scene_provider is not None:
            scene_provider.reset()
        session = getattr(self, "_session", None)
        if session is None:
            # 测试中可直接注入后端；正式 replay 始终使用 session。
            self._configure_kv_cache(bool(getattr(self.config, "use_kv_cache", True)))
            return self._observe_state(initial_timestamp)
        session.reset(self.config, initial_timestamp=initial_timestamp)
        return self._observe_state(initial_timestamp)

    def _next_scene_event_after(self, time_seconds: float) -> float | None:
        resolver = getattr(
            self.scene_provider,
            "next_state_event_after",
            None,
        )
        if callable(resolver):
            return resolver(time_seconds)
        resolver = getattr(
            self.scene_provider,
            "next_targetable_event_after",
            None,
        )
        return None if not callable(resolver) else resolver(time_seconds)

    def _advance_event_time(
        self,
        state,
        seconds: float,
        *,
        interrupt_on_scene_event: bool,
        end_time: float | None = None,
    ):
        """按共用决策调度器推进绝对时间。"""
        if float(seconds) < -EVENT_TIME_EPSILON:
            raise ValueError(f"event advance seconds must be >= 0, got {seconds}")
        return DecisionScheduler(
            self._sidecar,
            self._observe_state,
            self.scene_provider,
        ).advance_by(
            state,
            seconds,
            interrupt_on_scene_event=interrupt_on_scene_event,
            end_time=end_time,
        )

    def _advance_submitted_action(
        self,
        state,
        action_key,
        submission,
        *,
        end_time: float | None = None,
    ):
        """按提交结果推进到动作占用结束；实际推进由共用调度器负责。"""
        skill = self.skill_book.get(action_key)
        return DecisionScheduler(
            self._sidecar,
            self._observe_state,
            self.scene_provider,
        ).advance_submitted_action(
            state,
            submission,
            action_kind=skill.kind.value,
            end_time=end_time,
        )

    def run(self) -> ReplayResult:
        rows, _snapshots, final_state = self._generate_full_trajectory()
        return self._build_result(
            rows,
            history_limit=self.config.max_history,
            final_state=final_state,
        )

    def run_history_ablation(self, history_limits: tuple[int, ...]) -> tuple[ReplayResult, ...]:
        """先生成一条完整轨迹，再在相同决策状态上消融早期历史。"""
        if not history_limits:
            raise ValueError("history ablation requires at least one history limit")
        if any(limit < 0 for limit in history_limits):
            raise ValueError(f"history limits must be >= 0, got {history_limits}")
        if len(set(history_limits)) != len(history_limits):
            raise ValueError(f"history limits must be unique, got {history_limits}")

        full_rows, snapshots, final_state = self._generate_full_trajectory()
        results = [
            self._build_result(
                full_rows,
                history_limit=self.config.max_history,
                final_state=final_state,
            )
        ]
        self._configure_kv_cache(False)
        try:
            for history_limit in history_limits:
                rows = [row for row in full_rows if row.forced]
                for snapshot in snapshots:
                    row = self._predict_from_canonical(
                        snapshot.canonical,
                        gcd_step=snapshot.reference_row.gcd_step,
                        max_history=history_limit,
                    )
                    rows.append(
                        replace(
                            row,
                            reference_action_key=snapshot.reference_row.action_key,
                        )
                    )
                results.append(
                    self._build_result(
                        rows,
                        history_limit=history_limit,
                        is_history_ablation=True,
                    )
                )
        finally:
            self._configure_kv_cache(bool(getattr(self.config, "use_kv_cache", True)))
        return tuple(results)

    def _generate_full_trajectory(
        self,
    ) -> tuple[list[ReplayRow], tuple[ReplaySnapshot, ...], object]:
        # 一条新轨迹必须从空缓存开始；历史消融会走独立的完整 forward。
        initial_timestamp = 0.0
        if self.config.initial_action and self.config.initial_time_seconds is None:
            initial_skill = self.skill_book.get(self.config.initial_action)
            initial_timestamp = (
                -float(initial_skill.cast_time)
                if initial_skill.kind.value == GCD_ACTION_KIND
                else -decision_timing().ogcd_interval_seconds
            )
        state = self._reset_for_trajectory(initial_timestamp=initial_timestamp)
        action_limit = getattr(self.config, "max_steps", None)
        max_gcds = getattr(self.config, "max_gcds", None)
        end_time_candidates: list[float] = []
        scene_duration_seconds = getattr(self.config, "scene_duration_seconds", None)
        max_duration_seconds = getattr(self.config, "max_duration_seconds", None)
        if scene_duration_seconds is not None:
            end_time_candidates.append(float(scene_duration_seconds))
        if max_duration_seconds is not None:
            end_time_candidates.append(
                float(initial_timestamp) + float(max_duration_seconds)
            )
        if end_time_candidates:
            end_time = min(end_time_candidates)
        elif action_limit is None and max_gcds is None:
            # 没有 raw fights 元数据时，仍以状态机提供的战斗剩余时间为边界。
            end_time = float(state.time) + max(
                0.0,
                float(getattr(state, "fight_remaining", 0.0)),
            )
        else:
            # 普通回放仍由动作/GCD 上限控制；GRPO 传入 None 后才使用战斗时间。
            end_time = None
        gcd_step = 0
        rows = []

        if self.config.initial_action:
            initial_time = self.config.initial_time_seconds
            if initial_time is None:
                initial_time = initial_timestamp
            state = self._sync_scene_state(state)
            forced_result = self._sidecar.submit_action(
                float(initial_time),
                self.config.initial_action,
            )
            if not forced_result.accepted:
                raise RuntimeError(
                    f"initial action rejected: {self.config.initial_action} ({forced_result.reason})"
                )
            state = self._observe_state(float(initial_time))
            state = self._advance_submitted_action(
                state,
                self.config.initial_action,
                forced_result,
                end_time=end_time,
            )
            if self.skill_book.get(self.config.initial_action).kind.value == GCD_ACTION_KIND:
                gcd_step += 1
            rows.append(
                ReplayRow(
                    gcd_step=gcd_step,
                    action_key=self.config.initial_action,
                    probability=1.0,
                    top_candidates=(
                        (self.config.initial_action, 0.0, 1.0, True),
                    ),
                    forced=True,
                )
            )

        snapshots: list[ReplaySnapshot] = []
        event_count = 0
        max_event_count = (
            max(
                int(action_limit or 0) * EVENT_SAFETY_MULTIPLIER,
                int(max_gcds or 0) * EVENT_SAFETY_MULTIPLIER,
                1_024,
            )
            if action_limit is not None or max_gcds is not None
            else EVENT_SAFETY_LIMIT_WITHOUT_ACTION_CAP
        )
        while True:
            if end_time is not None and float(state.time) >= end_time - EVENT_TIME_EPSILON:
                break
            if action_limit is not None and len(rows) >= action_limit:
                break
            if max_gcds is not None and gcd_step >= max_gcds:
                break
            event_count += 1
            if event_count > max_event_count:
                raise RuntimeError(
                    "autoregressive replay exceeded its event safety limit "
                    f"({max_event_count}) before reaching the requested target"
                )
            state = self._sync_scene_state(state)
            next_observation = float(state.time) + gcd_request_delay(state)
            decision_canonical = self._sidecar.observe_at(
                float(state.time),
                format="vector",
                next_observation_timestamp=next_observation,
            ).context
            try:
                row = self._predict_row(
                    state,
                    gcd_step=gcd_step,
                )
            except NoLegalCandidateError:
                advanced = DecisionScheduler(
                    self._sidecar, self._observe_state, self.scene_provider,
                ).advance_to_next_decision(state, end_time=end_time)
                if advanced is None:
                    if end_time is not None and float(state.time) >= end_time - EVENT_TIME_EPSILON:
                        break
                    raise RuntimeError("live state has no legal candidate and no future time event") from None
                state = advanced
                continue
            if row.action_key == OGCD_WAIT_ACTION_KEY:
                wait_seconds = gcd_request_delay(state)
                if wait_seconds <= EVENT_TIME_EPSILON:
                    raise RuntimeError("selected ogcd_wait while GCD is already ready")
                next_observation = float(state.time) + gcd_request_delay(state)
                self._sidecar.record_policy_action(
                    float(state.time),
                    OGCD_WAIT_ACTION_KEY,
                    next_observation,
                )
                state = self._advance_event_time(
                    state,
                    wait_seconds,
                    interrupt_on_scene_event=False,
                    end_time=end_time,
                )
                snapshots.append(
                    ReplaySnapshot(
                        decision_canonical,
                        row,
                    )
                )
                rows.append(row)
                continue
            selected_skill = self.skill_book.get(row.action_key)
            result = self._sidecar.submit_action(float(state.time), row.action_key)
            if not result.accepted:
                raise RuntimeError(f"selected action rejected: {row.action_key} ({result.reason})")
            state = self._observe_state(float(state.time))
            state = self._advance_submitted_action(
                state,
                row.action_key,
                result,
                end_time=end_time,
            )

            if selected_skill.kind.value == GCD_ACTION_KIND:
                gcd_step += 1
            row = replace(row, gcd_step=gcd_step)
            snapshots.append(
                ReplaySnapshot(
                    decision_canonical,
                    row,
                )
            )
            rows.append(row)
        if max_gcds is not None and gcd_step < max_gcds:
            raise RuntimeError(
                "autoregressive replay did not reach requested GCD target: "
                f"{gcd_step} < {max_gcds} within "
                f"max_steps={action_limit}"
            )

        return rows, tuple(snapshots), state

    def _predict_row(
        self,
        state,
        *,
        gcd_step: int,
        max_history: int | None = None,
    ) -> ReplayRow:
        batch, candidate_keys = self.batcher.build(state, max_history=max_history)
        return self._score_row(
            batch,
            candidate_keys,
            gcd_step=gcd_step,
        )

    def _predict_from_canonical(
        self,
        canonical: dict[str, object],
        *,
        gcd_step: int,
        max_history: int,
    ) -> ReplayRow:
        batch, candidate_keys = self.batcher.build_from_canonical(
            canonical,
            max_history=max_history,
        )
        return self._score_row(
            batch,
            candidate_keys,
            gcd_step=gcd_step,
        )

    def _score_row(
        self,
        batch,
        candidate_keys: list[str],
        *,
        gcd_step: int,
    ) -> ReplayRow:
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
            probabilities = torch.softmax(legal_logits / self.config.temperature, dim=-1)
            probabilities = _apply_top_p(probabilities, order, self.config.top_p)
            selected_index = int(torch.multinomial(probabilities, 1).item())
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

    def _build_result(
        self,
        rows: list[ReplayRow],
        *,
        history_limit: int | None,
        is_history_ablation: bool = False,
        final_state=None,
    ) -> ReplayResult:
        output_gcds = max((row.gcd_step for row in rows), default=0)
        if final_state is None:
            cumulative_potency = cumulative_dot_potency = None
        elif hasattr(final_state, "cumulative_potency"):
            cumulative_potency = float(final_state.cumulative_potency)
            cumulative_dot_potency = float(final_state.cumulative_dot_potency)
        else:
            cumulative_potency, cumulative_dot_potency = read_replay_cumulative_potency(
                self._sidecar,
                float(final_state.time),
            )
        ppg = (
            None
            if output_gcds <= 0 or cumulative_potency is None or cumulative_dot_potency is None
            else (cumulative_potency + cumulative_dot_potency) / output_gcds
        )
        return ReplayResult(
            rows=tuple(rows),
            checkpoint_path=self.config.checkpoint_path,
            scene_json_path=self.config.scene_json_path,
            device=str(self.device),
            history_limit=history_limit,
            is_history_ablation=is_history_ablation,
            backend_name=self.backend.name,
            execution_provider=self.backend.execution_provider,
            model_source_path=self.backend.source_path,
            backend_metrics=self.backend.metrics().to_dict(),
            output_gcds=output_gcds,
            cumulative_potency=cumulative_potency,
            cumulative_dot_potency=cumulative_dot_potency,
            ppg=ppg,
        )

    def _configure_kv_cache(self, enabled: bool) -> None:
        """只负责通知模型开启或清空部署缓存，不持有 K/V 张量。"""
        backend = getattr(self, "backend", None)
        if backend is None:
            return
        backend.configure_cache(enabled)


def _create_backend(config: AutoregressiveReplayConfig) -> PolicyBackend:
    if config.backend == "onnxruntime":
        if config.onnx_package_path is None:
            raise ValueError("ONNX Runtime replay requires onnx_package_path")
        return OrtPolicyBackend(
            config.onnx_package_path,
            provider=config.ort_provider,
        )
    if config.checkpoint_path is None:
        raise ValueError("PyTorch replay requires checkpoint_path")
    return PyTorchPolicyBackend(
        config.checkpoint_path,
        device=config.device,
        use_kv_cache=config.use_kv_cache,
        precision=config.policy_precision,
    )


def _apply_top_p(
    probabilities: torch.Tensor,
    descending_order: torch.Tensor,
    top_p: float,
) -> torch.Tensor:
    """保留累计概率达到 top-p 的最小候选集合，并重新归一化。"""
    if top_p >= 1.0:
        return probabilities

    sorted_probabilities = probabilities[descending_order]
    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
    keep_sorted = torch.ones_like(sorted_probabilities, dtype=torch.bool)
    keep_sorted[1:] = cumulative_probabilities[:-1] < top_p
    kept_indices = descending_order[keep_sorted]

    filtered_probabilities = torch.zeros_like(probabilities)
    filtered_probabilities[kept_indices] = probabilities[kept_indices]
    return filtered_probabilities / filtered_probabilities.sum()


def _resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("autoregressive replay requested CUDA, but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cpu")
