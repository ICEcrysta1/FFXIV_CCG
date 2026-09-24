"""把 live 状态机输出翻译成训练模型 batch。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch

from common.contracts import SLIDECAST_WINDOW_SECONDS
from common.numeric import flatten_numeric_mapping
from common.policy.data.candidate_order import candidate_permutation
from common.policy.data.schema import (
    SCENE_TYPE_MOVEMENT,
    SCENE_TYPE_RAID_BUFF,
    SCENE_TYPE_TARGET_COUNT,
    SCENE_TYPE_TARGETABLE,
)

from .scheduler import gcd_request_delay, is_gcd_decision

REPLAY_SKILL_IGNORED_FIELDS = frozenset(
    {"skill_key", "skill_name", "invalid_reason", "skill_id"}
)
SCENE_TIME_EPSILON = 1e-6


@dataclass(frozen=True)
class SceneTargetableState:
    """某个绝对时间点对应的 Boss 可选中状态。"""

    boss_targetable: bool
    next_downtime_eta: float | None
    downtime_remaining: float


class SceneTemplateProvider:
    """从 compiled cache 固定读取已经归一化的 scene token。"""

    def __init__(
        self,
        reader,
        *,
        normalizer,
        initial_sample_index: int = 0,
        enabled: bool = True,
        backend=None,
    ):
        if reader.num_samples <= 0:
            raise ValueError("scene cache has no samples")
        if not 0 <= initial_sample_index < reader.num_samples:
            raise ValueError(
                f"scene sample index {initial_sample_index} out of range: {reader.num_samples} samples"
            )
        self._reader = reader
        self._enabled = enabled
        self._normalizer = normalizer
        self._backend = backend
        self._last_external_signature = None
        self._scene_dim = reader.schema.scene_feature_dim()
        targetable_window = next(
            (
                window
                for window in reader.schema.scene_windows
                if window.scene_type_id == SCENE_TYPE_TARGETABLE
                and "targetable" in window.feature_keys
            ),
            None,
        )
        target_count_window = next(
            (
                window
                for window in reader.schema.scene_windows
                if window.scene_type_id == SCENE_TYPE_TARGET_COUNT
                and "target_count" in window.feature_keys
            ),
            None,
        )
        self._target_count_index = (
            None
            if target_count_window is None
            else target_count_window.feature_keys.index("target_count")
        )
        self._targetable_flag_index = (
            None
            if targetable_window is None
            else targetable_window.feature_keys.index("targetable")
        )
        self._targetable_start_index = (
            None if targetable_window is None else targetable_window.start_offset_index
        )
        self._targetable_end_index = (
            None if targetable_window is None else targetable_window.end_offset_index
        )
        self._scene_start_index = (
            None
            if target_count_window is None
            else target_count_window.feature_keys.index("start_offset_seconds")
        )
        self._scene_end_index = (
            None
            if target_count_window is None
            else target_count_window.feature_keys.index("end_offset_seconds")
        )
        if self._enabled:
            self._scene_vectors, self._scene_types = reader.scene_tokens(
                initial_sample_index,
                float_dtype=torch.float32,
                int_dtype=torch.int32,
            )
        else:
            self._scene_vectors = torch.zeros((0, self._scene_dim), dtype=torch.float32)
            self._scene_types = torch.zeros((0,), dtype=torch.int32)
        if self._enabled and self._scene_vectors.shape[0] == 0:
            raise ValueError("scene cache has no usable scene tokens")
        if self._enabled:
            self._validate_normalized_scene_times(reader.schema)
        self._targetable_windows = self._build_targetable_windows()
        self._targetable_events = self._build_targetable_events()
        self._movement_windows = self._build_plain_windows(SCENE_TYPE_MOVEMENT)
        self._raid_buff_windows = self._build_plain_windows(SCENE_TYPE_RAID_BUFF)
        self._scene_events = self._build_scene_events()

    def reset(self) -> None:
        """重置轨迹级外部事实缓存，确保新轨迹首帧重新同步。"""
        self._last_external_signature = None

    def at_time(self, time_seconds: float):
        del time_seconds
        return self._scene_vectors, self._scene_types

    def target_count_at(self, time_seconds: float) -> int:
        """从 cache scene 模板读取当前时间的目标数量。"""
        if self._target_count_index is None:
            return 1
        target = self._normalizer.normalize_scene_time(time_seconds)
        scene_vectors, scene_types = self.at_time(time_seconds)
        for vector, scene_type in zip(scene_vectors, scene_types):
            if int(scene_type.item()) != SCENE_TYPE_TARGET_COUNT:
                continue
            start = float(vector[self._scene_start_index].item())
            end = float(vector[self._scene_end_index].item())
            count = self._normalizer.denormalize_scene_target_count(
                float(vector[self._target_count_index].item())
            )
            if start - SCENE_TIME_EPSILON <= target < end - SCENE_TIME_EPSILON:
                return count
        return 1

    def sync_state(self, state) -> object:
        """把 scene 时间线中的状态变化作为带时间戳的外部事实提交。

        状态机接收 Boss 可选中、移动、目标数和团辅窗口四类外部事实。
        移动窗口在滑步豁免点提前结束，允许已经接近完成的读条自然结算。
        相同快照不会重复提交。
        """
        if self._backend is None:
            return state
        current_time = float(state.time)
        raid_buff_end = self._active_window_end(
            self._raid_buff_windows,
            current_time,
        )
        targetable = self.targetable_state_at(current_time)
        moving = self.is_moving_at(current_time)
        signature = (
            bool(targetable.boss_targetable),
            bool(moving),
            int(self.target_count_at(current_time)),
            raid_buff_end is not None,
        )
        if signature == self._last_external_signature:
            return state
        self._backend.apply_external_event(
            current_time,
            "boss_targetable_changed",
            value=signature[0],
        )
        self._backend.apply_external_event(
            current_time,
            "movement_changed",
            value=signature[1],
        )
        self._backend.apply_external_event(
            current_time,
            "target_count_changed",
            target_count=signature[2],
        )
        self._backend.apply_external_event(
            current_time,
            "raid_buff_window_changed",
            value=signature[3],
            **(
                {}
                if raid_buff_end is None
                else {"remaining_seconds": max(0.0, raid_buff_end - current_time)}
            ),
        )
        self._last_external_signature = signature
        return state

    def targetable_at(self, time_seconds: float) -> bool:
        """供时间推进内核判断延迟伤害发生时 Boss 是否可选中。"""
        return self.targetable_state_at(time_seconds).boss_targetable

    def is_moving_at(self, time_seconds: float) -> bool:
        """返回考虑固定滑步豁免后的移动状态。"""
        current_time = float(time_seconds)
        return any(
            start - SCENE_TIME_EPSILON <= current_time
            and (end - current_time)
            > SLIDECAST_WINDOW_SECONDS + SCENE_TIME_EPSILON
            for start, end in self._movement_windows
        )

    def targetable_windows(self) -> list[dict[str, object]]:
        """返回状态机时间推进所需的绝对时间可选中窗口。"""
        return [
            {
                "start": float(start),
                "end": float(end),
                "targetable": bool(targetable),
            }
            for start, end, targetable in self._targetable_windows
        ]

    def last_targetable_end(self) -> float:
        """返回缓存 scene 中最后一个 Boss 可选中窗口的结束时间。"""
        targetable_ends = [
            end for _start, end, targetable in self._targetable_windows if targetable
        ]
        if not targetable_ends:
            raise ValueError("scene cache has no targetable Boss window")
        return max(targetable_ends)

    def targetable_state_at(self, time_seconds: float) -> SceneTargetableState:
        """解析当前可选中状态、下次上天 ETA 和剩余停机时间。"""
        current_time = float(time_seconds)
        active = next(
            (
                window
                for window in self._targetable_windows
                if window[0] - SCENE_TIME_EPSILON
                <= current_time
                < window[1] - SCENE_TIME_EPSILON
            ),
            None,
        )
        future_downtime_starts = [
            start
            for start, _end, targetable in self._targetable_windows
            if not targetable and start > current_time + SCENE_TIME_EPSILON
        ]
        next_downtime_eta = (
            None
            if not future_downtime_starts
            else max(0.0, min(future_downtime_starts) - current_time)
        )
        if active is None:
            return SceneTargetableState(True, next_downtime_eta, 0.0)
        _start, end, targetable = active
        if targetable:
            return SceneTargetableState(True, next_downtime_eta, 0.0)
        return SceneTargetableState(
            False,
            0.0,
            max(0.0, end - current_time),
        )

    def next_targetable_event_after(self, time_seconds: float) -> float | None:
        """返回下一次 Boss 上天/落地的绝对时间。"""
        current_time = float(time_seconds)
        return next(
            (
                event_time
                for event_time in self._targetable_events
                if event_time > current_time + SCENE_TIME_EPSILON
            ),
            None,
        )

    def next_state_event_after(self, time_seconds: float) -> float | None:
        """返回下一次会改变状态向量或候选合法性的 scene 边界。"""
        current_time = float(time_seconds)
        return next(
            (
                event_time
                for event_time in self._scene_events
                if event_time > current_time + SCENE_TIME_EPSILON
            ),
            None,
        )

    def _build_targetable_windows(self) -> tuple[tuple[float, float, bool], ...]:
        if (
            not self._enabled
            or self._targetable_flag_index is None
            or self._targetable_start_index is None
            or self._targetable_end_index is None
        ):
            return ()
        windows = []
        fight_time_max = float(self._normalizer.fight_time_max)
        for vector, scene_type in zip(self._scene_vectors, self._scene_types):
            if int(scene_type.item()) != SCENE_TYPE_TARGETABLE:
                continue
            start = float(vector[self._targetable_start_index].item()) * fight_time_max
            end = float(vector[self._targetable_end_index].item()) * fight_time_max
            if end <= start + SCENE_TIME_EPSILON:
                continue
            windows.append(
                (
                    start,
                    end,
                    bool(float(vector[self._targetable_flag_index].item()) >= 0.5),
                )
            )
        return tuple(sorted(windows, key=lambda item: (item[0], item[1])))

    def _build_targetable_events(self) -> tuple[float, ...]:
        boundaries = sorted(
            {
                boundary
                for start, end, _targetable in self._targetable_windows
                for boundary in (start, end)
                if boundary > SCENE_TIME_EPSILON
            }
        )
        events = []
        for boundary in boundaries:
            before = self.targetable_at(max(0.0, boundary - SCENE_TIME_EPSILON * 10.0))
            after = self.targetable_at(boundary + SCENE_TIME_EPSILON * 10.0)
            if before != after:
                events.append(boundary)
        return tuple(events)

    def _build_plain_windows(self, scene_type_id: int) -> tuple[tuple[float, float], ...]:
        if not self._enabled:
            return ()
        window_schema = next(
            (
                window
                for window in self._reader.schema.scene_windows
                if window.scene_type_id == scene_type_id
            ),
            None,
        )
        if window_schema is None:
            return ()
        fight_time_max = float(self._normalizer.fight_time_max)
        windows = []
        for vector, scene_type in zip(self._scene_vectors, self._scene_types):
            if int(scene_type.item()) != scene_type_id:
                continue
            start = float(vector[window_schema.start_offset_index].item()) * fight_time_max
            end = float(vector[window_schema.end_offset_index].item()) * fight_time_max
            if end > start + SCENE_TIME_EPSILON:
                windows.append((start, end))
        return tuple(sorted(windows))

    def _build_scene_events(self) -> tuple[float, ...]:
        boundaries = set(self._targetable_events)
        for start, end in self._movement_windows:
            boundaries.add(start)
            boundaries.add(max(start, end - SLIDECAST_WINDOW_SECONDS))
        for start, end in self._raid_buff_windows:
            boundaries.update((start, end))
        if self._scene_start_index is not None and self._scene_end_index is not None:
            fight_time_max = float(self._normalizer.fight_time_max)
            for vector, scene_type in zip(self._scene_vectors, self._scene_types):
                if int(scene_type.item()) != SCENE_TYPE_TARGET_COUNT:
                    continue
                boundaries.add(float(vector[self._scene_start_index].item()) * fight_time_max)
                boundaries.add(float(vector[self._scene_end_index].item()) * fight_time_max)
        return tuple(
            sorted(
                boundary
                for boundary in boundaries
                if boundary > SCENE_TIME_EPSILON
            )
        )

    @staticmethod
    def _active_window_end(
        windows: tuple[tuple[float, float], ...],
        time_seconds: float,
    ) -> float | None:
        return next(
            (
                end
                for start, end in windows
                if start - SCENE_TIME_EPSILON
                <= time_seconds
                < end - SCENE_TIME_EPSILON
            ),
            None,
        )

    def _validate_normalized_scene_times(self, schema) -> None:
        """拒绝把原始秒值 scene token 误当作 cache token 使用。"""
        epsilon = SCENE_TIME_EPSILON
        for window in schema.scene_windows:
            start_index = getattr(
                window,
                "start_offset_index",
                window.feature_keys.index("start_offset_seconds"),
            )
            end_index = getattr(
                window,
                "end_offset_index",
                window.feature_keys.index("end_offset_seconds"),
            )
            start_values = self._scene_vectors[:, start_index]
            end_values = self._scene_vectors[:, end_index]
            if bool(
                (
                    (start_values < -epsilon)
                    | (start_values > 1.0 + epsilon)
                    | (end_values < -epsilon)
                    | (end_values > 1.0 + epsilon)
                ).any().item()
            ):
                raise ValueError(
                    "scene cache time fields must be normalized by "
                    "Normalizer.normalize_scene_tokens"
                )


@dataclass(frozen=True)
class _CachedHistoryFeatureRow:
    """单条不可变历史事件对应的模型特征。"""

    identity: tuple[object, ...]
    skill_token: dict[str, object]
    state_token: dict[str, object]
    skill_id: int
    skill_features: torch.Tensor
    state_vector: torch.Tensor
    state_null_mask: torch.Tensor
    action_key: str


class LiveBatchBuilder:
    """使用 compiled cache 的 schema 和 scene 模板构造 live 推理输入。

    上下文原料（历史/候选）统一来自 C# 后端 observe_at(format="vector"),
    scene 向量来自 scene provider（模型输入，与状态机解耦）。
    """

    def __init__(
        self,
        *,
        backend,
        vocab,
        normalizer,
        schema,
        skill_feature_names,
        scene_provider,
        device,
        max_history: int,
        candidate_action_keys=None,
    ):
        self._backend = backend
        self._vocab = vocab
        self._normalizer = normalizer
        self._schema = schema
        self._skill_feature_names = tuple(skill_feature_names)
        self._scene_provider = scene_provider
        self._device = device
        self._max_history = max_history
        self._candidate_action_keys = (
            None
            if candidate_action_keys is None
            else tuple(str(key) for key in candidate_action_keys)
        )
        self._state_groups = tuple(schema.state_group_feature_keys)
        self._state_dim = schema.state_vector_dim()
        self._scene_dim = schema.scene_feature_dim()
        self._cached_history_rows: list[_CachedHistoryFeatureRow] = []
        self._cached_history_max_history: int | None = None

    def build(
        self,
        state,
        *,
        max_history: int | None = None,
        next_observation_timestamp: float | None = None,
    ):
        sync_state = getattr(self._scene_provider, "sync_state", None)
        if callable(sync_state):
            state = sync_state(state)
        observation_timestamp = float(state.time)
        next_timestamp = (
            observation_timestamp + gcd_request_delay(state)
            if next_observation_timestamp is None
            else max(observation_timestamp, float(next_observation_timestamp))
        )
        canonical = self._backend.observe_at(
            observation_timestamp,
            format="vector",
            next_observation_timestamp=next_timestamp,
        ).context
        return self._build_batch_from_canonical(
            canonical,
            max_history=self._max_history if max_history is None else max_history,
        )

    def build_from_canonical(
        self,
        canonical,
        *,
        max_history: int,
    ):
        """从缓存的决策时刻上下文直接构建 batch（历史消融旁路，不驱动后端）。"""
        return self._build_batch_from_canonical(
            canonical,
            max_history=max_history,
        )

    def _build_batch_from_canonical(
        self,
        canonical,
        *,
        max_history: int,
    ):
        candidate_context = canonical["candidate_skill_context"]
        candidate_state_context = canonical["candidate_state_context"]
        if not candidate_context:
            raise RuntimeError("no candidate actions in live state")
        feature_keys = candidate_state_context["player_state_feature_keys"]
        remaining_index = feature_keys.index("before.gcd_remaining_seconds")
        remaining = candidate_state_context["tokens"][0]["player_state"][remaining_index]
        gcd_phase = is_gcd_decision(remaining)
        source_candidate_keys = tuple(
            str(token.get("skill_key", "")) for token in candidate_context
        )
        if self._candidate_action_keys is not None:
            permutation = candidate_permutation(
                source_candidate_keys,
                self._candidate_action_keys,
            )
            candidate_context = [candidate_context[index] for index in permutation]
            state_tokens = candidate_state_context.get("tokens", [])
            if len(state_tokens) != len(permutation):
                raise ValueError("candidate skill/state rows must have the same length")
            candidate_state_context = dict(candidate_state_context)
            candidate_state_context["tokens"] = [
                state_tokens[index] for index in permutation
            ]
        skill_history = _tail_history(canonical["skill_history_context"], max_history)
        state_history = canonical["state_history_context"]
        state_history_tokens = _tail_history(state_history["tokens"], max_history)
        (
            history_skill_ids,
            history_skill_features,
            history_state_vectors,
            history_state_null_mask,
            history_action_keys,
        ) = self._build_history_batch(
            skill_history,
            state_history_tokens,
            max_history=max_history,
        )
        # scene provider 的 at_time 忽略时间参数：scene 模板固定整场
        scene_vectors, scene_types = self._scene_provider.at_time(0.0)
        candidate_skill_ids = torch.tensor(
            [self._map_skill_id(token.get("skill_id")) for token in candidate_context],
            dtype=torch.int32,
        )
        candidate_skill_features = self._build_skill_features(candidate_context)
        candidate_state_vectors, candidate_state_null_mask = self._build_state_tensors(
            candidate_state_context["tokens"]
        )
        candidate_legal_mask = torch.tensor(
            [
                bool(token.get("is_legal", False))
                and ((int(token["kind"]) == 1) == gcd_phase)
                for token in candidate_context
            ],
            dtype=torch.bool,
        )

        history_length = len(history_action_keys)
        batch = {
            "scene_vectors": scene_vectors.unsqueeze(0),
            "scene_types": scene_types.unsqueeze(0),
            "scene_mask": torch.ones(
                (1, scene_vectors.shape[0]),
                dtype=torch.bool,
            ),
            "history_skill_ids": history_skill_ids.unsqueeze(0),
            "history_skill_features": history_skill_features.unsqueeze(0),
            "history_state_vectors": history_state_vectors.unsqueeze(0),
            "history_state_null_mask": history_state_null_mask.unsqueeze(0),
            "history_mask": torch.ones((1, history_length), dtype=torch.bool),
            "candidate_skill_ids": candidate_skill_ids.unsqueeze(0),
            "candidate_skill_features": candidate_skill_features.unsqueeze(0),
            "candidate_state_vectors": candidate_state_vectors.unsqueeze(0),
            "candidate_state_null_mask": candidate_state_null_mask.unsqueeze(0),
            "candidate_legal_mask": candidate_legal_mask.unsqueeze(0),
            "history_action_keys": [history_action_keys],
            "candidate_action_keys": [
                [str(token.get("skill_key", "")) for token in candidate_context]
            ],
        }
        return (
            {
                key: value.to(self._device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            },
            [str(token.get("skill_key", "")) for token in candidate_context],
        )

    def _build_history_batch(
        self,
        skill_tokens,
        state_tokens,
        *,
        max_history: int,
    ):
        """只转换新增历史行；缓存不匹配时安全地重建当前窗口。"""
        if len(skill_tokens) != len(state_tokens):
            raise ValueError(
                "live skill and state history rows must have the same length: "
                f"skills={len(skill_tokens)} states={len(state_tokens)}"
            )

        if self._cached_history_max_history != max_history:
            self._cached_history_rows.clear()
            self._cached_history_max_history = max_history

        identities = [self._history_row_identity(token) for token in skill_tokens]
        cached_rows, overlap = self._find_history_overlap(
            skill_tokens,
            state_tokens,
            identities,
        )
        rows = list(cached_rows)
        for index in range(overlap, len(skill_tokens)):
            rows.append(
                self._build_cached_history_row(
                    skill_tokens[index],
                    state_tokens[index],
                    identities[index],
                )
            )

        # 输入已经按 max_history 裁剪；这里再限制一次，避免缓存持有旧窗口。
        self._cached_history_rows = rows[-max_history:] if max_history > 0 else []
        rows = self._cached_history_rows
        if not rows:
            return (
                torch.empty((0,), dtype=torch.int32),
                torch.empty((0, len(self._skill_feature_names)), dtype=torch.float32),
                torch.empty((0, self._state_dim), dtype=torch.float32),
                torch.empty((0, self._state_dim), dtype=torch.bool),
                [],
            )

        return (
            torch.tensor([row.skill_id for row in rows], dtype=torch.int32),
            torch.stack([row.skill_features for row in rows]),
            torch.stack([row.state_vector for row in rows]),
            torch.stack([row.state_null_mask for row in rows]),
            [row.action_key for row in rows],
        )

    @staticmethod
    def _history_row_identity(skill_token) -> tuple[object, ...]:
        """用稳定的事件字段定位历史行；完整 token 仍会用于缓存命中校验。"""
        return (
            skill_token.get("time_seconds"),
            skill_token.get("gcd_index"),
            skill_token.get("skill_key"),
            skill_token.get("skill_id"),
        )

    def _find_history_overlap(
        self,
        skill_tokens,
        state_tokens,
        identities: list[tuple[object, ...]],
    ) -> tuple[list[_CachedHistoryFeatureRow], int]:
        """识别追加或滑动窗口的重合区，并拒绝复用已变化的历史行。"""
        prefix_length = 0
        for index, row in enumerate(self._cached_history_rows):
            if index >= len(skill_tokens):
                break
            if (
                row.identity != identities[index]
                or row.skill_token != skill_tokens[index]
                or row.state_token != state_tokens[index]
            ):
                break
            prefix_length += 1
        if prefix_length:
            return self._cached_history_rows[:prefix_length], prefix_length

        max_overlap = min(len(self._cached_history_rows), len(skill_tokens))
        for overlap in range(max_overlap, 0, -1):
            cached_rows = self._cached_history_rows[-overlap:]
            if any(
                row.identity != identities[index]
                or row.skill_token != skill_tokens[index]
                or row.state_token != state_tokens[index]
                for index, row in enumerate(cached_rows)
            ):
                continue
            return cached_rows, overlap
        return [], 0

    def _build_cached_history_row(
        self,
        skill_token,
        state_token,
        identity: tuple[object, ...],
    ) -> _CachedHistoryFeatureRow:
        """把一条历史 token 转成可跨决策复用的 CPU 特征行。"""
        skill_features = self._build_skill_features([skill_token])[0]
        state_vectors, state_null_mask = self._build_state_tensors([state_token])
        return _CachedHistoryFeatureRow(
            identity=identity,
            skill_token=deepcopy(skill_token),
            state_token=deepcopy(state_token),
            skill_id=self._map_skill_id(skill_token.get("skill_id")),
            skill_features=skill_features.detach().cpu(),
            state_vector=state_vectors[0].detach().cpu(),
            state_null_mask=state_null_mask[0].detach().cpu(),
            action_key=str(skill_token.get("skill_key", "")),
        )

    def _map_skill_id(self, raw_skill_id) -> int:
        if raw_skill_id is None:
            return 0
        return self._vocab.require_lookup(int(raw_skill_id), context="live replay")

    def _build_skill_features(self, tokens) -> torch.Tensor:
        values = torch.zeros(
            (len(tokens), len(self._skill_feature_names)),
            dtype=torch.float32,
        )
        for row_index, token in enumerate(tokens):
            flattened = flatten_numeric_mapping(
                token,
                ignored_keys=REPLAY_SKILL_IGNORED_FIELDS,
            )
            for feature_index, feature_name in enumerate(self._skill_feature_names):
                values[row_index, feature_index] = float(flattened.get(feature_name, 0.0))
        return self._normalizer.normalize_skill_features(values, self._skill_feature_names)

    def _build_state_tensors(self, tokens):
        values = []
        null_masks = []
        for group_key in self._state_groups:
            group_values = []
            group_null_masks = []
            for token in tokens:
                group_value, group_null_mask = _extract_nullable_vector(
                    token.get(group_key),
                )
                if group_value.shape[0] != len(self._schema.state_group_feature_keys[group_key]):
                    raise ValueError(
                        f"live {group_key} vector width mismatch: "
                        f"{group_value.shape[0]} != {len(self._schema.state_group_feature_keys[group_key])}"
                    )
                group_values.append(group_value)
                group_null_masks.append(group_null_mask)
            if group_values:
                group_tensor = torch.stack(group_values)
                group_null_tensor = torch.stack(group_null_masks)
            else:
                width = len(self._schema.state_group_feature_keys[group_key])
                group_tensor = torch.zeros((0, width), dtype=torch.float32)
                group_null_tensor = torch.zeros((0, width), dtype=torch.bool)
            group_tensor = self._normalizer.normalize(
                group_tensor,
                group_key,
                null_mask=group_null_tensor,
            )
            values.append(group_tensor)
            null_masks.append(group_null_tensor)
        if not values:
            return (
                torch.zeros((0, self._state_dim), dtype=torch.float32),
                torch.zeros((0, self._state_dim), dtype=torch.bool),
            )
        return torch.cat(values, dim=-1), torch.cat(null_masks, dim=-1)


def _tail_history(values, max_history: int):
    """保留最近的历史；0 明确表示不提供历史。"""
    if max_history < 0:
        raise ValueError(f"max_history must be >= 0, got {max_history}")
    if max_history == 0:
        return []
    return values[-max_history:]


def _extract_nullable_vector(node):
    if not isinstance(node, (list, tuple)):
        raise ValueError("live state vector group must be a list")
    values = []
    null_mask = []
    for item in node:
        if item is None:
            values.append(0.0)
            null_mask.append(True)
        elif isinstance(item, bool):
            values.append(1.0 if item else 0.0)
            null_mask.append(False)
        elif isinstance(item, (int, float)):
            values.append(float(item))
            null_mask.append(False)
        else:
            raise ValueError(f"unsupported live state vector value: {item!r}")
    return torch.tensor(values, dtype=torch.float32), torch.tensor(null_mask, dtype=torch.bool)
