"""把 live 状态机输出翻译成训练模型 batch。"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass

import torch

from common.numeric import flatten_numeric_mapping
from common.output_context_schema import CANONICAL_CONTEXT_SCHEMA_VERSION
from common.policy.data.context_encoding import ContextEncoder, raw_state_delta
from common.policy.data.history_window import history_window_length
from common.policy.data.state_features import read_state_tokens, validate_state_feature_keys
from scripts.common.scene_state import SceneFactScheduler, rewrite_scene_player_state

from .scheduler import gcd_request_delay, is_gcd_decision

REPLAY_SKILL_IGNORED_FIELDS = frozenset(
    {"skill_key", "skill_name", "invalid_reason", "skill_id"}
)


class SceneTemplateProvider:
    """完整 raw scene 的容器适配；查询和事实调度只由 scene_state 持有。"""

    def __init__(self, reader, *, initial_sample_index=0, enabled=True, backend=None):
        if reader.num_samples <= 0:
            raise ValueError("scene cache has no samples")
        if not 0 <= initial_sample_index < reader.num_samples:
            raise ValueError(f"scene sample index {initial_sample_index} out of range: {reader.num_samples} samples")
        self._backend = backend
        scene_dim = reader.schema.scene_feature_dim()
        if enabled:
            self._scene_abs_values, self._scene_types = reader.scene_tokens(
                initial_sample_index, float_dtype=torch.float32, int_dtype=torch.int32,
            )
        else:
            self._scene_abs_values = torch.zeros((0, scene_dim), dtype=torch.float32)
            self._scene_types = torch.zeros((0,), dtype=torch.int32)
        if enabled and self._scene_abs_values.shape[0] == 0:
            raise ValueError("scene cache has no usable scene tokens")
        if self._scene_abs_values.dtype != torch.float32:
            raise ValueError("raw scene cache must use float32")
        self._facts = SceneFactScheduler(_scene_context_from_tensors(
            reader.schema, self._scene_abs_values, self._scene_types,
        ))

    def reset(self) -> None:
        self._facts.reset()

    def at_time(self, time_seconds: float):
        del time_seconds
        return self._scene_abs_values, self._scene_types

    def state_at(self, time_seconds: float):
        return self._facts.state_at(time_seconds)

    def sync_state(self, state):
        if self._backend is not None:
            self._facts.sync_through(self._backend, float(state.time))
        return state

    def last_targetable_end(self) -> float:
        return self._facts.lookup.last_targetable_end()

    def next_state_event_after(self, time_seconds: float) -> float | None:
        return self._facts.next_event_after(time_seconds)


def _scene_context_from_tensors(schema, values, types):
    """只适配保存的列和容器；精度、窗口与边界规则委托共享场景父级。"""
    rows, type_ids = values.tolist(), types.tolist()
    if len(rows) != len(type_ids):
        raise ValueError("raw scene values and types must align")
    windows = {window.scene_type_id: window for window in schema.scene_windows}
    if any(type_id not in windows for type_id in type_ids):
        raise ValueError("raw scene contains an unknown scene type")
    return {
        window.context_key: {
            "feature_keys": list(window.feature_keys),
            "tokens": [row[:len(window.feature_keys)] for row, type_id in zip(rows, type_ids, strict=True)
                       if type_id == window.scene_type_id],
        }
        for window in schema.scene_windows
    }


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
    state_skill_availability: torch.Tensor
    action_key: str


class LiveBatchBuilder:
    """使用 compiled cache 的 schema 和 scene 模板构造 live 推理输入。

    上下文原料（历史/动作）统一来自 C# 后端 observe_at(format="vector"),
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
        model_config,
        action_keys=None,
        action_is_gcd=None,
    ):
        self._backend = backend
        self._vocab = vocab
        self._normalizer = normalizer
        self._schema = schema
        if schema.context_schema_version != CANONICAL_CONTEXT_SCHEMA_VERSION:
            raise ValueError("unsupported saved canonical context schema version; rebuild input artifacts")
        self._skill_feature_names = tuple(skill_feature_names)
        self._scene_provider = scene_provider
        self._device = device
        self._max_history = max_history
        self._model_config = model_config
        self._action_keys = (
            None
            if action_keys is None
            else tuple(str(key) for key in action_keys)
        )
        if self._action_keys is None or action_is_gcd is None:
            raise ValueError("live batch requires the saved action vocabulary and kinds")
        kinds = tuple(action_is_gcd)
        if len(kinds) != len(self._action_keys) or any(type(value) is not bool for value in kinds):
            raise ValueError("saved action kinds must be boolean and align with action vocabulary")
        self._action_kinds = dict(zip(self._action_keys, kinds, strict=True))
        self._layout = schema.state_layout(self._action_keys)
        self._context_encoder = ContextEncoder(
            normalizer, schema, model_config, layout=self._layout,
        ).to(device=device)
        self._state_dim = self._layout.base_state_dim
        self._scene_dim = schema.scene_feature_dim()
        player_group = next(group for group in self._layout.base_groups if group.group_key == "player_state")
        self._history_request_time_index = player_group.feature_keys.index("request_state.time_seconds")
        self._cached_history_rows: list[_CachedHistoryFeatureRow] = []
        self._cached_history_max_history: int | None = None
        self._cached_scene_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._cached_history_device_rows: tuple[_CachedHistoryFeatureRow, ...] | None = None
        self._cached_history_device_tensors: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ] | None = None
        self._history_cursor: int | None = None

    @property
    def context_metadata(self) -> dict[str, int]:
        """完整历史游标属于宿主审计信息，不进入模型 tensor batch。"""
        if self._history_cursor is None:
            raise RuntimeError("live context metadata requires a successfully built context")
        return {"history_cursor": self._history_cursor}

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
            gcd_phase=is_gcd_decision(state.gcd_remaining),
            max_history=self._max_history if max_history is None else max_history,
        )

    def build_from_canonical(
        self,
        canonical,
        *,
        gcd_phase: bool,
        max_history: int,
    ):
        """复用决策上下文和调用方阶段；调度计时不从模型特征反推。"""
        return self._build_batch_from_canonical(
            canonical,
            gcd_phase=gcd_phase,
            max_history=max_history,
        )

    def _build_batch_from_canonical(
        self,
        canonical,
        *,
        gcd_phase: bool,
        max_history: int,
    ):
        if canonical.get("schema_version") != self._schema.context_schema_version:
            raise ValueError("live canonical context schema version differs from saved input contract")
        canonical = rewrite_scene_player_state(canonical, scene_state_at=self._scene_provider.state_at)
        # 元数据始终校验；历史数值仅解析未命中的不可变行，保留增量缓存收益。
        history_context = canonical["state_history_context"]
        validate_state_feature_keys(history_context, self._layout)
        current_state = read_state_tokens(canonical["current_state_context"], self._layout)
        action_keys = tuple(str(key) for key in canonical["action_keys"])
        if not action_keys or len(set(action_keys)) != len(action_keys):
            raise ValueError("live action vocabulary must be nonempty and unique")
        if self._action_keys is not None and action_keys != self._action_keys:
            raise ValueError("live action order differs from policy DataSpec")
        current_tokens = canonical["current_state_context"]["tokens"]
        if len(current_tokens) != 1:
            raise ValueError("live context must contain exactly one current state")
        cursor = canonical.get("history_cursor")
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise ValueError("live canonical context requires a nonnegative history_cursor")
        if isinstance(max_history, bool) or not isinstance(max_history, int) or max_history < 0:
            raise ValueError("max_history must be a non-negative integer")
        # 正式重置周期只由模型配置决定；消融仅裁短已经选出的窗口。
        visible_length = min(history_window_length(
            cursor, self._model_config.history_capacity, self._model_config.history_reset_keep,
        ), max_history)
        if len(canonical["skill_history_context"]) < visible_length:
            raise ValueError("retained canonical history cannot supply the planned context window")
        if cursor < len(canonical["skill_history_context"]):
            raise ValueError("history_cursor cannot be smaller than retained canonical history")
        skill_history = _tail_history(canonical["skill_history_context"], visible_length)
        state_history = canonical["state_history_context"]
        state_history_tokens = _tail_history(state_history["tokens"], visible_length)
        (
            history_skill_ids,
            history_skill_features,
            history_state_vectors,
            history_state_null_mask,
            history_state_skill_availability,
            history_action_keys,
        ) = self._build_history_batch(
            skill_history,
            state_history_tokens,
            state_context=history_context,
            max_history=max_history,
        )
        scene_abs_values, scene_types, scene_mask = self._scene_batch_tensors()
        current_state_vectors, current_state_null_mask = current_state.base_values, current_state.base_null_mask
        current_state_vectors = current_state_vectors.to(self._device)
        current_state_null_mask = current_state_null_mask.to(self._device)
        previous_values = torch.cat((torch.zeros_like(history_state_vectors[:1]), history_state_vectors[:-1]), dim=0)
        previous_nulls = torch.cat((torch.ones_like(history_state_null_mask[:1]), history_state_null_mask[:-1]), dim=0)
        state_deltas, state_resets = raw_state_delta(
            history_state_vectors, history_state_null_mask, previous_values, previous_nulls,
        )
        current_deltas, current_resets = raw_state_delta(
            current_state_vectors,
            current_state_null_mask,
            history_state_vectors[-1:] if visible_length else None,
            history_state_null_mask[-1:] if visible_length else None,
        )
        action_legal_mask = torch.tensor(canonical["action_legal_mask"], dtype=torch.bool)
        if action_legal_mask.shape != (len(action_keys),):
            raise ValueError("live action legality width differs from action vocabulary")
        try:
            phase_mask = torch.tensor([self._action_kinds[key] == gcd_phase for key in action_keys], dtype=torch.bool)
        except KeyError as exc:
            raise ValueError("live action is absent from the configured action space") from exc
        action_legal_mask &= phase_mask

        history_length = len(history_action_keys)
        batch = {
            "scene_abs_values": scene_abs_values,
            "scene_types": scene_types,
            "scene_mask": scene_mask,
            "history_skill_ids": history_skill_ids.unsqueeze(0),
            "history_skill_features": history_skill_features.unsqueeze(0),
            "history_state_abs_values": history_state_vectors.unsqueeze(0),
            "history_state_delta_values": state_deltas.unsqueeze(0),
            "history_state_null_mask": history_state_null_mask.unsqueeze(0),
            "history_state_delta_reset_mask": state_resets.unsqueeze(0),
            "history_state_skill_availability": history_state_skill_availability.unsqueeze(0),
            "history_mask": torch.ones(
                (1, history_length),
                dtype=torch.bool,
                device=self._device,
            ),
            "current_state_abs_values": current_state_vectors,
            "current_state_delta_values": current_deltas,
            "current_state_null_mask": current_state_null_mask,
            "current_state_delta_reset_mask": current_resets,
            "current_state_skill_availability": current_state.availability,
            "action_legal_mask": action_legal_mask.unsqueeze(0),
            "history_action_keys": [history_action_keys],
            "action_keys": [list(action_keys)],
        }
        prepared = self._context_encoder.encode({
            key: value.to(self._device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        })
        self._history_cursor = cursor
        return prepared, list(action_keys)

    def _build_history_batch(
        self,
        skill_tokens,
        state_tokens,
        *,
        state_context,
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
            self._cached_history_device_rows = None
            self._cached_history_device_tensors = None

        identities = [
            self._history_row_identity(skill_token, state_token)
            for skill_token, state_token in zip(skill_tokens, state_tokens, strict=True)
        ]
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
                    state_context=state_context,
                )
            )

        # 输入已经按 max_history 裁剪；这里再限制一次，避免缓存持有旧窗口。
        self._cached_history_rows = rows[-max_history:] if max_history > 0 else []
        rows = self._cached_history_rows
        history_tensors = self._history_tensors_for_rows(rows)
        return (*history_tensors, [row.action_key for row in rows])

    def _scene_batch_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """每个验证副本只把固定 scene 张量搬到目标设备一次。"""
        if self._cached_scene_batch is None:
            scene_abs_values, scene_types = self._scene_provider.at_time(0.0)
            scene_abs_values = scene_abs_values.unsqueeze(0).to(self._device)
            scene_types = scene_types.unsqueeze(0).to(self._device)
            scene_mask = torch.ones(
                (1, scene_abs_values.shape[1]),
                dtype=torch.bool,
                device=self._device,
            )
            self._cached_scene_batch = (scene_abs_values, scene_types, scene_mask)
        return self._cached_scene_batch

    def _history_tensors_for_rows(
        self,
        rows: list[_CachedHistoryFeatureRow],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """复用设备端历史窗口；只传新行，并正确处理窗口滑动或重建。"""
        current_rows = tuple(rows)
        previous_rows = self._cached_history_device_rows
        previous_tensors = self._cached_history_device_tensors

        if previous_rows is None or previous_tensors is None:
            tensors = self._tensorize_history_rows(current_rows)
        elif self._same_history_rows(current_rows, previous_rows):
            tensors = previous_tensors
        elif len(current_rows) < len(previous_rows) and self._same_history_rows(
            current_rows,
            previous_rows[: len(current_rows)],
        ):
            tensors = tuple(tensor[: len(current_rows)] for tensor in previous_tensors)
        elif len(current_rows) > len(previous_rows) and self._same_history_rows(
            previous_rows,
            current_rows[: len(previous_rows)],
        ):
            new_tensors = self._tensorize_history_rows(current_rows[len(previous_rows) :])
            tensors = self._append_history_tensors(previous_tensors, new_tensors)
        else:
            overlap = 0
            for candidate_overlap in range(min(len(previous_rows), len(current_rows)), 0, -1):
                if self._same_history_rows(
                    previous_rows[-candidate_overlap:],
                    current_rows[:candidate_overlap],
                ):
                    overlap = candidate_overlap
                    break
            if overlap == 0:
                tensors = self._tensorize_history_rows(current_rows)
            else:
                dropped_rows = len(previous_rows) - overlap
                base_tensors = tuple(
                    tensor[dropped_rows:] for tensor in previous_tensors
                )
                new_rows = current_rows[overlap:]
                if new_rows:
                    new_tensors = self._tensorize_history_rows(new_rows)
                    tensors = self._append_history_tensors(base_tensors, new_tensors)
                else:
                    tensors = base_tensors

        self._cached_history_device_rows = current_rows
        self._cached_history_device_tensors = tensors
        return tensors

    @staticmethod
    def _same_history_rows(left, right) -> bool:
        return len(left) == len(right) and all(
            left_row is right_row for left_row, right_row in zip(left, right)
        )

    def _tensorize_history_rows(
        self,
        rows: tuple[_CachedHistoryFeatureRow, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not rows:
            return (
                torch.empty((0,), dtype=torch.int32, device=self._device),
                torch.empty(
                    (0, len(self._skill_feature_names)),
                    dtype=torch.float32,
                    device=self._device,
                ),
                torch.empty((0, self._state_dim), dtype=torch.float32, device=self._device),
                torch.empty((0, self._state_dim), dtype=torch.bool, device=self._device),
                torch.empty((0, self._layout.availability_dim), dtype=torch.bool, device=self._device),
            )
        return (
            torch.tensor(
                [row.skill_id for row in rows],
                dtype=torch.int32,
                device=self._device,
            ),
            torch.stack([row.skill_features for row in rows]).to(self._device),
            torch.stack([row.state_vector for row in rows]).to(self._device),
            torch.stack([row.state_null_mask for row in rows]).to(self._device),
            torch.stack([row.state_skill_availability for row in rows]).to(self._device),
        )

    @staticmethod
    def _append_history_tensors(base_tensors, new_tensors):
        return tuple(
            torch.cat((base, new), dim=0)
            for base, new in zip(base_tensors, new_tensors)
        )

    def _history_row_identity(self, skill_token, state_token) -> tuple[object, ...]:
        """按技能身份和冻结请求时间定位；同刻重复行仍校验完整技能、状态。"""
        player_state = state_token.get("player_state")
        if not isinstance(player_state, (list, tuple)) or len(player_state) <= self._history_request_time_index:
            raise ValueError("live history state lacks request_state.time_seconds")
        request_time = player_state[self._history_request_time_index]
        if isinstance(request_time, bool) or not isinstance(request_time, (int, float)) or not math.isfinite(request_time):
            raise ValueError("live history request_state.time_seconds must be finite numeric")
        return (
            skill_token.get("skill_key"),
            skill_token.get("skill_id"),
            float(request_time),
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
        *,
        state_context,
    ) -> _CachedHistoryFeatureRow:
        """把一条历史 token 转成可跨决策复用的 CPU 特征行。"""
        skill_features = self._build_skill_features([skill_token])[0]
        parsed = read_state_tokens({**state_context, "tokens": [state_token]}, self._layout)
        return _CachedHistoryFeatureRow(
            identity=identity,
            skill_token=deepcopy(skill_token),
            state_token=deepcopy(state_token),
            skill_id=self._map_skill_id(skill_token.get("skill_id")),
            skill_features=skill_features.detach().cpu(),
            state_vector=parsed.base_values[0],
            state_null_mask=parsed.base_null_mask[0],
            state_skill_availability=parsed.availability[0],
            action_key=str(skill_token.get("skill_key", "")),
        )

    def _map_skill_id(self, raw_skill_id) -> int:
        return self._vocab.require_lookup(raw_skill_id, context="live replay")

    def _build_skill_features(self, tokens) -> torch.Tensor:
        feature_rows = []
        for token in tokens:
            if "time_seconds" in token:
                raise ValueError("live skill token must not include time_seconds; rebuild old input artifacts")
            flattened = flatten_numeric_mapping(
                token,
                ignored_keys=REPLAY_SKILL_IGNORED_FIELDS,
            )
            feature_rows.append(
                [
                    float(flattened.get(feature_name, 0.0))
                    for feature_name in self._skill_feature_names
                ]
            )
        if feature_rows:
            values = torch.tensor(feature_rows, dtype=torch.float32)
        else:
            values = torch.zeros(
                (0, len(self._skill_feature_names)),
                dtype=torch.float32,
            )
        return self._normalizer.normalize_skill_features(values, self._skill_feature_names)


def _tail_history(values, max_history: int):
    """保留最近的历史；0 明确表示不提供历史。"""
    if max_history < 0:
        raise ValueError(f"max_history must be >= 0, got {max_history}")
    if max_history == 0:
        return []
    return values[-max_history:]
