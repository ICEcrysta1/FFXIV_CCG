"""raw JSON 转换结果的内存训练源读取器。"""

from __future__ import annotations

from common.contracts import SCENE_CONTEXT_ABSOLUTE_MODE
from common.output_context_schema import build_output_context_schema_metadata
from common.torch_dependencies import import_torch

from .source_helpers import (
    SKILL_ID_FIELD,
    build_skill_feature_matrix,
    derive_skill_feature_names,
    extract_history_after_value,
    require_numeric_skill_kind,
    to_optional_int,
)
from common.policy.data.schema import (
    SCENE_WINDOW_ORDER,
    SceneWindowSchema,
    TRAINING_SOURCE_FORMAT,
    TrainingSchema,
)


class TrainingSourceReader:
    """读取转换脚本生成的内存 training payload，不落盘中间格式。"""

    def __init__(self, payload: dict[str, object]):
        self._torch = import_torch()
        self._payload = payload
        samples = payload.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError("training payload must contain at least one sample")
        self._samples = samples
        self._schema = _build_training_source_schema(payload)
        self._job_tag = str(payload["job_tag"])
        self._fight_id = str(payload["fight_id"])
        self._num_samples = len(samples)
        first_context = self._sample_context(0)
        candidate_rows = first_context["candidate_skill_context"]
        if not isinstance(candidate_rows, list) or not candidate_rows:
            raise ValueError("training payload must contain candidate skill rows")
        self._num_candidates = len(candidate_rows)
        for sample_index in range(self._num_samples):
            context = self._sample_context(sample_index)
            rows = context.get("candidate_skill_context")
            if not isinstance(rows, list) or len(rows) != self._num_candidates:
                raise ValueError("candidate count must stay stable across a training payload")
            for candidate_index, row in enumerate(rows):
                if not isinstance(row, dict):
                    raise ValueError(
                        f"candidate skill row must be a mapping: sample={sample_index} "
                        f"index={candidate_index}"
                    )
                require_numeric_skill_kind(
                    row,
                    context=(
                        f"candidate sample={sample_index} index={candidate_index} "
                        f"fight={self.fight_id}"
                    ),
                )
        self._skill_feature_names = derive_skill_feature_names(self)

    @property
    def schema(self) -> TrainingSchema:
        return self._schema

    @property
    def job_tag(self) -> str:
        return self._job_tag

    @property
    def fight_id(self) -> str:
        return self._fight_id

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @property
    def num_candidates(self) -> int:
        return self._num_candidates

    @property
    def skill_feature_names(self) -> tuple[str, ...]:
        return self._skill_feature_names

    def step_metadata(self, sample_idx: int) -> dict[str, object]:
        sample = self._sample(sample_idx)
        return {
            "fight_id": self.fight_id,
            "job_tag": self.job_tag,
            "step": int(sample.get("step", 0)),
            "source_step": int(sample.get("source_step", sample.get("step", 0))),
            "time_offset": float(sample.get("time_offset", 0.0)),
        }

    def label(self, sample_idx: int) -> dict[str, object]:
        label = self._sample(sample_idx).get("label", {})
        return dict(label) if isinstance(label, dict) else {}

    def history_length(self, sample_idx: int) -> int:
        return len(self._history_rows(sample_idx))

    def history_tail(self, sample_idx: int) -> tuple[dict[str, object], dict[str, object]]:
        """返回一个样本历史末尾的技能行与状态 token。

        编译增量历史缓存只需要每个新动作对应的一行，因此这里避免把完整历史
        前缀重新装配成 tensor。历史技能和历史状态必须保持一一对应；不一致时
        直接失败，避免生成无法和旧 dense cache 对齐的缓存。
        """
        rows = self._history_rows(sample_idx)
        context = self._sample_context(sample_idx)
        state_context = context.get("state_history_context", {})
        tokens = state_context.get("tokens", []) if isinstance(state_context, dict) else []
        if len(rows) != len(tokens):
            raise ValueError(
                "skill/state history length mismatch: "
                f"sample={sample_idx} skills={len(rows)} states={len(tokens)}"
            )
        if not rows:
            raise ValueError(f"history tail is empty for sample={sample_idx}")
        state_token = tokens[-1]
        if not isinstance(state_token, dict):
            raise ValueError(f"history state tail must be a mapping: sample={sample_idx}")
        return dict(rows[-1]), dict(state_token)

    def history_delta(
        self,
        sample_idx: int,
        start: int,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        """返回该样本从 ``start`` 开始新增的技能与状态历史。

        绝对时间队列下，相邻决策可能共享同一段已生效历史，也可能在两个决策间
        一次结算多条事件；compiled bank 因此按实际新增行构建，不能再按样本号取尾。
        """
        rows = self._history_rows(sample_idx)
        context = self._sample_context(sample_idx)
        state_context = context.get("state_history_context", {})
        tokens = state_context.get("tokens", []) if isinstance(state_context, dict) else []
        if len(rows) != len(tokens):
            raise ValueError(
                "skill/state history length mismatch: "
                f"sample={sample_idx} skills={len(rows)} states={len(tokens)}"
            )
        if not 0 <= start <= len(rows):
            raise ValueError(
                f"history delta start out of range: sample={sample_idx} "
                f"start={start} length={len(rows)}"
            )
        delta_tokens = tokens[start:]
        if not all(isinstance(token, dict) for token in delta_tokens):
            raise ValueError(f"history state delta must contain mappings: sample={sample_idx}")
        return (
            [dict(row) for row in rows[start:]],
            [dict(token) for token in delta_tokens],
        )

    def history_action_keys(self, sample_idx: int, *, max_history: int | None = None) -> list[str]:
        return [
            str(row.get("skill_key", ""))
            for row in self.history_skill_rows(sample_idx, max_history=max_history)
        ]

    def history_skill_ids(self, sample_idx: int, *, max_history: int | None = None) -> list[int | None]:
        return [
            to_optional_int(row.get(SKILL_ID_FIELD))
            for row in self.history_skill_rows(sample_idx, max_history=max_history)
        ]

    def history_skill_rows(
        self,
        sample_idx: int,
        *,
        max_history: int | None = None,
    ) -> list[dict[str, object]]:
        rows = self._history_rows(sample_idx)
        if max_history is not None:
            if max_history < 0:
                raise ValueError(f"max_history must be >= 0, got {max_history}")
            rows = [] if max_history == 0 else rows[-max_history:]
        return [dict(row) for row in rows]

    def history_skill_feature_matrix(self, sample_idx: int, *, max_history: int | None, dtype):
        return build_skill_feature_matrix(
            self.history_skill_rows(sample_idx, max_history=max_history),
            feature_names=self.skill_feature_names,
            torch=self._torch,
            dtype=dtype,
        )

    def history_skill_metrics(
        self,
        sample_idx: int,
        *,
        max_history: int | None = None,
    ) -> tuple[list[float], list[float]]:
        """返回历史技能原始威力与累计 DoT，供验证 PPG 使用。"""
        rows = self.history_skill_rows(sample_idx, max_history=max_history)
        context = self._sample_context(sample_idx)
        state_context = context.get("state_history_context", {})
        tokens = state_context.get("tokens", []) if isinstance(state_context, dict) else []
        if max_history is not None:
            if max_history < 0:
                raise ValueError(f"max_history must be >= 0, got {max_history}")
            tokens = [] if max_history == 0 else tokens[-max_history:]
        if len(rows) != len(tokens):
            raise ValueError(
                "skill/state history length mismatch: "
                f"sample={sample_idx} skills={len(rows)} states={len(tokens)}"
            )

        potencies: list[float] = []
        cumulative_dot_potencies: list[float] = []
        for index, (row, token) in enumerate(zip(rows, tokens)):
            context_label = f"history sample={sample_idx} index={index} fight={self.fight_id}"
            require_numeric_skill_kind(row, context=context_label)
            potencies.append(float(row.get("potency", 0.0)))
            if not isinstance(token, dict):
                raise ValueError(f"{context_label} state history token must be a mapping")
            cumulative_dot_potencies.append(
                extract_history_after_value(
                    token,
                    self.schema,
                    feature_name="target.cumulative_dot_potency",
                    context=context_label,
                )
            )
        return potencies, cumulative_dot_potencies

    def history_state_matrix(
        self,
        sample_idx: int,
        *,
        max_history: int | None = None,
        dtype,
        normalizer=None,
    ):
        context = self._sample_context(sample_idx)
        state_context = context["state_history_context"]
        tokens = state_context.get("tokens", []) if isinstance(state_context, dict) else []
        if max_history is not None:
            if max_history < 0:
                raise ValueError(f"max_history must be >= 0, got {max_history}")
            tokens = [] if max_history == 0 else tokens[-max_history:]
        return self._build_state_matrix(tokens, dtype=dtype, normalizer=normalizer)

    def state_matrix_from_tokens(self, tokens, *, dtype, normalizer=None):
        """把已经选好的状态 token 批量转为向量；供增量历史 bank 一次性构建。"""
        return self._build_state_matrix(tokens, dtype=dtype, normalizer=normalizer)

    def candidate_action_keys(self, sample_idx: int) -> list[str]:
        return [
            str(row.get("skill_key", ""))
            for row in self.candidate_skill_rows(sample_idx)
        ]

    def candidate_skill_ids(self, sample_idx: int) -> list[int | None]:
        return [
            to_optional_int(row.get(SKILL_ID_FIELD))
            for row in self.candidate_skill_rows(sample_idx)
        ]

    def candidate_invalid_reasons(self, sample_idx: int) -> list[str]:
        return [
            str(row.get("invalid_reason", ""))
            for row in self.candidate_skill_rows(sample_idx)
        ]

    def candidate_skill_rows(self, sample_idx: int) -> list[dict[str, object]]:
        context = self._sample_context(sample_idx)
        rows = context["candidate_skill_context"]
        if not isinstance(rows, list):
            raise ValueError("candidate_skill_context must be a list")
        return [dict(row) for row in rows]

    def candidate_skill_feature_matrix(self, sample_idx: int, *, dtype):
        return build_skill_feature_matrix(
            self.candidate_skill_rows(sample_idx),
            feature_names=self.skill_feature_names,
            torch=self._torch,
            dtype=dtype,
        )

    def candidate_legal_mask(self, sample_idx: int):
        return self._torch.tensor(
            [bool(row.get("is_legal", True)) for row in self.candidate_skill_rows(sample_idx)],
            dtype=self._torch.bool,
        )

    def candidate_state_matrix(self, sample_idx: int, *, dtype, normalizer=None):
        context = self._sample_context(sample_idx)
        state_context = context["candidate_state_context"]
        tokens = state_context.get("tokens", []) if isinstance(state_context, dict) else []
        return self._build_state_matrix(tokens, dtype=dtype, normalizer=normalizer)

    def scene_tokens(self, sample_idx: int, *, float_dtype, int_dtype):
        feature_dim = self.schema.scene_feature_dim()
        if feature_dim == 0:
            return (
                self._torch.zeros((0, 0), dtype=float_dtype),
                self._torch.zeros((0,), dtype=int_dtype),
            )

        self._sample(sample_idx)
        vectors = []
        scene_types: list[int] = []
        fight_scene_context = self._payload.get("fight_scene_context", {})
        if not isinstance(fight_scene_context, dict):
            fight_scene_context = {}

        for window_schema in self.schema.scene_windows:
            context_payload = fight_scene_context.get(window_schema.context_key, {})
            if not isinstance(context_payload, dict):
                continue
            raw_tokens = context_payload.get("tokens", [])
            if not isinstance(raw_tokens, list) or not raw_tokens:
                continue
            token_values = self._torch.tensor(raw_tokens, dtype=float_dtype)
            if token_values.shape[-1] == feature_dim:
                scene_vectors = token_values
            else:
                scene_vectors = self._pad_scene_tokens(
                    token_values,
                    dtype=float_dtype,
                    feature_dim=feature_dim,
                )
            vectors.append(scene_vectors)
            scene_types.extend([window_schema.scene_type_id] * scene_vectors.shape[0])

        if not vectors:
            return (
                self._torch.zeros((0, feature_dim), dtype=float_dtype),
                self._torch.zeros((0,), dtype=int_dtype),
            )
        return self._torch.cat(vectors, dim=0), self._torch.tensor(scene_types, dtype=int_dtype)

    def _build_state_matrix(self, tokens, *, dtype, normalizer=None):
        values: list[list[float]] = []
        nulls: list[list[bool]] = []
        for token in tokens:
            row_values: list[float] = []
            row_nulls: list[bool] = []
            for group_key, feature_keys in self.schema.state_group_feature_keys.items():
                group_values = token.get(group_key, []) if isinstance(token, dict) else []
                if len(group_values) != len(feature_keys):
                    raise ValueError(
                        f"state group width mismatch for {group_key!r}: "
                        f"{len(group_values)} != {len(feature_keys)}"
                    )
                for value in group_values:
                    is_null = value is None
                    row_values.append(0.0 if is_null else float(value))
                    row_nulls.append(is_null)
            values.append(row_values)
            nulls.append(row_nulls)

        width = self.schema.state_vector_dim()
        if not values:
            return _nullable_tensor(
                self._torch.zeros((0, width), dtype=dtype),
                self._torch.zeros((0, width), dtype=self._torch.bool),
            )

        tensor = self._torch.tensor(values, dtype=dtype)
        null_mask = self._torch.tensor(nulls, dtype=self._torch.bool)
        if normalizer is not None:
            cursor = 0
            for group_key, feature_keys in self.schema.state_group_feature_keys.items():
                group_width = len(feature_keys)
                group_values = normalizer.normalize(
                    tensor[:, cursor : cursor + group_width],
                    group_key,
                    null_mask=null_mask[:, cursor : cursor + group_width],
                )
                tensor[:, cursor : cursor + group_width] = group_values
                cursor += group_width
        else:
            tensor.masked_fill_(null_mask, -1.0)
        return _nullable_tensor(tensor, null_mask)

    def _pad_scene_tokens(
        self,
        tokens,
        *,
        dtype,
        feature_dim: int,
    ):
        scene_tokens = tokens.to(dtype=dtype)
        if scene_tokens.shape[-1] == feature_dim:
            return scene_tokens
        padded = self._torch.zeros((scene_tokens.shape[0], feature_dim), dtype=dtype)
        padded[:, : scene_tokens.shape[-1]] = scene_tokens
        return padded

    def _sample(self, sample_idx: int) -> dict[str, object]:
        if not 0 <= sample_idx < self._num_samples:
            raise IndexError(f"training sample index out of range: {sample_idx}")
        sample = self._samples[sample_idx]
        if not isinstance(sample, dict):
            raise ValueError(f"training sample {sample_idx} must be a mapping")
        return sample

    def _sample_context(self, sample_idx: int) -> dict[str, object]:
        context = self._sample(sample_idx).get("context")
        if not isinstance(context, dict):
            raise ValueError(f"training sample {sample_idx} context must be a mapping")
        return context

    def _history_rows(self, sample_idx: int) -> list[dict[str, object]]:
        rows = self._sample_context(sample_idx).get("skill_history_context", [])
        if not isinstance(rows, list):
            raise ValueError("skill_history_context must be a list")
        return [dict(row) for row in rows]


class _NullableTensor:
    def __init__(self, values, null_mask):
        self.values = values
        self.null_mask = null_mask


def _nullable_tensor(values, null_mask):
    return _NullableTensor(values, null_mask)


def _build_training_source_schema(payload: dict[str, object]) -> TrainingSchema:
    samples = payload["samples"]
    assert isinstance(samples, list) and samples
    first_sample = samples[0]
    if not isinstance(first_sample, dict):
        raise ValueError("first training sample must be a mapping")
    first_context = first_sample.get("context")
    if not isinstance(first_context, dict):
        raise ValueError("first training sample context must be a mapping")

    context_schema = build_output_context_schema_metadata(first_context)
    history_state_keys = context_schema["state_history_feature_keys"]
    candidate_state_keys = context_schema["candidate_state_feature_keys"]
    if history_state_keys != candidate_state_keys:
        raise ValueError("state history and candidate state feature keys must match")

    fight_scene_context = payload.get("fight_scene_context", {})
    if not isinstance(fight_scene_context, dict):
        fight_scene_context = {}
    scene_windows = []
    for context_key, scene_type_id in SCENE_WINDOW_ORDER:
        context_payload = fight_scene_context.get(context_key, {})
        feature_keys = ()
        if isinstance(context_payload, dict):
            feature_keys = tuple(context_payload.get("feature_keys", ()))
        scene_windows.append(
            SceneWindowSchema.from_feature_keys(
                context_key=context_key,
                feature_keys=feature_keys,
                scene_type_id=scene_type_id,
            )
        )

    candidate_fields = set()
    history_fields = set()
    for sample in samples:
        if not isinstance(sample, dict) or not isinstance(sample.get("context"), dict):
            continue
        context = sample["context"]
        candidate_rows = context.get("candidate_skill_context", [])
        history_rows = context.get("skill_history_context", [])
        for row in candidate_rows if isinstance(candidate_rows, list) else []:
            if isinstance(row, dict):
                candidate_fields.update(row)
        for row in history_rows if isinstance(history_rows, list) else []:
            if isinstance(row, dict):
                history_fields.update(row)

    return TrainingSchema(
        serialization_format=TRAINING_SOURCE_FORMAT,
        sample_schema_version=int(payload.get("sample_schema_version", 0)),
        context_schema_version=int(context_schema["schema_version"]),
        scene_context_mode=SCENE_CONTEXT_ABSOLUTE_MODE,
        scene_windows=tuple(scene_windows),
        state_group_feature_keys={
            str(group_key): tuple(feature_keys)
            for group_key, feature_keys in history_state_keys.items()
        },
        candidate_skill_fields=tuple(sorted(candidate_fields)),
        skill_history_fields=tuple(sorted(history_fields)),
    )
