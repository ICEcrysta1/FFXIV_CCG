"""raw JSON 转换结果的内存训练源读取器。"""

from __future__ import annotations

import math

from common.contracts import SCENE_CONTEXT_ABSOLUTE_MODE
from common.output_context_schema import CANONICAL_CONTEXT_SCHEMA_VERSION, STATE_SNAPSHOTS, STATE_VECTOR_GROUP_SCHEMAS, build_output_context_schema_metadata
from common.policy.data.state_features import read_state_tokens, validate_state_feature_keys
from common.torch_dependencies import import_torch
from common.policy.data.action_space import ActionSpace

from .source_helpers import (
    SKILL_ID_FIELD,
    SKILL_HISTORY_FIELDS,
    derive_skill_feature_names,
    to_optional_int,
)
from common.policy.data.schema import (
    SCENE_WINDOW_ORDER,
    SceneWindowSchema,
    TRAINING_SOURCE_FORMAT,
    TRAINING_SAMPLE_SCHEMA_VERSION,
    TrainingSchema,
    StateFeatureGroup,
)


class TrainingSourceReader:
    """读取转换脚本生成的内存 training payload，不落盘中间格式。"""

    def __init__(self, payload: dict[str, object]):
        self._torch = import_torch()
        if payload.get("sample_schema_version") != TRAINING_SAMPLE_SCHEMA_VERSION:
            raise ValueError("unsupported training sample schema version; recompile raw source")
        self._payload = payload
        samples = payload.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError("training payload must contain at least one sample")
        self._samples = samples
        self._schema = _build_training_source_schema(payload)
        self._job_tag = str(payload["job_tag"])
        self._fight_id = str(payload["fight_id"])
        self._num_samples = len(samples)
        action_space = ActionSpace.from_job_tag(self._job_tag)
        self._action_keys = action_space.action_keys
        self._state_layout = self.schema.state_layout(self._action_keys)
        self._action_to_vocab_id = action_space.action_to_vocab_id
        self._action_is_gcd = action_space.action_is_gcd
        for sample_index in range(self._num_samples):
            context = self._sample_context(sample_index)
            if context.get("schema_version") != CANONICAL_CONTEXT_SCHEMA_VERSION:
                raise ValueError("unsupported canonical context schema version; recompile raw source")
            if any("time_seconds" in row for row in self._history_rows(sample_index)):
                raise ValueError("removed skill time_seconds field; recompile raw source")
            if tuple(context.get("action_keys", ())) != self._action_keys:
                raise ValueError("training output action space must match enabled configured actions")
            for key in ("action_legal_mask", "action_values"):
                values = context.get(key)
                if not isinstance(values, list) or len(values) != self.num_actions:
                    raise ValueError(f"{key} must match the fixed output action space")
            if any(not isinstance(value, bool) for value in context["action_legal_mask"]):
                raise ValueError("action_legal_mask must contain booleans")
            values = context["action_values"]
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
                raise ValueError("action_values must contain numeric request-time values")
            current = context.get("current_state_context")
            if not isinstance(current, dict) or len(current.get("tokens", ())) != 1:
                raise ValueError("current_state_context must contain exactly one request state token")
            current_metadata = build_output_context_schema_metadata(context)
            if current_metadata["current_state_feature_keys"] != current_metadata["state_history_feature_keys"]:
                raise ValueError("current request and history state feature keys must match")
            read_state_tokens(current, self._state_layout)
            # 完整历史行只在 bank 构建时解析一次，这里只校验每步的字段元数据。
            validate_state_feature_keys(context["state_history_context"], self._state_layout)
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
    def ranking(self) -> dict[str, object] | None:
        return self._payload.get("ranking")

    @property
    def annotation_status(self) -> str:
        return str(self._payload.get("annotation_status", "unannotated"))

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @property
    def num_actions(self) -> int:
        return len(self._action_keys)

    @property
    def action_keys(self) -> tuple[str, ...]:
        return self._action_keys

    @property
    def action_to_vocab_id(self) -> tuple[int, ...]:
        return self._action_to_vocab_id

    @property
    def action_is_gcd(self) -> tuple[bool, ...]:
        return self._action_is_gcd

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

    def history_execution_metrics(self, sample_idx: int, *, max_history: int | None = None):
        """真实执行统计与模型状态分离，顺序与技能历史一一对应。"""
        state_context = self._sample_context(sample_idx)["state_history_context"]
        metrics = state_context.get("execution_metrics")
        if not isinstance(metrics, list) or len(metrics) != self.history_length(sample_idx):
            raise ValueError(f"history execution metrics length mismatch: sample={sample_idx}")
        if max_history is not None:
            if max_history < 0:
                raise ValueError(f"max_history must be >= 0, got {max_history}")
            metrics = [] if max_history == 0 else metrics[-max_history:]
        return metrics

    def state_matrix_from_tokens(self, tokens, *, dtype):
        """把已经选好的状态 token 批量转为向量；供增量历史 bank 一次性构建。"""
        if dtype != self._torch.float32:
            raise ValueError("raw state matrices require FP32 before delta encoding")
        context = {group.feature_keys_field: group.feature_keys for group in self._state_layout.groups}
        return read_state_tokens({**context, "tokens": tokens}, self._state_layout)

    def action_legal_mask(self, sample_idx: int):
        return self._torch.tensor(
            self._sample_context(sample_idx)["action_legal_mask"],
            dtype=self._torch.bool,
        )

    def action_values(self, sample_idx: int, *, dtype):
        return self._torch.tensor(self._sample_context(sample_idx)["action_values"], dtype=dtype)

    def current_state_matrix(self, sample_idx: int, *, dtype):
        if dtype != self._torch.float32:
            raise ValueError("raw state matrices require FP32 before delta encoding")
        return read_state_tokens(self._sample_context(sample_idx)["current_state_context"], self._state_layout)

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


def _build_training_source_schema(payload: dict[str, object]) -> TrainingSchema:
    samples = payload["samples"]
    assert isinstance(samples, list) and samples
    first_sample = samples[0]
    if not isinstance(first_sample, dict):
        raise ValueError("first training sample must be a mapping")
    first_context = first_sample.get("context")
    if not isinstance(first_context, dict):
        raise ValueError("first training sample context must be a mapping")
    if first_context.get("schema_version") != CANONICAL_CONTEXT_SCHEMA_VERSION:
        raise ValueError("unsupported canonical context schema version; recompile raw source")

    context_schema = build_output_context_schema_metadata(first_context)
    history_state_keys = context_schema["state_history_feature_keys"]
    current_state_keys = context_schema["current_state_feature_keys"]
    if history_state_keys != current_state_keys:
        raise ValueError("state history and current state feature keys must match")

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

    return TrainingSchema(
        serialization_format=TRAINING_SOURCE_FORMAT,
        sample_schema_version=int(payload.get("sample_schema_version", 0)),
        context_schema_version=int(context_schema["schema_version"]),
        scene_context_mode=SCENE_CONTEXT_ABSOLUTE_MODE,
        scene_windows=tuple(scene_windows),
        state_groups=tuple(StateFeatureGroup(group.group_key, group.feature_keys_field,
                                            tuple(history_state_keys[group.group_key]), group.encoding)
                           for group in STATE_VECTOR_GROUP_SCHEMAS),
        state_snapshots=STATE_SNAPSHOTS,
        skill_history_fields=SKILL_HISTORY_FIELDS,
    )
