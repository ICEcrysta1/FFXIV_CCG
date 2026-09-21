"""raw source 与 compiled cache 共用的训练 schema。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from common.contracts import (
    FORCED_MOVEMENT_CONTEXT_KEY,
    RAID_BUFF_WINDOW_CONTEXT_KEY,
    TARGET_COUNT_WINDOW_CONTEXT_KEY,
    TARGETABLE_WINDOW_CONTEXT_KEY,
)
from common.scene_window import resolve_scene_window_indices

SCENE_TYPE_TARGETABLE = 0
SCENE_TYPE_MOVEMENT = 1
SCENE_TYPE_RAID_BUFF = 2
SCENE_TYPE_TARGET_COUNT = 3
TRAINING_SOURCE_FORMAT = "raw_training_source_v1"


@dataclass(frozen=True)
class SceneWindowSchema:
    """单类 scene window 的 schema。"""

    context_key: str
    feature_keys: tuple[str, ...]
    scene_type_id: int
    start_offset_index: int
    end_offset_index: int
    duration_index: int

    @classmethod
    def from_feature_keys(
        cls,
        *,
        context_key: str,
        feature_keys: tuple[str, ...],
        scene_type_id: int,
    ) -> "SceneWindowSchema":
        """从 scene 字段名一次性解析并校验窗口索引。"""
        start_offset_index, end_offset_index, duration_index = resolve_scene_window_indices(
            feature_keys,
            context_key=context_key,
        )
        return cls(
            context_key=context_key,
            feature_keys=feature_keys,
            scene_type_id=scene_type_id,
            start_offset_index=start_offset_index,
            end_offset_index=end_offset_index,
            duration_index=duration_index,
        )

    @property
    def feature_dim(self) -> int:
        return len(self.feature_keys)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SceneWindowSchema":
        """从 checkpoint 输入契约恢复 scene window schema。"""
        if not isinstance(payload, Mapping):
            raise ValueError("scene window schema must be a mapping")
        feature_keys = payload.get("feature_keys")
        if not isinstance(feature_keys, (list, tuple)):
            raise ValueError("scene window feature_keys must be a list")
        return cls.from_feature_keys(
            context_key=str(payload["context_key"]),
            feature_keys=tuple(str(key) for key in feature_keys),
            scene_type_id=int(payload["scene_type_id"]),
        )


@dataclass(frozen=True)
class TrainingSchema:
    """raw 转换源与 compiled cache 共用的训练 schema。"""

    serialization_format: str
    sample_schema_version: int
    context_schema_version: int
    scene_context_mode: str
    scene_windows: tuple[SceneWindowSchema, ...]
    state_group_feature_keys: dict[str, tuple[str, ...]]
    candidate_skill_fields: tuple[str, ...]
    skill_history_fields: tuple[str, ...]

    def state_group_keys(self) -> tuple[str, ...]:
        return tuple(self.state_group_feature_keys.keys())

    def state_vector_dim(self) -> int:
        return sum(len(feature_keys) for feature_keys in self.state_group_feature_keys.values())

    def state_group_slices(self) -> dict[str, slice]:
        slices: dict[str, slice] = {}
        cursor = 0
        for group_key, feature_keys in self.state_group_feature_keys.items():
            next_cursor = cursor + len(feature_keys)
            slices[group_key] = slice(cursor, next_cursor)
            cursor = next_cursor
        return slices

    def scene_feature_dim(self) -> int:
        if not self.scene_windows:
            return 0
        return max(window.feature_dim for window in self.scene_windows)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TrainingSchema":
        """从 checkpoint 输入契约恢复训练 schema。"""
        if not isinstance(payload, Mapping):
            raise ValueError("training schema must be a mapping")
        scene_windows = payload.get("scene_windows")
        state_groups = payload.get("state_group_feature_keys")
        if not isinstance(scene_windows, (list, tuple)):
            raise ValueError("training schema scene_windows must be a list")
        if not isinstance(state_groups, Mapping):
            raise ValueError("training schema state_group_feature_keys must be a mapping")
        normalized_state_groups: dict[str, tuple[str, ...]] = {}
        for group_key, feature_keys in state_groups.items():
            if not isinstance(feature_keys, (list, tuple)):
                raise ValueError(
                    f"training schema feature group {group_key!r} must be a list"
                )
            normalized_state_groups[str(group_key)] = tuple(
                str(key) for key in feature_keys
            )
        return cls(
            serialization_format=str(payload["serialization_format"]),
            sample_schema_version=int(payload["sample_schema_version"]),
            context_schema_version=int(payload["context_schema_version"]),
            scene_context_mode=str(payload["scene_context_mode"]),
            scene_windows=tuple(
                SceneWindowSchema.from_dict(window) for window in scene_windows
            ),
            state_group_feature_keys=normalized_state_groups,
            candidate_skill_fields=_string_tuple_field(
                payload,
                "candidate_skill_fields",
            ),
            skill_history_fields=_string_tuple_field(
                payload,
                "skill_history_fields",
            ),
        )

    def assert_compatible_with(self, other: TrainingSchema) -> None:
        """显式校验两个训练 schema 是否兼容。"""
        if self.serialization_format != other.serialization_format:
            raise ValueError(
                "training serialization format mismatch: "
                f"{self.serialization_format!r} != {other.serialization_format!r}"
            )
        if self.sample_schema_version != other.sample_schema_version:
            raise ValueError(
                "training sample schema version mismatch: "
                f"{self.sample_schema_version} != {other.sample_schema_version}"
            )
        if self.context_schema_version != other.context_schema_version:
            raise ValueError(
                "training context schema version mismatch: "
                f"{self.context_schema_version} != {other.context_schema_version}"
            )
        if self.scene_context_mode != other.scene_context_mode:
            raise ValueError(
                "training scene context mode mismatch: "
                f"{self.scene_context_mode!r} != {other.scene_context_mode!r}"
            )
        if self.state_group_feature_keys != other.state_group_feature_keys:
            raise ValueError("training state feature keys mismatch")
        if set(self.candidate_skill_fields) != set(other.candidate_skill_fields):
            raise ValueError("training candidate skill field set mismatch")

        self_scene_windows = tuple(
            (window.context_key, window.feature_keys, window.scene_type_id)
            for window in self.scene_windows
        )
        other_scene_windows = tuple(
            (window.context_key, window.feature_keys, window.scene_type_id)
            for window in other.scene_windows
        )
        if self_scene_windows != other_scene_windows:
            raise ValueError("training scene window schema mismatch")


def _string_tuple_field(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"training schema {key} must be a list")
    return tuple(str(item) for item in value)


SCENE_WINDOW_ORDER = (
    (TARGETABLE_WINDOW_CONTEXT_KEY, SCENE_TYPE_TARGETABLE),
    (FORCED_MOVEMENT_CONTEXT_KEY, SCENE_TYPE_MOVEMENT),
    (RAID_BUFF_WINDOW_CONTEXT_KEY, SCENE_TYPE_RAID_BUFF),
    (TARGET_COUNT_WINDOW_CONTEXT_KEY, SCENE_TYPE_TARGET_COUNT),
)
