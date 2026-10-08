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
TRAINING_SAMPLE_SCHEMA_VERSION = 11


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
class StateFeatureGroup:
    """保存有序列名及编码职责，不依赖 JSON 对象键顺序。"""

    group_key: str
    feature_keys_field: str
    feature_keys: tuple[str, ...]
    encoding: str

    def __post_init__(self) -> None:
        if not isinstance(self.feature_keys, tuple):
            raise ValueError("state feature keys must be an immutable tuple")
        if any(not isinstance(value, str) or not value for value in (self.group_key, self.feature_keys_field)):
            raise ValueError("state group names must be nonempty strings")
        if self.encoding not in {"anchored_delta", "absolute_binary"}:
            raise ValueError(f"unknown state encoding: {self.encoding!r}")
        if any(not isinstance(key, str) or not key for key in self.feature_keys):
            raise ValueError("state feature keys must be nonempty strings")
        if len(set(self.feature_keys)) != len(self.feature_keys):
            raise ValueError("state feature keys must be unique within each group")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "StateFeatureGroup":
        if not isinstance(payload, Mapping):
            raise ValueError("state group must be a mapping")
        return cls(str(payload["group_key"]), str(payload["feature_keys_field"]),
                   _string_tuple_field(payload, "feature_keys"), str(payload["encoding"]))


@dataclass(frozen=True)
class StateFeatureLayout:
    """从保存的分组和固定动作顺序正向推导全部状态尺寸。"""

    groups: tuple[StateFeatureGroup, ...]
    snapshots: tuple[str, ...]
    action_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.snapshots != ("previous_action_after", "request_state"):
            raise ValueError("state snapshots must preserve previous_action_after/request_state order")
        # torch 安全反序列化会恢复 dataclass 字段而不调用其构造器，父级再次核验。
        if not isinstance(self.groups, tuple) or any(not isinstance(group, StateFeatureGroup) for group in self.groups):
            raise ValueError("state groups must be an immutable sequence of saved group records")
        for group in self.groups:
            group.__post_init__()
        if (not self.action_keys or any(not isinstance(key, str) or not key for key in self.action_keys)
                or len(set(self.action_keys)) != len(self.action_keys)):
            raise ValueError("state layout action keys must be nonempty and unique")
        names = tuple(group.group_key for group in self.groups)
        fields = tuple(group.feature_keys_field for group in self.groups)
        if len(set(names)) != len(names) or len(set(fields)) != len(fields):
            raise ValueError("state group and metadata field names must be unique")
        if not self.base_groups or len(self.binary_groups) != 1:
            raise ValueError("state layout requires base groups and one absolute binary group")
        if self.groups != (*self.base_groups, *self.binary_groups):
            raise ValueError("absolute binary state group must follow all base groups")
        expected = tuple(f"{snapshot}.{key}" for snapshot in self.snapshots for key in self.action_keys)
        if self.binary_groups[0].feature_keys != expected:
            raise ValueError("availability feature order must match snapshots and fixed action keys")
        if self.base_feature_keys.count("request_state.time_seconds") != 1:
            raise ValueError("state layout requires exactly one request_state.time_seconds")
        if len(set(self.base_feature_keys)) != len(self.base_feature_keys):
            raise ValueError("base state feature keys must be unique across groups")

    @property
    def base_groups(self) -> tuple[StateFeatureGroup, ...]:
        return tuple(group for group in self.groups if group.encoding == "anchored_delta")

    @property
    def binary_groups(self) -> tuple[StateFeatureGroup, ...]:
        return tuple(group for group in self.groups if group.encoding == "absolute_binary")

    @property
    def base_feature_keys(self) -> tuple[str, ...]:
        return tuple(key for group in self.base_groups for key in group.feature_keys)

    @property
    def base_state_dim(self) -> int:
        return sum(len(group.feature_keys) for group in self.base_groups)

    @property
    def availability_dim(self) -> int:
        return sum(len(group.feature_keys) for group in self.binary_groups)

    @property
    def state_dim(self) -> int:
        return self.base_state_dim + self.availability_dim

    @property
    def request_time_index(self) -> int:
        return self.base_feature_keys.index("request_state.time_seconds")

    @property
    def group_slices(self) -> dict[str, slice]:
        result = {}
        cursor = 0
        for group in self.groups:
            result[group.group_key] = slice(cursor, cursor + len(group.feature_keys))
            cursor += len(group.feature_keys)
        return result

    def assert_matches_data_spec(self, spec) -> None:
        if (spec.base_state_dim != self.base_state_dim or spec.state_dim != self.state_dim
                or tuple(spec.action_keys) != self.action_keys):
            raise ValueError("data spec state dimensions/action order differ from saved state layout")


@dataclass(frozen=True)
class TrainingSchema:
    """raw 转换源与 compiled cache 共用的训练 schema。"""

    serialization_format: str
    sample_schema_version: int
    context_schema_version: int
    scene_context_mode: str
    scene_windows: tuple[SceneWindowSchema, ...]
    state_groups: tuple[StateFeatureGroup, ...]
    state_snapshots: tuple[str, ...]
    skill_history_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if "time_seconds" in self.skill_history_fields:
            raise ValueError("removed skill time_seconds field; recompile raw source")

    def state_layout(self, action_keys: tuple[str, ...]) -> StateFeatureLayout:
        return StateFeatureLayout(self.state_groups, self.state_snapshots, tuple(action_keys))

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
        state_groups = payload.get("state_groups")
        if not isinstance(scene_windows, (list, tuple)):
            raise ValueError("training schema scene_windows must be a list")
        if not isinstance(state_groups, (list, tuple)):
            raise ValueError("training schema state_groups must be an ordered list")
        return cls(
            serialization_format=str(payload["serialization_format"]),
            sample_schema_version=int(payload["sample_schema_version"]),
            context_schema_version=int(payload["context_schema_version"]),
            scene_context_mode=str(payload["scene_context_mode"]),
            scene_windows=tuple(
                SceneWindowSchema.from_dict(window) for window in scene_windows
            ),
            state_groups=tuple(StateFeatureGroup.from_dict(group) for group in state_groups),
            state_snapshots=_string_tuple_field(payload, "state_snapshots"),
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
        if self.state_groups != other.state_groups or self.state_snapshots != other.state_snapshots:
            raise ValueError("training state feature keys mismatch")
        if self.skill_history_fields != other.skill_history_fields:
            raise ValueError("training skill history field layout mismatch")

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
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"training schema {key} must contain nonempty strings")
    return tuple(value)


SCENE_WINDOW_ORDER = (
    (TARGETABLE_WINDOW_CONTEXT_KEY, SCENE_TYPE_TARGETABLE),
    (FORCED_MOVEMENT_CONTEXT_KEY, SCENE_TYPE_MOVEMENT),
    (RAID_BUFF_WINDOW_CONTEXT_KEY, SCENE_TYPE_RAID_BUFF),
    (TARGET_COUNT_WINDOW_CONTEXT_KEY, SCENE_TYPE_TARGET_COUNT),
)
