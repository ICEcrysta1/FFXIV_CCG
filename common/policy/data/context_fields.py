"""上下文字段的阶段、dtype、逻辑维度和 padding 权威声明。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal


@dataclass(frozen=True, slots=True)
class TensorField:
    """逻辑轴同时决定形状和序列轴，禁止两个位置独立维护。"""

    name: str
    stages: tuple[str, ...]
    dtype: Literal["float32", "float", "index", "bool"]
    axes: tuple[str, ...]
    padding_value: float | int | bool = 0
    cache_name: str | None = None
    semantic: Literal["skill", "state", "scene", "execution", "mask"] = "state"
    cache_axes: tuple[str, ...] | None = None

    @property
    def sequence_axis(self) -> int | None:
        return next((index for index, axis in enumerate(self.axes) if axis in {"scene", "history", "bank"}), None)

    @property
    def sequence_group(self) -> Literal["scene", "history"] | None:
        if "scene" in self.axes:
            return "scene"
        return "history" if "history" in self.axes or "bank" in self.axes else None

    def torch_dtype(self, torch, *, float_dtype=None, index_dtype=None):
        """仅在调用端传入 torch，字段声明自身不初始化运行时。"""
        if self.dtype == "float":
            return float_dtype if float_dtype is not None else torch.float32
        if self.dtype == "index":
            return index_dtype if index_dtype is not None else torch.int64
        return torch.float32 if self.dtype == "float32" else torch.bool

    def resolve_shape(self, dimensions: Mapping[str, int | str], *, stage: str | None = None) -> tuple[int | str, ...]:
        axes = self.cache_axes if stage == "cache" and self.cache_axes is not None else self.axes
        return tuple(dimensions[axis] for axis in axes)

    def validate_tensor(self, value, dimensions, *, torch, float_dtype=None, index_dtype=None, stage=None) -> None:
        """只检查结构，不读取设备标量；有限值检查留在 CPU 原料边界。"""
        shape = self.resolve_shape(dimensions, stage=stage)
        if not isinstance(value, torch.Tensor) or value.ndim != len(shape):
            raise ValueError(f"{self.name} must have shape {shape}")
        if any(isinstance(size, int) and value.shape[index] != size for index, size in enumerate(shape)):
            raise ValueError(f"{self.name} must have shape {shape}, got {tuple(value.shape)}")
        if self.dtype == "index" and index_dtype is None:
            valid_dtype = value.dtype in (torch.int32, torch.int64)
        else:
            valid_dtype = value.dtype == self.torch_dtype(torch, float_dtype=float_dtype, index_dtype=index_dtype)
        if not valid_dtype:
            raise ValueError(f"{self.name} has invalid dtype {value.dtype}")


def tensor_dimensions(data_spec, *, batch=None, scene=None, history=None, bank=None, skill_feature_dim=None) -> dict[str, int | str]:
    """模型规格由父级 layout 校验；容量由各调用方自己的生命周期提供。"""
    return {
        "batch": "batch" if batch is None else batch,
        "scene": "scene" if scene is None else scene,
        "history": "history" if history is None else history,
        "bank": "bank" if bank is None else bank,
        "base_state": data_spec.base_state_dim,
        "availability": data_spec.state_dim - data_spec.base_state_dim,
        "state": data_spec.state_dim,
        "scene_feature": data_spec.scene_dim,
        "skill_feature": data_spec.skill_feature_dim if skill_feature_dim is None else skill_feature_dim,
    }


_ENCODING_STAGES = ("cache", "dataset", "collator", "encoder")
HISTORY_BANK_FIELDS = (
    TensorField("skill_ids", _ENCODING_STAGES, "index", ("bank",), semantic="skill"),
    TensorField("skill_features", _ENCODING_STAGES, "float", ("bank", "skill_feature"), semantic="skill"),
    TensorField("state_abs_values", _ENCODING_STAGES, "float32", ("bank", "base_state")),
    TensorField("state_delta_values", _ENCODING_STAGES, "float32", ("bank", "base_state")),
    TensorField("state_null_mask", _ENCODING_STAGES, "bool", ("bank", "base_state"), False),
    TensorField("state_delta_reset_mask", _ENCODING_STAGES, "bool", ("bank", "base_state"), False),
    TensorField("state_skill_availability", _ENCODING_STAGES, "bool", ("bank", "availability"), False),
    TensorField("skill_potencies", ("cache", "dataset"), "float", ("bank",), semantic="execution"),
    TensorField("cumulative_dot_potencies", ("cache", "dataset"), "float", ("bank",), semantic="execution"),
)
HISTORY_BANK_METADATA_NAMES = ("action_keys",)
ENCODING_BANK_FIELDS = tuple(field for field in HISTORY_BANK_FIELDS if "encoder" in field.stages)
STATE_BANK_FIELDS = tuple(field for field in ENCODING_BANK_FIELDS if field.semantic == "state")
HISTORY_RAW_FIELDS = tuple(
    replace(field, name="history_" + field.name, stages=("raw",), axes=("batch", "history", *field.axes[1:]))
    for field in ENCODING_BANK_FIELDS
)
CURRENT_STATE_RAW_FIELDS = tuple(
    replace(field, name="current_" + field.name, stages=("cache", "raw"), axes=("batch", *field.axes[1:]), cache_axes=field.axes[1:])
    for field in STATE_BANK_FIELDS
)
RAW_SCENE_FIELDS = (
    TensorField("scene_abs_values", ("raw",), "float32", ("batch", "scene", "scene_feature"), cache_name="scene_vectors", semantic="scene"),
    TensorField("scene_types", ("raw",), "index", ("batch", "scene"), semantic="scene"),
)
_RAW_FIELDS_BY_NAME = {field.name: field for field in (*HISTORY_RAW_FIELDS, *CURRENT_STATE_RAW_FIELDS, *RAW_SCENE_FIELDS)}

# 顺序就是部署图的公开输入顺序；技能表并入状态值，不增加图输入。
MODEL_INPUT_FIELDS = (
    TensorField("scene_vectors", ("model", "onnx"), "float32", ("batch", "scene", "scene_feature"), semantic="scene"),
    replace(_RAW_FIELDS_BY_NAME["scene_types"], stages=("model", "onnx")),
    TensorField("scene_mask", ("raw", "model", "onnx"), "bool", ("batch", "scene"), False, semantic="mask"),
    replace(_RAW_FIELDS_BY_NAME["history_skill_ids"], stages=("model", "onnx")),
    replace(_RAW_FIELDS_BY_NAME["history_skill_features"], stages=("model", "onnx")),
    TensorField("history_state_vectors", ("model", "onnx"), "float32", ("batch", "history", "state")),
    replace(_RAW_FIELDS_BY_NAME["history_state_null_mask"], stages=("model", "onnx"), padding_value=True),
    TensorField("history_mask", ("raw", "model", "onnx"), "bool", ("batch", "history"), False, semantic="mask"),
    TensorField("current_state_vectors", ("model", "onnx"), "float32", ("batch", "state")),
    replace(_RAW_FIELDS_BY_NAME["current_state_null_mask"], stages=("model", "onnx")),
    TensorField("history_state_reset_mask", ("model", "onnx"), "bool", ("batch", "history", "base_state"), False),
    TensorField("current_state_reset_mask", ("model", "onnx"), "bool", ("batch", "base_state"), False),
)
MODEL_INPUT_FIELDS_BY_NAME = MappingProxyType({field.name: field for field in MODEL_INPUT_FIELDS})
MODEL_INPUT_NAMES = tuple(field.name for field in MODEL_INPUT_FIELDS)
RAW_ONLY_FIELD_NAMES = frozenset(_RAW_FIELDS_BY_NAME).difference(MODEL_INPUT_NAMES)
