"""上下文字段的阶段、传输 dtype、序列轴和 padding 权威声明。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal


@dataclass(frozen=True, slots=True)
class TensorField:
    """序列轴针对所属阶段；FP32 编码完成后，部署边界才转换目标浮点精度。"""

    name: str
    stages: tuple[str, ...]
    dtype: Literal["float32", "float", "index", "bool"]
    sequence_axis: int | None
    padding_value: float | int | bool = 0
    cache_name: str | None = None
    sequence_group: Literal["scene", "history"] | None = None

    def torch_dtype(self, torch, *, float_dtype=None, index_dtype=None):
        """解析阶段规定的 dtype，不在公共声明模块提前加载 torch。"""
        if self.dtype == "float":
            return float_dtype if float_dtype is not None else torch.float32
        if self.dtype == "index":
            return index_dtype if index_dtype is not None else torch.int64
        return torch.float32 if self.dtype == "float32" else torch.bool


# bank 轴 0 是完整历史行；阶段子集决定 dataset 挂载和模型 gather 的范围。
_ENCODING_STAGES = ("cache", "dataset", "collator", "encoder")
HISTORY_BANK_FIELDS = (
    TensorField("skill_ids", _ENCODING_STAGES, "index", 0),
    TensorField("skill_features", _ENCODING_STAGES, "float", 0),
    TensorField("state_abs_values", _ENCODING_STAGES, "float32", 0),
    TensorField("state_delta_values", _ENCODING_STAGES, "float32", 0),
    TensorField("state_null_mask", _ENCODING_STAGES, "bool", 0, False),
    TensorField("state_delta_reset_mask", _ENCODING_STAGES, "bool", 0, False),
    TensorField("skill_potencies", ("cache", "dataset"), "float", 0),
    TensorField("cumulative_dot_potencies", ("cache", "dataset"), "float", 0),
)
HISTORY_BANK_METADATA_NAMES = ("action_keys",)
ENCODING_BANK_FIELDS = tuple(field for field in HISTORY_BANK_FIELDS if "encoder" in field.stages)
STATE_BANK_FIELDS = tuple(field for field in ENCODING_BANK_FIELDS if field.name.startswith("state_"))
HISTORY_RAW_FIELDS = tuple(
    replace(field, name="history_" + field.name, stages=("raw",), sequence_axis=1)
    for field in ENCODING_BANK_FIELDS
)
CURRENT_STATE_RAW_FIELDS = tuple(
    replace(field, name="current_" + field.name, stages=("cache", "raw"), sequence_axis=None)
    for field in STATE_BANK_FIELDS
)
RAW_SCENE_FIELDS = (
    # 磁盘 v22 原始记录保持既有布局；dataset 显式映射到 raw 输入名。
    TensorField("scene_abs_values", ("raw",), "float32", 1, cache_name="scene_vectors"),
    TensorField("scene_types", ("raw",), "index", 1),
)
_RAW_FIELDS_BY_NAME = {field.name: field for field in (*HISTORY_RAW_FIELDS, *CURRENT_STATE_RAW_FIELDS, *RAW_SCENE_FIELDS)}

# 顺序就是部署图的公开输入顺序；新增模型输入只在这里声明传输规则。
MODEL_INPUT_FIELDS = (
    TensorField("scene_vectors", ("model", "onnx"), "float32", 1),
    replace(_RAW_FIELDS_BY_NAME["scene_types"], stages=("model", "onnx")),
    TensorField("scene_mask", ("raw", "model", "onnx"), "bool", 1, False),
    replace(_RAW_FIELDS_BY_NAME["history_skill_ids"], stages=("model", "onnx")),
    replace(_RAW_FIELDS_BY_NAME["history_skill_features"], stages=("model", "onnx")),
    TensorField("history_state_vectors", ("model", "onnx"), "float32", 1),
    replace(_RAW_FIELDS_BY_NAME["history_state_null_mask"], stages=("model", "onnx"), padding_value=True),
    TensorField("history_mask", ("raw", "model", "onnx"), "bool", 1, False),
    TensorField("current_state_vectors", ("model", "onnx"), "float32", None),
    replace(_RAW_FIELDS_BY_NAME["current_state_null_mask"], stages=("model", "onnx")),
    TensorField("history_state_reset_mask", ("model", "onnx"), "bool", 1, False),
    TensorField("current_state_reset_mask", ("model", "onnx"), "bool", None, False),
)
MODEL_INPUT_FIELDS = tuple(
    replace(field, sequence_group=("scene" if field.name.startswith("scene_") else "history") if field.sequence_axis is not None else None)
    for field in MODEL_INPUT_FIELDS
)
MODEL_INPUT_FIELDS_BY_NAME = MappingProxyType({field.name: field for field in MODEL_INPUT_FIELDS})
MODEL_INPUT_NAMES = tuple(field.name for field in MODEL_INPUT_FIELDS)
RAW_ONLY_FIELD_NAMES = frozenset(_RAW_FIELDS_BY_NAME).difference(MODEL_INPUT_NAMES)
