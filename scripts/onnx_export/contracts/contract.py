"""固定容量 ONNX 输入契约与确定性样本构造。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from common.policy.data.input_contract import TOKEN_ENCODING_CONTRACT
from common.policy.data.context_fields import MODEL_INPUT_FIELDS, MODEL_INPUT_FIELDS_BY_NAME, MODEL_INPUT_NAMES, tensor_dimensions


TENSOR_INPUT_NAMES = MODEL_INPUT_NAMES
OUTPUT_NAMES = ("raw_logits",)
TOKEN_ORDER = TOKEN_ENCODING_CONTRACT["token_order"]
POSITION_ID_SEMANTICS = (
    "logical per-sample positions; right padding excluded; "
    "separate skill/state positions; RoPE applied to Q/K"
)


@dataclass(frozen=True)
class CapacityContract:
    """部署图固定容量；历史仍按动作条数计量，每条占技能和状态两个 token。"""

    scene_capacity: int
    history_capacity: int
    batch_size: int = 1
    padding_direction: str = "right"

    @property
    def total_token_count(self) -> int:
        """阶段 2 使用独立技能/状态 token；末尾追加一个当前状态。"""
        return self.scene_capacity + 2 * self.history_capacity + 1

    def validate(self) -> None:
        if self.batch_size != 1:
            raise ValueError("ONNX v1 contract requires batch_size=1")
        if self.scene_capacity < 1:
            raise ValueError("scene_capacity must be >= 1 to represent empty scene")
        if self.history_capacity < 1:
            raise ValueError("history_capacity must be >= 1 to represent empty history")
        if self.padding_direction != "right":
            raise ValueError("ONNX v1 contract only supports right padding")

    @classmethod
    def from_dict(cls, payload) -> "CapacityContract":
        if not isinstance(payload, dict):
            raise ValueError("capacity contract must be an object")
        expected = {
            "scene_capacity",
            "history_capacity",
            "batch_size",
            "padding_direction",
            "scene_padding_mask_value",
            "history_padding_mask_value",
            "position_ids",
            "over_capacity",
            "history_capacity_unit",
            "history_tokens_per_action",
            "token_order",
            "total_token_count",
            "effective_sequence_length",
        }
        if set(payload) != expected:
            raise ValueError("capacity contract fields are incomplete or unsupported")
        if payload["scene_padding_mask_value"] is not False:
            raise ValueError("scene padding mask value must be false")
        if payload["history_padding_mask_value"] is not False:
            raise ValueError("history padding mask value must be false")
        if payload["over_capacity"] != "reject":
            raise ValueError("deployment contract v1 only supports reject over-capacity")
        if payload["position_ids"] != POSITION_ID_SEMANTICS:
            raise ValueError("unsupported deployment position id semantics")
        if (
            payload["history_capacity_unit"] != TOKEN_ENCODING_CONTRACT["history_capacity_unit"]
            or payload["history_tokens_per_action"] != TOKEN_ENCODING_CONTRACT["history_tokens_per_action"]
        ):
            raise ValueError("deployment history capacity must represent two tokens per action")
        if payload["token_order"] != TOKEN_ORDER:
            raise ValueError("unsupported deployment token order")
        if payload["effective_sequence_length"] != "scene_valid + 2 * history_valid + 1":
            raise ValueError("unsupported deployment effective sequence length")
        contract = cls(
            scene_capacity=int(payload["scene_capacity"]),
            history_capacity=int(payload["history_capacity"]),
            batch_size=int(payload["batch_size"]),
            padding_direction=str(payload["padding_direction"]),
        )
        contract.validate()
        if payload["total_token_count"] != contract.total_token_count:
            raise ValueError("deployment total token count differs from independent token capacities")
        return contract

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "scene_padding_mask_value": False,
            "history_padding_mask_value": False,
            "position_ids": POSITION_ID_SEMANTICS,
            "over_capacity": "reject",
            "history_capacity_unit": TOKEN_ENCODING_CONTRACT["history_capacity_unit"],
            "history_tokens_per_action": TOKEN_ENCODING_CONTRACT["history_tokens_per_action"],
            "token_order": TOKEN_ORDER,
            "total_token_count": self.total_token_count,
            "effective_sequence_length": "scene_valid + 2 * history_valid + 1",
        }


def make_inputs(
    data_spec,
    contract: CapacityContract,
    *,
    vocab_size: int,
    scene_valid: int,
    history_valid: int,
    dtype: torch.dtype,
    seed: int,
    padding_fill: str = "random",
) -> tuple[torch.Tensor, ...]:
    """生成右侧 padding 的固定容量输入；超容量一律拒绝，不静默截断。"""
    if not 0 <= scene_valid <= contract.scene_capacity:
        raise ValueError("scene_valid is outside capacity")
    if not 0 <= history_valid <= contract.history_capacity:
        raise ValueError("history_valid is outside capacity")
    if padding_fill not in {"zero", "random"}:
        raise ValueError("padding_fill must be zero or random")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch_size = contract.batch_size
    dimensions = tensor_dimensions(data_spec, batch=batch_size, scene=contract.scene_capacity, history=contract.history_capacity)

    def shape(name: str) -> tuple[int, ...]:
        return MODEL_INPUT_FIELDS_BY_NAME[name].resolve_shape(dimensions)

    def floats(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=dtype)

    scene_vectors = floats(*shape("scene_vectors"))
    scene_types = torch.randint(
        0,
        data_spec.num_scene_types,
        shape("scene_types"),
        generator=generator,
        dtype=torch.long,
    )
    scene_mask = _length_mask(scene_valid, contract.scene_capacity)
    history_skill_ids = torch.randint(
        1,
        max(vocab_size, 2),
        shape("history_skill_ids"),
        generator=generator,
        dtype=torch.long,
    )
    history_skill_features = floats(
        *shape("history_skill_features"),
    )
    history_state_vectors = floats(
        *shape("history_state_vectors"),
    )
    history_state_null_mask = torch.rand(
        shape("history_state_null_mask"),
        generator=generator,
    ) < 0.1
    history_mask = _length_mask(history_valid, contract.history_capacity)
    history_state_null_mask[:, history_valid:] = True
    current_state_vectors = floats(*shape("current_state_vectors"))
    current_state_null_mask = torch.rand(
        shape("current_state_null_mask"), generator=generator,
    ) < 0.1
    history_state_reset_mask = (
        torch.rand(history_state_null_mask.shape, generator=generator) < 0.1
    ) & ~history_state_null_mask
    history_state_reset_mask[:, history_valid:] = False
    if history_valid:
        history_state_reset_mask[:, 0] = ~history_state_null_mask[:, 0]
    current_state_reset_mask = (
        torch.rand(current_state_null_mask.shape, generator=generator) < 0.1
    ) & ~current_state_null_mask
    if not history_valid:
        current_state_reset_mask = ~current_state_null_mask
    # 有效状态的技能可用性为绝对布尔值；脏 padding 仍用于检验 mask。
    for values in (history_state_vectors[:, :history_valid], current_state_vectors):
        availability = values[..., data_spec.base_state_dim:]
        availability.copy_(torch.randint(0, 2, availability.shape, generator=generator).to(dtype))
    named_inputs = {
        "scene_vectors": scene_vectors,
        "scene_types": scene_types,
        "scene_mask": scene_mask,
        "history_skill_ids": history_skill_ids,
        "history_skill_features": history_skill_features,
        "history_state_vectors": history_state_vectors,
        "history_state_null_mask": history_state_null_mask,
        "history_mask": history_mask,
        "current_state_vectors": current_state_vectors,
        "current_state_null_mask": current_state_null_mask,
        "history_state_reset_mask": history_state_reset_mask,
        "current_state_reset_mask": current_state_reset_mask,
    }
    result = tuple(named_inputs[name] for name in TENSOR_INPUT_NAMES)
    if padding_fill == "zero":
        return fill_padding_values(
            result,
            scene_valid=scene_valid,
            history_valid=history_valid,
            value=0.0,
        )
    return result


def slice_dynamic_inputs(
    inputs: tuple[torch.Tensor, ...],
    *,
    scene_valid: int,
    history_valid: int,
) -> tuple[torch.Tensor, ...]:
    """从右侧补位输入恢复动态长度 PyTorch 基线。"""
    lengths = {"scene": scene_valid, "history": history_valid}
    values = []
    for field, tensor in zip(MODEL_INPUT_FIELDS, inputs, strict=True):
        if field.sequence_axis is not None:
            indices = [slice(None)] * tensor.ndim
            indices[field.sequence_axis] = slice(None, lengths[field.sequence_group])
            tensor = tensor[tuple(indices)]
        values.append(tensor)
    return tuple(values)


def fill_padding_values(
    inputs: tuple[torch.Tensor, ...],
    *,
    scene_valid: int,
    history_valid: int,
    value: float,
) -> tuple[torch.Tensor, ...]:
    """只改无效 token 的载荷值，用于验证 mask 是唯一 padding 权威。"""
    lengths = {"scene": scene_valid, "history": history_valid}
    values = []
    for field, source in zip(MODEL_INPUT_FIELDS, inputs, strict=True):
        tensor = source.clone()
        if field.sequence_axis is not None and field.name != field.sequence_group + "_mask":
            indices = [slice(None)] * tensor.ndim
            indices[field.sequence_axis] = slice(lengths[field.sequence_group], None)
            tensor[tuple(indices)] = value if field.dtype in {"float", "float32"} else field.padding_value
        values.append(tensor)
    return tuple(values)


def _length_mask(valid: int, capacity: int) -> torch.Tensor:
    positions = torch.arange(capacity).unsqueeze(0)
    return positions < valid
