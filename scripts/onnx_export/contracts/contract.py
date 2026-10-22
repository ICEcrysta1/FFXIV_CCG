"""固定容量 ONNX 输入契约与确定性样本构造。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from common.policy.data.contracts.context_fields import (
    MODEL_INPUT_FIELDS,
    MODEL_INPUT_FIELDS_BY_NAME,
    MODEL_INPUT_NAMES,
    tensor_dimensions,
)
from common.policy.data.contracts.input_contract import TOKEN_ENCODING_CONTRACT
from common.policy.data.schema import StateFeatureLayout

TENSOR_INPUT_NAMES = MODEL_INPUT_NAMES
OUTPUT_NAMES = ("raw_logits",)
TOKEN_ORDER = TOKEN_ENCODING_CONTRACT["token_order"]
POSITION_ID_SEMANTICS = (
    "logical per-sample positions; right padding excluded; "
    "separate skill/state positions; RoPE applied to Q/K"
)


@dataclass(frozen=True)
class CapacityContract:
    """部署图共用 token 预算；原始输入轴上界不代表独立可用容量。"""

    context_capacity: int
    batch_size: int = 1
    padding_direction: str = "right"

    @property
    def scene_capacity(self) -> int:
        """场景输入的运输上界，另一个槽始终留给当前状态。"""
        return self.context_capacity - 1

    @property
    def history_capacity(self) -> int:
        """动作输入的运输上界，每条动作占状态和技能两个 token。"""
        return (self.context_capacity - 1) // 2

    @property
    def dead_capacity(self) -> int:
        """死亡事件各占一个 token，与场景和动作共用总预算。"""
        return self.context_capacity - 1

    @property
    def total_token_count(self) -> int:
        """场景、历史与当前状态共用的物理 token 上限。"""
        return self.context_capacity

    def validate(self) -> None:
        if self.batch_size != 1:
            raise ValueError("ONNX v1 contract requires batch_size=1")
        if type(self.context_capacity) is not int or self.context_capacity < 1:
            raise ValueError("context_capacity must be a positive integer")
        if self.padding_direction != "right":
            raise ValueError("ONNX v1 contract only supports right padding")

    @classmethod
    def from_dict(cls, payload) -> CapacityContract:
        if not isinstance(payload, dict):
            raise TypeError("capacity contract must be an object")
        expected = {
            "context_capacity",
            "batch_size",
            "padding_direction",
            "scene_padding_mask_value",
            "history_padding_mask_value",
            "dead_padding_mask_value",
            "position_ids",
            "over_capacity",
            "capacity_unit",
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
        if payload["dead_padding_mask_value"] is not False:
            raise ValueError("dead padding mask value must be false")
        if payload["over_capacity"] != "reject":
            raise ValueError("deployment contract v1 only supports reject over-capacity")
        if payload["position_ids"] != POSITION_ID_SEMANTICS:
            raise ValueError("unsupported deployment position id semantics")
        if payload["capacity_unit"] != "tokens":
            raise ValueError("deployment context capacity must represent tokens")
        if payload["history_tokens_per_action"] != TOKEN_ENCODING_CONTRACT["history_tokens_per_action"]:
            raise ValueError("deployment history capacity must represent two tokens per action")
        if payload["token_order"] != TOKEN_ORDER:
            raise ValueError("unsupported deployment token order")
        if payload["effective_sequence_length"] != "scene_valid + 2 * history_valid + dead_valid + 1":
            raise ValueError("unsupported deployment effective sequence length")
        contract = cls(
            context_capacity=payload["context_capacity"],
            batch_size=int(payload["batch_size"]),
            padding_direction=str(payload["padding_direction"]),
        )
        contract.validate()
        if payload["total_token_count"] != contract.total_token_count:
            raise ValueError("deployment total token count differs from shared context capacity")
        return contract

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "scene_padding_mask_value": False,
            "history_padding_mask_value": False,
            "dead_padding_mask_value": False,
            "position_ids": POSITION_ID_SEMANTICS,
            "over_capacity": "reject",
            "capacity_unit": "tokens",
            "history_tokens_per_action": TOKEN_ENCODING_CONTRACT["history_tokens_per_action"],
            "token_order": TOKEN_ORDER,
            "total_token_count": self.total_token_count,
            "effective_sequence_length": "scene_valid + 2 * history_valid + dead_valid + 1",
        }


def make_inputs(
    data_spec,
    contract: CapacityContract,
    *,
    state_layout: StateFeatureLayout,
    vocab_size: int,
    scene_valid: int,
    history_valid: int,
    dtype: torch.dtype,
    seed: int,
    dead_valid: int = 0,
    padding_fill: str = "random",
) -> tuple[torch.Tensor, ...]:
    """生成右侧 padding 的固定容量输入；超容量一律拒绝，不静默截断。"""
    if not 0 <= scene_valid <= contract.scene_capacity:
        raise ValueError("scene_valid is outside capacity")
    if not 0 <= history_valid <= contract.history_capacity:
        raise ValueError("history_valid is outside capacity")
    if not 0 <= dead_valid <= contract.dead_capacity:
        raise ValueError("dead_valid is outside capacity")
    if scene_valid + 2 * history_valid + dead_valid + 1 > contract.context_capacity:
        raise ValueError("valid scene and history exceed shared context_capacity")
    if padding_fill not in {"zero", "random"}:
        raise ValueError("padding_fill must be zero or random")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch_size = contract.batch_size
    dimensions = tensor_dimensions(data_spec, batch=batch_size, scene=contract.scene_capacity,
                                   history=contract.history_capacity, dead=contract.dead_capacity)

    def shape(name: str) -> tuple[int, ...]:
        return MODEL_INPUT_FIELDS_BY_NAME[name].resolve_shape(dimensions)

    def floats(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32)

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
    state_layout.assert_matches_data_spec(data_spec)
    time_fields = {f"{snapshot}.time_seconds" for snapshot in state_layout.snapshots}
    time_mask = torch.tensor([key in time_fields for key in state_layout.base_feature_keys])
    history_state_reset_mask &= time_mask
    current_state_reset_mask &= time_mask
    # 有效技能状态是归一化绝对值，可用性单独保持 0/1；脏 padding 继续检验 mask。
    for values in (history_state_vectors[:, :history_valid], current_state_vectors):
        skills = values[..., data_spec.base_state_dim:]
        skills.copy_(torch.rand(skills.shape, generator=generator))
        indices = list(state_layout.availability_indices)
        values[..., indices] = torch.randint(0, 2, values[..., indices].shape, generator=generator).float()
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
        "history_dead_positions": torch.div(
            torch.arange(contract.dead_capacity) * (history_valid + 1),
            max(dead_valid, 1), rounding_mode="floor",
        ).clamp(max=history_valid).unsqueeze(0),
        "history_dead_mask": _length_mask(dead_valid, contract.dead_capacity),
    }
    result = tuple(named_inputs[name] for name in TENSOR_INPUT_NAMES)
    if padding_fill == "zero":
        return fill_padding_values(
            result,
            scene_valid=scene_valid,
            history_valid=history_valid,
            dead_valid=dead_valid,
            value=0.0,
        )
    return result


def slice_dynamic_inputs(
    inputs: tuple[torch.Tensor, ...],
    *,
    scene_valid: int,
    history_valid: int,
    dead_valid: int = 0,
) -> tuple[torch.Tensor, ...]:
    """从右侧补位输入恢复动态长度 PyTorch 基线。"""
    lengths = {"scene": scene_valid, "history": history_valid, "dead": dead_valid}
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
    dead_valid: int = 0,
) -> tuple[torch.Tensor, ...]:
    """只改无效 token 的载荷值，用于验证 mask 是唯一 padding 权威。"""
    lengths = {"scene": scene_valid, "history": history_valid, "dead": dead_valid}
    values = []
    for field, source in zip(MODEL_INPUT_FIELDS, inputs, strict=True):
        tensor = source.clone()
        if field.sequence_axis is not None and field.semantic != "mask":
            indices = [slice(None)] * tensor.ndim
            indices[field.sequence_axis] = slice(lengths[field.sequence_group], None)
            tensor[tuple(indices)] = value if field.dtype in {"float", "float32"} else field.padding_value
        values.append(tensor)
    return tuple(values)


def _length_mask(valid: int, capacity: int) -> torch.Tensor:
    positions = torch.arange(capacity).unsqueeze(0)
    return positions < valid
