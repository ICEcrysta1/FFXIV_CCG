"""固定容量 ONNX 输入契约与确定性样本构造。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from common.policy.data.input_contract import TOKEN_ENCODING_CONTRACT


TENSOR_INPUT_NAMES = (
    "scene_vectors",
    "scene_types",
    "scene_mask",
    "history_skill_ids",
    "history_skill_features",
    "history_state_vectors",
    "history_state_null_mask",
    "history_mask",
    "current_state_vectors",
    "current_state_null_mask",
    "history_state_reset_mask",
    "current_state_reset_mask",
)
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

    def floats(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=dtype)

    scene_vectors = floats(batch_size, contract.scene_capacity, data_spec.scene_dim)
    scene_types = torch.randint(
        0,
        data_spec.num_scene_types,
        (batch_size, contract.scene_capacity),
        generator=generator,
        dtype=torch.long,
    )
    scene_mask = _length_mask(scene_valid, contract.scene_capacity)
    history_skill_ids = torch.randint(
        1,
        max(vocab_size, 2),
        (batch_size, contract.history_capacity),
        generator=generator,
        dtype=torch.long,
    )
    history_skill_features = floats(
        batch_size,
        contract.history_capacity,
        data_spec.skill_feature_dim,
    )
    history_state_vectors = floats(
        batch_size,
        contract.history_capacity,
        data_spec.state_dim,
    )
    history_state_null_mask = torch.rand(
        batch_size,
        contract.history_capacity,
        data_spec.state_dim,
        generator=generator,
    ) < 0.1
    history_mask = _length_mask(history_valid, contract.history_capacity)
    history_state_null_mask[:, history_valid:] = True
    current_state_vectors = floats(batch_size, data_spec.state_dim)
    current_state_null_mask = torch.rand(
        batch_size, data_spec.state_dim, generator=generator,
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
    result = (
        scene_vectors,
        scene_types,
        scene_mask,
        history_skill_ids,
        history_skill_features,
        history_state_vectors,
        history_state_null_mask,
        history_mask,
        current_state_vectors,
        current_state_null_mask,
        history_state_reset_mask,
        current_state_reset_mask,
    )
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
    values = list(inputs)
    for index in (0, 1, 2):
        values[index] = values[index][:, :scene_valid]
    for index in (3, 4, 5, 6, 7, 10):
        values[index] = values[index][:, :history_valid]
    return tuple(values)


def fill_padding_values(
    inputs: tuple[torch.Tensor, ...],
    *,
    scene_valid: int,
    history_valid: int,
    value: float,
) -> tuple[torch.Tensor, ...]:
    """只改无效 token 的载荷值，用于验证 mask 是唯一 padding 权威。"""
    values = [tensor.clone() for tensor in inputs]
    values[0][:, scene_valid:] = value
    values[1][:, scene_valid:] = 0
    values[3][:, history_valid:] = 0
    for index in (4, 5):
        values[index][:, history_valid:] = value
    values[6][:, history_valid:] = True
    values[10][:, history_valid:] = False
    return tuple(values)


def _length_mask(valid: int, capacity: int) -> torch.Tensor:
    positions = torch.arange(capacity).unsqueeze(0)
    return positions < valid
