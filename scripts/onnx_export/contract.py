"""固定容量 ONNX 输入契约与确定性样本构造。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


TENSOR_INPUT_NAMES = (
    "scene_vectors",
    "scene_types",
    "scene_mask",
    "history_skill_ids",
    "history_skill_features",
    "history_state_vectors",
    "history_state_null_mask",
    "history_mask",
    "candidate_skill_ids",
    "candidate_skill_features",
    "candidate_state_vectors",
    "candidate_state_null_mask",
)
OUTPUT_NAMES = ("raw_logits",)


@dataclass(frozen=True)
class CapacityContract:
    """首版部署图的 batch、scene、history 和候选固定容量。"""

    scene_capacity: int
    history_capacity: int
    candidate_count: int
    batch_size: int = 1
    padding_direction: str = "right"

    @property
    def candidate_token_count(self) -> int:
        """模型内部的候选 token 数；双向候选只保留一份。"""
        return self.candidate_count

    @property
    def total_token_count(self) -> int:
        """由 scene、history、候选和 CLS 容量自动换算物理 token 数。"""
        return self.scene_capacity + self.history_capacity + self.candidate_token_count + 1

    def validate(self) -> None:
        if self.batch_size != 1:
            raise ValueError("ONNX v1 contract requires batch_size=1")
        if self.scene_capacity < 1:
            raise ValueError("scene_capacity must be >= 1 to represent empty scene")
        if self.history_capacity < 1:
            raise ValueError("history_capacity must be >= 1 to represent empty history")
        if self.candidate_count < 1:
            raise ValueError("candidate_count must be >= 1")
        if self.padding_direction != "right":
            raise ValueError("ONNX v1 contract only supports right padding")

    @classmethod
    def from_dict(cls, payload) -> "CapacityContract":
        if not isinstance(payload, dict):
            raise ValueError("capacity contract must be an object")
        expected = {
            "scene_capacity",
            "history_capacity",
            "candidate_count",
            "batch_size",
            "padding_direction",
            "scene_padding_mask_value",
            "history_padding_mask_value",
            "position_ids",
            "over_capacity",
        }
        if set(payload) != expected:
            raise ValueError("capacity contract fields are incomplete or unsupported")
        if payload["scene_padding_mask_value"] is not False:
            raise ValueError("scene padding mask value must be false")
        if payload["history_padding_mask_value"] is not False:
            raise ValueError("history padding mask value must be false")
        if payload["over_capacity"] != "reject":
            raise ValueError("deployment contract v1 only supports reject over-capacity")
        if payload["position_ids"] != (
            "logical per-sample positions; right padding excluded; RoPE applied to Q/K"
        ):
            raise ValueError("unsupported deployment position id semantics")
        return cls(
            scene_capacity=int(payload["scene_capacity"]),
            history_capacity=int(payload["history_capacity"]),
            candidate_count=int(payload["candidate_count"]),
            batch_size=int(payload["batch_size"]),
            padding_direction=str(payload["padding_direction"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "scene_padding_mask_value": False,
            "history_padding_mask_value": False,
            "position_ids": (
                "logical per-sample positions; right padding excluded; RoPE applied to Q/K"
            ),
            "over_capacity": "reject",
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
    candidate_skill_ids = (
        torch.arange(data_spec.num_candidates, dtype=torch.long)
        .remainder(max(vocab_size - 1, 1))
        .add(1)
        .unsqueeze(0)
    )
    candidate_skill_features = floats(
        batch_size,
        data_spec.num_candidates,
        data_spec.skill_feature_dim,
    )
    candidate_state_vectors = floats(
        batch_size,
        data_spec.num_candidates,
        data_spec.state_dim,
    )
    candidate_state_null_mask = torch.rand(
        batch_size,
        data_spec.num_candidates,
        data_spec.state_dim,
        generator=generator,
    ) < 0.1
    result = (
        scene_vectors,
        scene_types,
        scene_mask,
        history_skill_ids,
        history_skill_features,
        history_state_vectors,
        history_state_null_mask,
        history_mask,
        candidate_skill_ids,
        candidate_skill_features,
        candidate_state_vectors,
        candidate_state_null_mask,
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
    for index in (3, 4, 5, 6, 7):
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
    values[6][:, history_valid:] = False
    return tuple(values)


def _length_mask(valid: int, capacity: int) -> torch.Tensor:
    positions = torch.arange(capacity).unsqueeze(0)
    return positions < valid
