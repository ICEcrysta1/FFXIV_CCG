"""模型输入编码器：把 batch 字段转换成 Transformer token。"""

from __future__ import annotations

from dataclasses import replace
import torch
import torch.nn as nn

from ..config import ModelConfig
from ..data.input_contract import TOKEN_ENCODING_CONTRACT
from ..data.spec import DataSpec
from ..data.context_fields import MODEL_INPUT_FIELDS, tensor_dimensions


ROLE_SCENE = TOKEN_ENCODING_CONTRACT["role_ids"]["scene"]
ROLE_STATE = TOKEN_ENCODING_CONTRACT["role_ids"]["state"]
ROLE_SKILL = TOKEN_ENCODING_CONTRACT["role_ids"]["skill"]


class _StateResetProjection(nn.Linear):
    """绝对值重置标识从零开始，不消耗公共参数的初始化随机数。"""

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.weight)


class CausalInputEncoder(nn.Module):
    """将场景、技能和状态独立编码到同一个因果上下文。"""

    def __init__(self, data_spec: DataSpec, config: ModelConfig, vocab_size: int):
        super().__init__()
        if data_spec.scene_dim <= 0:
            raise ValueError("CausalInputEncoder requires a non-empty scene vector")

        d_model = config.d_model
        self.data_spec = data_spec
        self.config = config
        self.skill_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.skill_feat_proj = nn.Linear(data_spec.skill_feature_dim, d_model)
        self.state_proj = nn.Linear(data_spec.state_dim, d_model)
        self.state_null_proj = nn.Linear(data_spec.base_state_dim, d_model, bias=False)
        self.scene_proj = nn.ModuleList(
            nn.Linear(data_spec.scene_dim, d_model)
            for _ in range(data_spec.num_scene_types)
        )
        self.role_embed = nn.Embedding(3, d_model)
        self.state_reset_proj = _StateResetProjection(data_spec.base_state_dim, d_model, bias=False)

    @property
    def max_token_count(self) -> int:
        """返回由各上下文块容量自动换算出的最大物理 token 数。"""
        return (
            self.config.scene_capacity
            + 2 * self.config.history_capacity
            + 1
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self._validate_batch(batch)
        scene_vectors = batch["scene_vectors"].to(dtype=self.scene_proj[0].weight.dtype)
        batch_size, scene_length, _ = scene_vectors.shape
        history_length = batch["history_skill_ids"].shape[1]
        # 物理 token 布局仍按 batch 的最大 scene/history 宽度补齐；
        # RoPE 使用的逻辑位置由有效长度单独生成，不再把 padding 当成时间步。
        history_token_length = 2 * history_length
        total_length = scene_length + history_token_length + 1
        if total_length > self.max_token_count:
            raise ValueError(
                "physical token sequence length exceeds computed model capacity: "
                f"{total_length} > {self.max_token_count}"
            )
        device = scene_vectors.device
        d_model = self.config.d_model

        # 所有 scene 类型都先投影，再由张量索引选择对应类型，避免根据输入值
        # 进入 Python 分支。这样 scene_types 仍然是图输入，而不是导出时的常量。
        scene_type_embeds = torch.stack(
            [projection(scene_vectors) for projection in self.scene_proj],
            dim=2,
        )
        scene_type_indices = batch["scene_types"].unsqueeze(-1).unsqueeze(-1)
        scene_embeds = torch.gather(
            scene_type_embeds,
            dim=2,
            index=scene_type_indices.expand(-1, -1, 1, d_model),
        ).squeeze(2)
        history = self.embed_history(batch)
        # 请求时冻结的状态位于该次技能之前；技能只能读取此前已知的状态。
        history_tokens = torch.stack((history["state"], history["skill"]), dim=2).reshape(
            batch_size, history_token_length, d_model,
        )
        # 历史与最新状态使用同一个投影，不增加特殊当前状态参数。
        current_state = self._embed_state(
            batch["current_state_vectors"],
            batch.get("current_state_null_mask"),
            batch["current_state_reset_mask"],
        ).unsqueeze(1)

        content_tokens = torch.cat(
            (scene_embeds, history_tokens, current_state),
            dim=1,
        )

        position_ids = build_position_ids(
            batch_size=batch_size,
            scene_length=scene_length,
            history_length=history_length,
            device=device,
            scene_mask=batch["scene_mask"],
            history_mask=batch["history_mask"],
        )
        role_ids = build_role_ids(
            batch_size=batch_size,
            scene_length=scene_length,
            history_length=history_length,
            device=device,
        )
        tokens = content_tokens + self.role_embed(role_ids)
        # 全部角色合成后统一尺度，不去均值，也不引入可学习仿射参数。
        tokens = torch.nn.functional.rms_norm(tokens, (d_model,), eps=1e-5)

        current_state_position = scene_length + history_token_length
        current_state_positions = torch.full(
            (batch_size,), current_state_position, dtype=torch.long, device=device,
        )
        prefix_valid = torch.cat(
            (batch["scene_mask"], batch["history_mask"].repeat_interleave(2, dim=1)),
            dim=1,
        )
        current_state_valid = torch.ones(
            (batch_size, 1),
            dtype=torch.bool,
            device=device,
        )
        valid = torch.cat((prefix_valid, current_state_valid), dim=1)
        history_state_positions = (
            scene_length + 2 * torch.arange(history_length, device=device, dtype=torch.long)
        ).unsqueeze(0).expand(batch_size, -1)
        return {
            "tokens": tokens,
            "padding_mask": ~valid,
            "prefix_valid": prefix_valid,
            "valid": valid,
            "scene_length": scene_length,
            "history_length": history_length,
            "history_token_length": history_token_length,
            "prefix_length": scene_length + history_token_length,
            "position_ids": position_ids,
            "role_ids": role_ids,
            "history_skill_positions": history_state_positions + 1,
            "history_state_positions": history_state_positions,
            "current_state_position": current_state_position,
            "current_state_positions": current_state_positions,
        }

    def embed_history(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """返回历史技能、状态的原始 content；role 和 RMSNorm 由 forward 统一施加。"""
        self._validate_batch(batch)
        return {
            "skill": self.skill_embed(batch["history_skill_ids"])
            + self.skill_feat_proj(batch["history_skill_features"].to(dtype=self.skill_feat_proj.weight.dtype)),
            "state": self._embed_state(
                batch["history_state_vectors"],
                batch.get("history_state_null_mask"),
                batch["history_state_reset_mask"],
            ),
        }

    def _embed_state(self, values, null_mask, reset_mask) -> torch.Tensor:
        """投影已编码数值、缺失及 ABS 重置标识；精度转换发生在编码完成后。"""
        values = values.to(dtype=self.state_proj.weight.dtype)
        if null_mask is None:
            null_mask = torch.zeros((*values.shape[:-1], self.data_spec.base_state_dim), dtype=torch.bool, device=values.device)
        return (
            self.state_proj(values)
            + self.state_null_proj(null_mask.to(dtype=values.dtype))
            + self.state_reset_proj(reset_mask.to(dtype=values.dtype))
        )

    def _validate_batch(self, batch: dict[str, torch.Tensor]) -> None:
        expected = self.data_spec
        scene_length = batch["scene_vectors"].shape[1]
        history_length = batch["history_skill_ids"].shape[1]
        if scene_length > self.config.scene_capacity:
            raise ValueError(
                "scene context length exceeds model.scene_capacity: "
                f"{scene_length} > {self.config.scene_capacity}"
            )
        if history_length > self.config.history_capacity:
            raise ValueError(
                "skill history context length exceeds model.history_capacity: "
                f"{history_length} > {self.config.history_capacity}"
            )
        dimensions = tensor_dimensions(expected, batch=batch["scene_vectors"].shape[0], scene=scene_length, history=history_length)
        for field in MODEL_INPUT_FIELDS:
            if field.name.endswith("null_mask") and field.name not in batch:
                continue
            # 神经入口允许部署已转换的浮点精度，结构与 bool/index 仍按统一声明检查。
            value = batch[field.name]
            if field.dtype in {"float", "float32"}:
                if not value.is_floating_point():
                    raise ValueError(f"{field.name} must be floating point")
                replace(field, dtype="float").validate_tensor(value, dimensions, torch=torch, float_dtype=value.dtype)
            else:
                field.validate_tensor(value, dimensions, torch=torch)
        if "action_legal_mask" in batch and batch["action_legal_mask"].shape != (
            batch["scene_vectors"].shape[0], expected.num_actions,
        ):
            raise ValueError("action legal mask does not match the data spec")


def build_position_ids(
    *,
    batch_size: int,
    scene_length: int,
    history_length: int,
    device,
    scene_mask: torch.Tensor | None = None,
    history_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """构造按样本有效长度生成的 RoPE 逻辑位置编号。

    物理布局可以包含任意位置的 padding，但有效 token 的编号始终遵循：
    ``scene -> state, skill -> current state``。scene/history 的位置按各自
    mask 的有效计数生成，不依赖有效 token 是否位于物理布局前段；无效
    scene/history token 的位置固定为 0，并由 attention mask 完全排除。
    """
    if scene_mask is None:
        scene_mask = torch.ones(
            (batch_size, scene_length), dtype=torch.bool, device=device
        )
    if history_mask is None:
        history_mask = torch.ones(
            (batch_size, history_length), dtype=torch.bool, device=device
        )
    if tuple(scene_mask.shape) != (batch_size, scene_length):
        raise ValueError("scene_mask shape does not match the physical scene layout")
    if tuple(history_mask.shape) != (batch_size, history_length):
        raise ValueError("history_mask shape does not match the physical history layout")

    scene_mask_long = scene_mask.to(dtype=torch.long)
    history_token_mask = history_mask.repeat_interleave(2, dim=1)
    history_mask_long = history_token_mask.to(dtype=torch.long)
    scene_valid_lengths = scene_mask_long.sum(dim=1)
    history_valid_lengths = history_mask_long.sum(dim=1)
    scene_positions = torch.cumsum(scene_mask_long, dim=1) - 1
    scene_positions = torch.where(
        scene_mask,
        scene_positions,
        torch.zeros_like(scene_positions),
    )
    history_positions = (
        torch.cumsum(history_mask_long, dim=1) - 1
    ) + scene_valid_lengths.unsqueeze(1)
    history_positions = torch.where(
        history_token_mask,
        history_positions,
        torch.zeros_like(history_positions),
    )
    current_state_positions = (scene_valid_lengths + history_valid_lengths).unsqueeze(1)
    return torch.cat(
        (
            scene_positions,
            history_positions,
            current_state_positions,
        ),
        dim=1,
    )


def build_role_ids(
    *, batch_size: int, scene_length: int, history_length: int, device
) -> torch.Tensor:
    """按场景、状态、技能顺序编号；最新状态不拥有单独的类型。"""
    history_roles = torch.stack(
        (
            torch.full((history_length,), ROLE_STATE, dtype=torch.long, device=device),
            torch.full((history_length,), ROLE_SKILL, dtype=torch.long, device=device),
        ),
        dim=1,
    ).reshape(2 * history_length)
    role_ids = torch.cat(
        (
            torch.full((scene_length,), ROLE_SCENE, dtype=torch.long, device=device),
            history_roles,
            torch.full((1,), ROLE_STATE, dtype=torch.long, device=device),
        )
    ).unsqueeze(0).expand(batch_size, -1)
    return role_ids
