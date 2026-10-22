"""模型输入编码器：把 batch 字段转换成 Transformer token。"""

from __future__ import annotations

import torch
from torch import nn

from ...config import ModelConfig
from ...data.contracts.context_fields import MODEL_INPUT_FIELDS, tensor_dimensions
from ...data.contracts.input_contract import TOKEN_ENCODING_CONTRACT
from ...data.contracts.spec import DataSpec

ROLE_SCENE = TOKEN_ENCODING_CONTRACT["role_ids"]["scene"]
ROLE_STATE = TOKEN_ENCODING_CONTRACT["role_ids"]["state"]
ROLE_SKILL = TOKEN_ENCODING_CONTRACT["role_ids"]["skill"]
ROLE_DEAD = TOKEN_ENCODING_CONTRACT["role_ids"]["dead"]


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
        self.dead_embed = nn.Embedding(1, d_model)
        self.role_embed = nn.Embedding(4, d_model)
        # 跳过默认随机初始化，让重置标识从零开始且不改变公共参数的随机数序列。
        self.state_reset_proj = nn.utils.skip_init(
            nn.Linear, data_spec.base_state_dim, d_model, bias=False,
            device=self.role_embed.weight.device, dtype=self.role_embed.weight.dtype,
        )
        nn.init.zeros_(self.state_reset_proj.weight)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        history = self.embed_history(batch)
        scene_vectors = batch["scene_vectors"].float()
        batch_size, scene_length, _ = scene_vectors.shape
        history_length = batch["history_skill_ids"].shape[1]
        dead_length = batch["history_dead_positions"].shape[1]
        # 输入各块可以独立补齐；进入 Transformer 前按每个样本的有效顺序压紧，
        # 避免不同样本的场景/历史最大宽度相加后超过共享预算。
        history_token_length = 2 * history_length
        prefix_width = min(self.config.context_capacity - 1, scene_length + history_token_length + dead_length)
        device = scene_vectors.device
        d_model = self.config.d_model

        # 所有 scene 类型都先投影，再由张量索引选择对应类型，避免根据输入值
        # 进入 Python 分支。这样 scene_types 仍然是图输入，而不是导出时的常量。
        # 保留普通 Linear，只在数值投影边界关闭混合精度。
        with torch.autocast(device_type=device.type, enabled=False):
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
        # 请求时冻结的状态位于该次技能之前；技能只能读取此前已知的状态。
        history_tokens = torch.stack((history["state"], history["skill"]), dim=2).reshape(
            batch_size, history_token_length, d_model,
        )
        # 历史与最新状态使用同一个投影，不增加特殊当前状态参数。
        current_state = self._embed_state(
            batch["current_state_vectors"],
            batch["current_state_null_mask"],
            batch["current_state_reset_mask"],
        ).unsqueeze(1)

        source_valid, source_positions, source_roles = _source_layout(
            batch["scene_mask"], batch["history_mask"],
            batch["history_dead_positions"], batch["history_dead_mask"],
        )
        dead_tokens = self.dead_embed.weight[0].float().view(1, 1, d_model).expand(batch_size, dead_length, -1)
        source_tokens = torch.cat((scene_embeds, history_tokens, dead_tokens), dim=1)
        # 无效项不能通过乘零消除：补齐内容即使为 NaN，也不得污染 scatter 的有效结果。
        source_tokens = source_tokens.masked_fill(~source_valid.unsqueeze(-1), 0)
        packed_prefix = source_tokens.new_zeros((batch_size, prefix_width, d_model))
        if prefix_width:
            packed_prefix = packed_prefix.scatter_add(
                1, source_positions.clamp(min=0).unsqueeze(-1).expand(-1, -1, d_model), source_tokens,
            )
        prefix_counts = source_valid.to(torch.long).sum(dim=1)
        prefix_valid = torch.arange(prefix_width, device=device)[None, :] < prefix_counts[:, None]
        content_tokens = torch.cat((packed_prefix, current_state), dim=1)
        position_ids = build_position_ids(prefix_valid=prefix_valid)
        role_ids = _pack_role_ids(source_positions, source_roles, source_valid, prefix_width)
        with torch.autocast(device_type=device.type, enabled=False):
            tokens = content_tokens + self.role_embed(role_ids).float()
            # 全部角色在 FP32 中合成和统一尺度，不去均值，也不引入仿射参数。
            tokens = torch.nn.functional.rms_norm(tokens, (d_model,), eps=1e-5)

        current_state_position = prefix_width
        current_state_positions = torch.full(
            (batch_size,), current_state_position, dtype=torch.long, device=device,
        )
        current_state_valid = torch.ones(
            (batch_size, 1),
            dtype=torch.bool,
            device=device,
        )
        valid = torch.cat((prefix_valid, current_state_valid), dim=1)
        history_state_positions = source_positions[:, scene_length:scene_length + history_token_length:2]
        history_state_positions = history_state_positions.masked_fill(~batch["history_mask"], 0)
        history_skill_positions = (history_state_positions + 1).masked_fill(~batch["history_mask"], 0)
        return {
            "tokens": tokens,
            "padding_mask": ~valid,
            "prefix_valid": prefix_valid,
            "valid": valid,
            "scene_length": scene_length,
            "prefix_length": prefix_width,
            "position_ids": position_ids,
            "role_ids": role_ids,
            "history_skill_positions": history_skill_positions,
            "history_state_positions": history_state_positions,
            "history_dead_positions": source_positions[:, scene_length + history_token_length:].masked_fill(~batch["history_dead_mask"], 0),
            "current_state_position": current_state_position,
            "current_state_positions": current_state_positions,
        }

    def embed_history(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """返回历史技能、状态的原始 content；role 和 RMSNorm 由 forward 统一施加。"""
        self._validate_batch(batch)
        with torch.autocast(device_type=batch["history_skill_features"].device.type, enabled=False):
            return {
                "skill": self.skill_embed(batch["history_skill_ids"]).float()
                + self.skill_feat_proj(batch["history_skill_features"].float()),
                "state": self._embed_state(
                    batch["history_state_vectors"],
                    batch["history_state_null_mask"],
                    batch["history_state_reset_mask"],
                ),
            }

    def _embed_state(self, values, null_mask, reset_mask) -> torch.Tensor:
        """已编码数值、缺失及 ABS 重置标识统一在 FP32 中投影和相加。"""
        with torch.autocast(device_type=values.device.type, enabled=False):
            return (
                self.state_proj(values.float())
                + self.state_null_proj(null_mask.float())
                + self.state_reset_proj(reset_mask.float())
            )

    def _validate_batch(self, batch: dict[str, torch.Tensor]) -> None:
        expected = self.data_spec
        scene_length = batch["scene_vectors"].shape[1]
        history_length = batch["history_skill_ids"].shape[1]
        dead = batch.get("history_dead_positions")
        dead_length = dead.shape[1] if isinstance(dead, torch.Tensor) and dead.ndim == 2 else None
        dimensions = tensor_dimensions(expected, batch=batch["scene_vectors"].shape[0], scene=scene_length, history=history_length, dead=dead_length)
        for field in MODEL_INPUT_FIELDS:
            if field.name not in batch:
                raise ValueError(f"missing model input: {field.name}")
            # 数值入口固定为 FP32，禁止在投影之前提前量化。
            field.validate_tensor(batch[field.name], dimensions, torch=torch)
        if (batch["scene_mask"].device.type == "cpu"
                and not torch.onnx.is_in_onnx_export() and not torch.compiler.is_compiling()):
            dead_mask = batch["history_dead_mask"]
            dead_positions = batch["history_dead_positions"]
            history_counts = batch["history_mask"].sum(dim=1)
            if bool((dead_mask & ((dead_positions < 0) | (dead_positions > history_counts[:, None]))).any()):
                raise ValueError("history dead positions must reference a visible action boundary")
            # 忽略补齐位置，只检查有效标记的稳定顺序。
            previous = dead_positions.masked_fill(~dead_mask, 0).cummax(dim=1).values
            if bool((dead_mask & (dead_positions < previous)).any()):
                raise ValueError("history dead positions must be nondecreasing")
            token_counts = batch["scene_mask"].sum(dim=1) + 2 * history_counts + dead_mask.sum(dim=1) + 1
            if bool((token_counts > self.config.context_capacity).any()):
                raise ValueError("context token count exceeds model.context_capacity")
        if "action_legal_mask" in batch and batch["action_legal_mask"].shape != (
            batch["scene_vectors"].shape[0], expected.num_actions,
        ):
            raise ValueError("action legal mask does not match the data spec")


def build_position_ids(*, prefix_valid: torch.Tensor) -> torch.Tensor:
    """压紧后的有效前缀连续编号，补齐位置为零，当前状态接在有效前缀之后。"""
    prefix_counts = prefix_valid.to(torch.long).sum(dim=1)
    positions = prefix_valid.to(torch.long).cumsum(dim=1) - 1
    positions = positions.masked_fill(~prefix_valid, 0)
    return torch.cat((positions, prefix_counts[:, None]), dim=1)


def build_role_ids(
    *, scene_mask: torch.Tensor, history_mask: torch.Tensor, prefix_width: int,
    history_dead_positions: torch.Tensor | None = None, history_dead_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """按真实 token 位置生成角色；死亡不改变普通状态/技能的配对。"""
    if history_dead_positions is None:
        history_dead_positions = torch.empty((scene_mask.shape[0], 0), dtype=torch.long, device=scene_mask.device)
    if history_dead_mask is None:
        history_dead_mask = torch.zeros_like(history_dead_positions, dtype=torch.bool)
    valid, positions, roles = _source_layout(scene_mask, history_mask, history_dead_positions, history_dead_mask)
    return _pack_role_ids(positions, roles, valid, prefix_width)


def _source_layout(scene_mask, history_mask, dead_positions, dead_mask):
    """分别计算场景、动作对和离散死亡的压紧位置，保持同边界标记原顺序。"""
    batch_size = scene_mask.shape[0]
    device = scene_mask.device
    scene_counts = scene_mask.to(torch.long).sum(dim=1)
    action_indices = history_mask.to(torch.long).cumsum(dim=1) - 1
    dead_before = ((dead_positions[:, None, :] <= action_indices[:, :, None]) & dead_mask[:, None, :]).sum(dim=-1)
    state_positions = scene_counts[:, None] + 2 * action_indices + dead_before
    action_positions = torch.stack((state_positions, state_positions + 1), dim=-1).reshape(batch_size, -1)
    marker_positions = scene_counts[:, None] + 2 * dead_positions + dead_mask.to(torch.long).cumsum(dim=1) - 1
    valid = torch.cat((scene_mask, history_mask.repeat_interleave(2, dim=1), dead_mask), dim=1)
    positions = torch.cat((scene_mask.to(torch.long).cumsum(dim=1) - 1, action_positions, marker_positions), dim=1)
    positions = positions.masked_fill(~valid, 0).to(torch.long)
    action_roles = torch.tensor([ROLE_STATE, ROLE_SKILL], dtype=torch.long, device=device).repeat(history_mask.shape[1])
    roles = torch.cat((torch.full_like(scene_mask, ROLE_SCENE, dtype=torch.long),
                       action_roles[None, :].expand(batch_size, -1),
                       torch.full_like(dead_mask, ROLE_DEAD, dtype=torch.long)), dim=1)
    return valid, positions, roles


def _pack_role_ids(positions, roles, valid, prefix_width):
    prefix_roles = roles.new_zeros((roles.shape[0], prefix_width))
    if prefix_width:
        prefix_roles = prefix_roles.scatter_add(1, positions, roles.masked_fill(~valid, 0))
    return torch.cat((prefix_roles, roles.new_full((roles.shape[0], 1), ROLE_STATE)), dim=1)
