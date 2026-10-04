"""模型输入编码器：把 batch 字段转换成 Transformer token。"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from ..data.input_contract import TOKEN_ENCODING_CONTRACT
from ..data.spec import DataSpec


ROLE_SCENE = TOKEN_ENCODING_CONTRACT["role_ids"]["scene"]
ROLE_STATE = TOKEN_ENCODING_CONTRACT["role_ids"]["state"]
ROLE_SKILL = TOKEN_ENCODING_CONTRACT["role_ids"]["skill"]


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
        self.skill_norm = nn.LayerNorm(d_model)
        self.state_proj = nn.Linear(data_spec.state_dim, d_model)
        self.state_null_proj = nn.Linear(data_spec.state_dim, d_model, bias=False)
        self.state_norm = nn.LayerNorm(d_model)
        self.scene_proj = nn.ModuleList(
            nn.Linear(data_spec.scene_dim, d_model)
            for _ in range(data_spec.num_scene_types)
        )
        self.scene_norm = nn.LayerNorm(d_model)
        self.role_embed = nn.Embedding(3, d_model)

    @property
    def max_token_count(self) -> int:
        """返回由各上下文块容量自动换算出的最大物理 token 数。"""
        return (
            self.config.scene_capacity
            + 2 * self.config.history_capacity
            + 1
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self._materialize_compact_history(batch)
        self._validate_batch(batch)
        scene_vectors = batch["scene_vectors"]
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
        scene_embeds = self.scene_norm(scene_embeds)

        history = self.embed_history(batch)
        # 请求时冻结的状态位于该次技能之前；技能只能读取此前已知的状态。
        history_tokens = torch.stack((history["state"], history["skill"]), dim=2).reshape(
            batch_size, history_token_length, d_model,
        )
        # 历史与最新状态使用同一个投影和归一化，不增加特殊当前状态参数。
        current_state = self._embed_state(
            batch["current_state_vectors"],
            batch.get("current_state_null_mask"),
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
        """返回独立的历史技能、状态 embedding，均为 d_model 维。"""
        self._materialize_compact_history(batch)
        self._validate_batch(batch)
        return {
            "skill": self.skill_norm(
                self.skill_embed(batch["history_skill_ids"])
                + self.skill_feat_proj(batch["history_skill_features"]),
            ),
            "state": self._embed_state(
                batch["history_state_vectors"],
                batch.get("history_state_null_mask"),
            ),
        }

    def _embed_state(self, values: torch.Tensor, null_mask: torch.Tensor | None) -> torch.Tensor:
        """历史状态和当前状态共用相同的数值、缺失值投影与归一化。"""
        if null_mask is None:
            null_mask = torch.zeros_like(values, dtype=torch.bool)
        return self.state_norm(
            self.state_proj(values) + self.state_null_proj(null_mask.to(dtype=values.dtype)),
        )

    def _materialize_compact_history(self, batch: dict[str, torch.Tensor]) -> None:
        """在 GPU 上按 end/length 从 source bank gather 出模型原有的 dense 窗口。"""
        if "history_skill_ids" in batch:
            return
        required = (
            "history_lengths",
            "history_ends",
            "history_mask",
            "history_bank_skill_ids",
            "history_bank_skill_features",
            "history_bank_state_vectors",
            "history_bank_state_null_mask",
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(f"compact history batch is missing fields: {missing}")

        device = batch["scene_vectors"].device
        lengths = batch["history_lengths"]
        ends = batch["history_ends"]
        bank_ids = batch["history_bank_skill_ids"]
        bank_features = batch["history_bank_skill_features"]
        bank_states = batch["history_bank_state_vectors"]
        bank_nulls = batch["history_bank_state_null_mask"]
        tensors = (lengths, ends, bank_ids, bank_features, bank_states, bank_nulls)
        if any(tensor.device != device for tensor in tensors):
            raise ValueError("compact history tensors must be moved to one device before model forward")

        history_width = batch["history_mask"].shape[1]
        relative = torch.arange(history_width, device=device, dtype=ends.dtype)
        valid = relative.unsqueeze(0) < lengths.unsqueeze(1)
        indices = ends.unsqueeze(1) - lengths.unsqueeze(1) + relative.unsqueeze(0)
        safe_indices = torch.where(valid, indices, torch.zeros_like(indices))

        selected_ids = bank_ids[safe_indices]
        selected_features = bank_features[safe_indices]
        selected_states = bank_states[safe_indices]
        selected_nulls = bank_nulls[safe_indices]
        invalid = ~valid.unsqueeze(-1)
        batch["history_skill_ids"] = selected_ids.masked_fill(~valid, 0)
        batch["history_skill_features"] = selected_features.masked_fill(invalid, 0.0)
        batch["history_state_vectors"] = selected_states.masked_fill(invalid, 0.0)
        batch["history_state_null_mask"] = torch.where(
            valid.unsqueeze(-1),
            selected_nulls,
            torch.ones_like(selected_nulls),
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
        if batch["history_skill_ids"].shape[1] != batch["history_state_vectors"].shape[1]:
            raise ValueError("history skill/state lengths must match")
        if batch["current_state_vectors"].ndim != 2:
            raise ValueError("current_state_vectors must have shape [batch, state_dim]")
        if batch["current_state_vectors"].shape[-1] != expected.state_dim:
            raise ValueError("current state dimension does not match the data spec")
        if batch["history_state_vectors"].shape[-1] != expected.state_dim:
            raise ValueError("history state dimension does not match the data spec")
        for key in ("current_state_null_mask", "history_state_null_mask"):
            values_key = key.replace("null_mask", "vectors")
            if key in batch and batch[key].shape != batch[values_key].shape:
                raise ValueError(f"{key} must match {values_key}")
        if batch["scene_vectors"].shape[-1] != expected.scene_dim:
            raise ValueError("scene dimension does not match the compiled cache data spec")
        if batch["history_skill_features"].shape[-1] != expected.skill_feature_dim:
            raise ValueError("skill feature dimension does not match the compiled cache data spec")
        if "action_legal_mask" in batch and batch["action_legal_mask"].shape != (
            batch["scene_vectors"].shape[0], expected.num_actions,
        ):
            raise ValueError("action legal mask does not match the data spec")
        if "label_index" in batch and not torch.all(
            (batch["label_index"] >= 0) & (batch["label_index"] < expected.num_actions)
        ):
            raise ValueError("label index is outside the action vocabulary range")


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
