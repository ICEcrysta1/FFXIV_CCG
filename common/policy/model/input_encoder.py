"""模型输入编码器：把 batch 字段转换成 Transformer token。"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from ..data.spec import DataSpec
from .activation import (
    activation_hidden,
    gated_hidden_dim,
    resolve_pointwise_activation,
    uses_gate,
)


ROLE_SCENE = 0
ROLE_HISTORY = 1
ROLE_CANDIDATE = 2
ROLE_CLS = 3

SEG_SCENE = 0
SEG_HISTORY = 1
SEG_CANDIDATE = 2


class CandidateInputEncoder(nn.Module):
    """编码 scene、历史和候选上下文，并返回稳定的 token 布局元数据。"""

    def __init__(self, data_spec: DataSpec, config: ModelConfig, vocab_size: int):
        super().__init__()
        if data_spec.scene_dim <= 0:
            raise ValueError("CandidateInputEncoder requires a non-empty scene vector")

        d_model = config.d_model
        self.data_spec = data_spec
        self.config = config
        pair_dim = config.pair_embedding_dim
        self.skill_embed = nn.Embedding(vocab_size, pair_dim, padding_idx=0)
        self.skill_feat_proj = nn.Linear(data_spec.skill_feature_dim, pair_dim)
        self.state_proj = nn.Linear(data_spec.state_dim, pair_dim)
        self.state_null_proj = nn.Linear(data_spec.state_dim, pair_dim, bias=False)
        # pair 融合与主干 FFN 共用 model.transformer_activation：SwiGLU 走门控
        # 三投影，GELU/ReLU 走原来的单条隐藏层。
        pair_fusion_input = pair_dim * 2
        self.pair_fusion_gated = uses_gate(config.transformer_activation)
        self.pair_fusion_activation = resolve_pointwise_activation(
            config.transformer_activation
        )
        pair_fusion_hidden = gated_hidden_dim(
            config.transformer_activation,
            in_features=pair_fusion_input,
            out_features=pair_dim,
            hidden_dim=pair_fusion_input,
        )
        if self.pair_fusion_gated:
            self.pair_fusion_gate = nn.Linear(pair_fusion_input, pair_fusion_hidden)
        self.pair_fusion_up = nn.Linear(pair_fusion_input, pair_fusion_hidden)
        self.pair_fusion_norm = nn.LayerNorm(pair_fusion_hidden)
        self.pair_fusion_down = nn.Linear(pair_fusion_hidden, pair_dim)
        self.scene_proj = nn.ModuleList(
            nn.Linear(data_spec.scene_dim, pair_dim)
            for _ in range(data_spec.num_scene_types)
        )
        self.token_embedding = nn.Sequential(
            nn.Linear(pair_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, pair_dim))
        self.role_embed = nn.Embedding(4, d_model)
        self.segment_embed = nn.Embedding(3, d_model)

    @property
    def max_token_count(self) -> int:
        """返回由各上下文块容量自动换算出的最大物理 token 数。"""
        return (
            self.config.scene_capacity
            + self.config.history_capacity
            + self.data_spec.num_candidates
            + 1
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self._materialize_compact_history(batch)
        self._validate_batch(batch)
        scene_vectors = batch["scene_vectors"]
        batch_size, scene_length, _ = scene_vectors.shape
        history_length = batch["history_skill_ids"].shape[1]
        candidate_count = self.data_spec.num_candidates
        candidate_token_count = candidate_count
        # 物理 token 布局仍按 batch 的最大 scene/history 宽度补齐；
        # RoPE 使用的逻辑位置由有效长度单独生成，不再把 padding 当成时间步。
        total_length = scene_length + history_length + candidate_token_count + 1
        if total_length > self.max_token_count:
            raise ValueError(
                "physical token sequence length exceeds computed model capacity: "
                f"{total_length} > {self.max_token_count}"
            )
        device = scene_vectors.device
        pair_dim = self.config.pair_embedding_dim

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
            index=scene_type_indices.expand(-1, -1, 1, pair_dim),
        ).squeeze(2)

        pair_embeddings = self.embed_pairs(batch)
        history_pair = pair_embeddings["history"]
        candidate_pair = pair_embeddings["candidate"]
        candidate_tokens = candidate_pair

        cls_tokens = self.cls_token.expand(batch_size, 1, pair_dim)
        content_tokens = torch.cat(
            (scene_embeds, history_pair, candidate_tokens, cls_tokens),
            dim=1,
        )
        tokens = self.token_embedding(content_tokens)

        position_ids = build_position_ids(
            batch_size=batch_size,
            scene_length=scene_length,
            history_length=history_length,
            candidate_count=candidate_token_count,
            device=device,
            scene_mask=batch["scene_mask"],
            history_mask=batch["history_mask"],
        )
        role_ids, segment_ids = build_role_and_segment_ids(
            batch_size=batch_size,
            scene_length=scene_length,
            history_length=history_length,
            candidate_count=candidate_token_count,
            device=device,
        )
        tokens = tokens + self.role_embed(role_ids) + self.segment_embed(segment_ids)

        candidate_start = scene_length + history_length
        candidate_positions = torch.arange(
            candidate_start,
            candidate_start + candidate_count,
            device=device,
        )
        candidate_position_ids = position_ids[
            :, candidate_start : candidate_start + candidate_count
        ]
        cls_position_ids = position_ids[:, candidate_start + candidate_count :]
        prefix_valid = torch.cat(
            (batch["scene_mask"], batch["history_mask"]),
            dim=1,
        )
        candidate_valid = torch.ones(
            (batch_size, candidate_count),
            dtype=torch.bool,
            device=device,
        )
        cls_valid = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
        valid = torch.cat((prefix_valid, candidate_valid, cls_valid), dim=1)
        return {
            "tokens": tokens,
            "padding_mask": ~valid,
            "prefix_valid": prefix_valid,
            "candidate_valid": candidate_valid,
            "cls_valid": cls_valid,
            "scene_length": scene_length,
            "history_length": history_length,
            "prefix_length": scene_length + history_length,
            "candidate_count": candidate_count,
            "position_ids": position_ids,
            "role_ids": role_ids,
            "segment_ids": segment_ids,
            "candidate_positions": candidate_positions,
            "candidate_position_ids": candidate_position_ids,
            "cls_position_ids": cls_position_ids,
        }

    def embed_pairs(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """返回未投影到 Transformer 维度的 history/candidate pair embedding。"""
        self._materialize_compact_history(batch)
        self._validate_batch(batch)
        return {
            "history": self._embed_pair(
                batch["history_skill_ids"],
                batch["history_skill_features"],
                batch["history_state_vectors"],
                batch.get("history_state_null_mask"),
            ),
            "candidate": self._embed_pair(
                batch["candidate_skill_ids"],
                batch["candidate_skill_features"],
                batch["candidate_state_vectors"],
                batch.get("candidate_state_null_mask"),
            ),
        }

    def _embed_pair(
        self,
        skill_ids: torch.Tensor,
        skill_features: torch.Tensor,
        state_vectors: torch.Tensor,
        state_null_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        skill_embed = self.skill_embed(skill_ids) + self.skill_feat_proj(skill_features)
        state_embed = self._embed_state(state_vectors, state_null_mask)
        return self._fuse_pair(torch.cat((skill_embed, state_embed), dim=-1))

    def _fuse_pair(self, paired: torch.Tensor) -> torch.Tensor:
        """按配置激活融合 skill/state pair，再归一化并投回 pair 维度。"""
        hidden = activation_hidden(
            self.pair_fusion_activation,
            self.pair_fusion_up(paired),
            gate=self.pair_fusion_gate(paired) if self.pair_fusion_gated else None,
        )
        return self.pair_fusion_down(self.pair_fusion_norm(hidden))

    def _embed_state(self, values: torch.Tensor, null_mask: torch.Tensor | None) -> torch.Tensor:
        if null_mask is None:
            null_mask = torch.zeros_like(values, dtype=torch.bool)
        return self.state_proj(values) + self.state_null_proj(null_mask.to(dtype=values.dtype))

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
        relative = torch.arange(history_width, device=device, dtype=torch.long)
        lengths = lengths.to(dtype=torch.long)
        ends = ends.to(dtype=torch.long)
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
        if batch["candidate_skill_ids"].shape[1] != expected.num_candidates:
            raise ValueError("candidate count does not match the compiled cache data spec")
        if batch["candidate_state_vectors"].shape[1] != expected.num_candidates:
            raise ValueError("candidate skill/state counts must match the compiled cache data spec")
        if batch["candidate_state_vectors"].shape[-1] != expected.state_dim:
            raise ValueError("candidate state dimension does not match the compiled cache data spec")
        if batch["scene_vectors"].shape[-1] != expected.scene_dim:
            raise ValueError("scene dimension does not match the compiled cache data spec")
        if batch["candidate_skill_features"].shape[-1] != expected.skill_feature_dim:
            raise ValueError("skill feature dimension does not match the compiled cache data spec")
        if batch["candidate_legal_mask"].shape[1] != expected.num_candidates:
            raise ValueError("candidate legal mask does not match the compiled cache data spec")
        if "label_index" in batch and not torch.all(
            (batch["label_index"] >= 0) & (batch["label_index"] < expected.num_candidates)
        ):
            raise ValueError("label index is outside the compiled cache candidate range")


def build_position_ids(
    *,
    batch_size: int,
    scene_length: int,
    history_length: int,
    candidate_count: int,
    device,
    scene_mask: torch.Tensor | None = None,
    history_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """构造按样本有效长度生成的 RoPE 逻辑位置编号。

    物理布局可以包含任意位置的 padding，但有效 token 的编号始终遵循：
    ``scene -> history -> candidate -> CLS``。scene/history 的位置按各自
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
    history_mask_long = history_mask.to(dtype=torch.long)
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
        history_mask,
        history_positions,
        torch.zeros_like(history_positions),
    )
    candidate_start = scene_valid_lengths + history_valid_lengths
    # 候选位置与其他有效 token 一样按逻辑序列递增，所有区域共用同一套 RoPE。
    candidate_positions = candidate_start.unsqueeze(1) + torch.arange(
        candidate_count, device=device
    ).unsqueeze(0)
    cls_positions = (candidate_start + candidate_count).unsqueeze(1)
    return torch.cat(
        (
            scene_positions,
            history_positions,
            candidate_positions,
            cls_positions,
        ),
        dim=1,
    )


def build_role_and_segment_ids(
    *, batch_size: int, scene_length: int, history_length: int, candidate_count: int, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """构造 token role 与上下文 segment 编号。"""
    role_ids = torch.cat(
        (
            torch.full((scene_length,), ROLE_SCENE, device=device),
            torch.full((history_length,), ROLE_HISTORY, device=device),
            torch.full((candidate_count,), ROLE_CANDIDATE, device=device),
            torch.tensor([ROLE_CLS], device=device),
        )
    ).unsqueeze(0).expand(batch_size, -1)
    segment_ids = torch.cat(
        (
            torch.full((scene_length,), SEG_SCENE, device=device),
            torch.full((history_length,), SEG_HISTORY, device=device),
            torch.full((candidate_count + 1,), SEG_CANDIDATE, device=device),
        )
    ).unsqueeze(0).expand(batch_size, -1)
    return role_ids, segment_ids
