"""复用 checkpoint 模型权重的纯 Tensor ONNX 候选打分入口。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from common.policy.model.split_encoder import run_split_encoder
from common.policy.model.trace import trace_encoder


class OnnxPolicy(nn.Module):
    """只接收张量并输出策略后处理前的候选 ``raw_logits``。"""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.model.enable_kv_cache(False)
        super().train(False)

    def train(self, mode: bool = True):
        """部署图固定为 eval 模式，避免导出 dropout 或训练态分支。"""
        if mode:
            raise ValueError("OnnxPolicy only supports eval mode")
        return super().train(False)

    def forward(
        self,
        scene_vectors: torch.Tensor,
        scene_types: torch.Tensor,
        scene_mask: torch.Tensor,
        history_skill_ids: torch.Tensor,
        history_skill_features: torch.Tensor,
        history_state_vectors: torch.Tensor,
        history_state_null_mask: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_skill_ids: torch.Tensor,
        candidate_skill_features: torch.Tensor,
        candidate_state_vectors: torch.Tensor,
        candidate_state_null_mask: torch.Tensor,
    ) -> torch.Tensor:
        """执行无字符串、无策略后处理、无内部 KV cache 的候选打分。"""
        batch = _build_batch(
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
        encoded = self.model.input_encoder(batch)
        _prefix_hidden, candidate_hidden, cls_hidden, _layers, _attentions = run_split_encoder(
            self.model.encoder,
            encoded,
            force_explicit_mask=True,
        )
        return self.model.scorer(
            cls_hidden=cls_hidden[:, -1, :],
            candidate_hidden=candidate_hidden,
        )

    @torch.no_grad()
    def trace(self, *inputs: torch.Tensor) -> "OnnxPolicyTrace":
        """保留稳定 attention 路径的逐层 hidden/权重，供导出验收使用。"""
        batch = _build_batch(*inputs)
        encoded = self.model.input_encoder(batch)
        trace = trace_encoder(self.model.encoder, encoded)
        hidden = trace.hidden
        candidate_hidden = hidden[:, encoded["candidate_positions"], :]
        logits = self.model.scorer(
            cls_hidden=hidden[:, -1, :],
            candidate_hidden=candidate_hidden,
        )
        return OnnxPolicyTrace(
            encoded=trace.encoded,
            layer_hidden=trace.layer_hidden,
            hidden=hidden,
            attentions=trace.attentions,
            logits=logits,
        )


@dataclass(frozen=True)
class OnnxPolicyTrace:
    """固定容量验收所需的逐层输出。"""

    encoded: dict[str, torch.Tensor]
    layer_hidden: tuple[torch.Tensor, ...]
    hidden: torch.Tensor
    attentions: tuple[torch.Tensor, ...]
    logits: torch.Tensor


def _build_batch(*inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    (
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
    ) = inputs
    return {
        "scene_vectors": scene_vectors,
        "scene_types": scene_types,
        "scene_mask": scene_mask,
        "history_skill_ids": history_skill_ids,
        "history_skill_features": history_skill_features,
        "history_state_vectors": history_state_vectors,
        "history_state_null_mask": history_state_null_mask,
        "history_mask": history_mask,
        "candidate_skill_ids": candidate_skill_ids,
        "candidate_skill_features": candidate_skill_features,
        "candidate_state_vectors": candidate_state_vectors,
        "candidate_state_null_mask": candidate_state_null_mask,
        # 合法性属于宿主后处理；该占位张量只满足共享 encoder 的 shape 契约。
        "candidate_legal_mask": torch.ones_like(
            candidate_skill_ids,
            dtype=torch.bool,
        ),
    }


def stable_masked_softmax(
    scores: torch.Tensor,
    blocked: torch.Tensor,
) -> torch.Tensor:
    """让全屏蔽注意力行稳定地产生零权重，而不是 ``NaN``。"""
    masked_scores = scores.masked_fill(blocked, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked_scores, dim=-1)
    return weights.masked_fill(blocked, 0.0)
