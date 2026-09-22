"""与职业无关的候选动作 Transformer 模型。"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import asdict

import torch
import torch.nn as nn

from .candidate_scorer import CandidateScorer
from ..config import ModelConfig
from .attention_residual import FullAttentionResidual
from .input_encoder import CandidateInputEncoder
from .kv_cache import TransformerKVCache, encode_with_kv_cache
from .position_encoding import RotaryPositionEncoding
from .repetition import RepetitionConfig, apply_repetition_penalty
from .split_encoder import run_split_encoder
from ..data.spec import DataSpec
from .trace import ModelTrace, TraceableTransformerEncoderLayer, trace_encoder


class CandidateTransformerModel(nn.Module):
    """对当前候选动作集合进行排序的策略模型。"""

    def __init__(
        self,
        data_spec: DataSpec,
        config: ModelConfig,
        vocab_size: int,
        repetition: RepetitionConfig | None = None,
    ):
        super().__init__()
        if config.n_heads <= 0 or config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if config.num_kv_heads <= 0 or config.num_kv_heads > config.n_heads:
            raise ValueError("num_kv_heads must be between 1 and n_heads")
        if config.n_heads % config.num_kv_heads != 0:
            raise ValueError("n_heads must be divisible by num_kv_heads")
        if config.full_attention_residuals and not config.transformer_norm_first:
            raise ValueError(
                "full_attention_residuals requires transformer_norm_first=true"
            )

        self.data_spec = data_spec
        self.config = config
        self.repetition = repetition or RepetitionConfig()
        self.input_encoder = CandidateInputEncoder(data_spec, config, vocab_size)
        encoder_layer = TraceableTransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            num_kv_heads=config.num_kv_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation=config.transformer_activation,
            batch_first=True,
            norm_first=config.transformer_norm_first,
        )
        # Pre-LN 保留残差路径的稳定尺度，末尾 LayerNorm 统一 scorer 和分析输出的输入尺度。
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            config.n_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        # 位置编码属于 attention 执行边界；输入编码器只提供逻辑 position_ids。
        self.encoder.rotary_position_encoding = RotaryPositionEncoding(
            config.d_model // config.n_heads,
            theta=config.rope_theta,
        )
        if config.full_attention_residuals:
            # attention 与 FFN 各占一个深度步，最后再用一个 query 聚合输出。
            self.encoder.attention_residual = FullAttentionResidual(
                d_model=config.d_model,
                num_queries=2 * config.n_layers + 1,
            )
        self.scorer = CandidateScorer(
            d_model=config.d_model,
            dropout=config.dropout,
            activation=config.transformer_activation,
        )
        self._init_weights()
        self._kv_cache_enabled = False
        self._kv_cache: TransformerKVCache | None = None
        self._runtime_debug = None

    def _init_weights(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        nn.init.normal_(self.input_encoder.cls_token, std=0.02)
        attention_residual = getattr(self.encoder, "attention_residual", None)
        if attention_residual is not None:
            attention_residual.reset_parameters()

    def train(self, mode: bool = True):
        """切换训练模式时清空推理缓存，避免缓存进入反向传播路径。"""
        result = super().train(mode)
        if mode:
            self.reset_kv_cache()
        return result

    def enable_kv_cache(self, enabled: bool = True) -> None:
        """开启或关闭模型内部的推理 KV-Cache。"""
        self._kv_cache_enabled = bool(enabled)
        self.reset_kv_cache()

    def reset_kv_cache(self) -> None:
        """清空当前战斗的运行时缓存。"""
        self._kv_cache = None

    def enable_activation_checkpoint_ffn(self, enabled: bool = True) -> None:
        """开启或关闭训练时的 FFN activation checkpoint。"""
        for layer in self.encoder.layers:
            layer.set_activation_checkpoint_ffn(enabled)

    def enable_activation_checkpoint_attention(self, enabled: bool = True) -> None:
        """开启或关闭训练时的 Attention activation checkpoint。"""
        for layer in self.encoder.layers:
            layer.set_activation_checkpoint_attention(enabled)

    def set_runtime_debug(self, recorder=None) -> None:
        """接入可选的训练运行时调试记录器。"""
        self._runtime_debug = recorder
        for index, layer in enumerate(self.encoder.layers):
            layer.set_runtime_debug(recorder, layer_index=index)

    def _debug_stage(self, name: str):
        if self._runtime_debug is None:
            return nullcontext()
        return self._runtime_debug.stage(name)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """执行前向传播；没有 label 时只返回 logits。"""
        with self._debug_stage("input_encoder"):
            encoded = self.input_encoder(batch)
        cached_candidate_positions = None
        if self.training or not self._kv_cache_enabled:
            with self._debug_stage("encoder"):
                prefix_hidden, candidate_hidden, cls_hidden, _, _ = run_split_encoder(
                    self.encoder,
                    encoded,
                )
                hidden = torch.cat(
                    (
                        prefix_hidden,
                        candidate_hidden,
                        cls_hidden,
                    ),
                    dim=1,
                )
        else:
            with self._debug_stage("encoder"):
                with torch.no_grad():
                    hidden, self._kv_cache = encode_with_kv_cache(
                        self.encoder,
                        encoded,
                        self._kv_cache,
                    )
            # cached hidden 只包含候选块和 CLS；候选位置需要换算为 suffix 局部索引。
            prefix_length = encoded["scene_length"] + encoded["history_length"]
            cached_candidate_positions = encoded["candidate_positions"] - prefix_length
        with self._debug_stage("scorer"):
            logits = self.score_hidden(
                encoded,
                hidden,
                batch,
                candidate_positions=cached_candidate_positions,
            )
        output: dict[str, torch.Tensor] = {"logits": logits}

        if "label_index" not in batch:
            return output

        with self._debug_stage("loss"):
            label_index = batch["label_index"]
            loss = nn.functional.cross_entropy(logits, label_index)
        predictions = logits.argmax(dim=-1)
        top_k = min(3, self.data_spec.num_candidates)
        top_indices = logits.topk(top_k, dim=-1).indices
        output.update(
            {
                "loss": loss,
                "top1_accuracy": (predictions == label_index).float().mean(),
                "top3_accuracy": (
                    (top_indices == label_index.unsqueeze(-1)).any(dim=-1).float().mean()
                ),
            }
        )
        return output

    @torch.no_grad()
    def predict(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.forward(batch)
        return output["logits"].argmax(dim=-1), output["logits"]

    @torch.no_grad()
    def trace(self, batch: dict[str, torch.Tensor]) -> ModelTrace:
        """返回 split attention 的逐层 hidden 和逐 head attention。"""
        encoded = self.input_encoder(batch)
        return trace_encoder(self.encoder, encoded)

    @torch.no_grad()
    def encode_with_attention(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, list[torch.Tensor]]:
        """返回编码结果、最终 hidden 和逐 head attention。"""
        trace = self.trace(batch)
        return trace.encoded, trace.hidden, list(trace.attentions)

    def score_hidden(
        self,
        encoded: dict[str, torch.Tensor],
        hidden: torch.Tensor,
        batch: dict[str, torch.Tensor],
        *,
        candidate_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """从指定 hidden 计算 logits，并应用重复动作策略。"""
        positions = (
            encoded["candidate_positions"]
            if candidate_positions is None
            else candidate_positions
        )
        candidate_hidden = hidden[:, positions, :]
        logits = self.scorer(
            cls_hidden=hidden[:, -1, :],
            candidate_hidden=candidate_hidden,
        )
        return apply_repetition_penalty(logits, batch, self.repetition)

    @staticmethod
    def checkpoint_model_config(checkpoint: dict[str, object]) -> ModelConfig:
        """从 checkpoint 读取模型结构配置。"""
        payload = checkpoint.get("model_config")
        if not isinstance(payload, dict):
            raise ValueError("checkpoint missing model_config")
        if payload.get("scorer_use_raw_projection") is True:
            raise ValueError(
                "checkpoint enables removed scorer_use_raw_projection; "
                "retrain it with Transformer-only candidate scoring"
            )
        state_dict = checkpoint.get("model_state_dict")
        if isinstance(state_dict, Mapping) and any(
            str(key).startswith("scorer.network.") for key in state_dict
        ):
            raise ValueError(
                "checkpoint uses the removed candidate scorer layout; "
                "retrain it with the activation-configured candidate scorer"
            )
        if payload.get("scorer_use_candidate_hidden") is False:
            raise ValueError(
                "checkpoint uses removed scorer_use_candidate_hidden=false; "
                "retrain it with fixed Transformer candidate scoring"
            )
        if "max_sequence_length" in payload:
            raise ValueError(
                "checkpoint contains removed model.max_sequence_length; "
                "retrain it with block capacities and derived physical context length"
            )
        if "history_capacity" not in payload:
            raise ValueError("checkpoint missing model.history_capacity")
        if payload.get("position_id_semantics") == "candidate_block_shared":
            raise ValueError(
                "checkpoint uses the removed candidate-shared RoPE architecture; "
                "retrain with the fixed sequential RoPE model architecture"
            )
        defaults = asdict(ModelConfig())
        config_payload = {key: payload[key] for key in defaults if key in payload}
        if "num_kv_heads" not in config_payload:
            # 旧 RoPE checkpoint 没有该字段，旧结构仍是每个 Q 头各自一组 K/V。
            config_payload["num_kv_heads"] = int(
                config_payload.get("n_heads", ModelConfig.n_heads)
            )
        config = ModelConfig(**config_payload)
        return config
