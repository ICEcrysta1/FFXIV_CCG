"""与职业无关、使用独立动作输出头的因果策略模型。"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext

import torch
import torch.nn as nn

from ..config import ModelConfig
from .attention_residual import FullAttentionResidual
from .input_encoder import CausalInputEncoder
from .kv_cache import TransformerKVCache, encode_with_kv_cache
from .position_encoding import RotaryPositionEncoding
from .repetition import RepetitionConfig, apply_repetition_penalty
from .residual_mix import LearnedResidualMix
from .causal_encoder import run_causal_encoder
from ..data.input_contract import ModelInputContract
from ..data.spec import DataSpec
from .trace import ModelTrace, TraceableTransformerEncoderLayer, trace_encoder


class _CopiedActionHead(nn.Linear):
    """复制动作 embedding 行作为独立权重，不消费额外初始化随机数。"""

    def __init__(self, initial_weight: torch.Tensor):
        super().__init__(
            initial_weight.shape[1], initial_weight.shape[0], bias=False,
            device=initial_weight.device, dtype=initial_weight.dtype,
        )
        with torch.no_grad():
            self.weight.copy_(initial_weight)

    def reset_parameters(self) -> None:
        # 构造器随后完整复制来源权重，省去会被覆盖的随机初始化。
        pass


class CausalPolicyModel(nn.Module):
    """从最新状态预测固定动作词表，输出头与输入技能 embedding 独立训练。"""

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
        self.input_encoder = CausalInputEncoder(data_spec, config, vocab_size)
        if any(index <= 0 or index >= vocab_size for index in data_spec.action_to_vocab_id):
            raise ValueError("output actions must map to registered non-padding embedding rows")
        self.register_buffer(
            "action_to_vocab_id",
            torch.tensor(data_spec.action_to_vocab_id, dtype=torch.long),
            persistent=False,
        )
        encoder_layer = TraceableTransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            num_kv_heads=config.num_kv_heads,
            qk_norm_scale=config.qk_norm_scale,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation=config.transformer_activation,
            batch_first=True,
            norm_first=config.transformer_norm_first,
        )
        # PreNorm 支路与最终输出统一使用无参数 RMSNorm，不对残差流施加幅度上限。
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            config.n_layers,
            norm=nn.RMSNorm(config.d_model, eps=1e-5, elementwise_affine=False),
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
        else:
            # 普通残差独立学习逐层尺度和初始 token 注入，不与 Full AttnRes 叠加。
            self.encoder.residual_mix = LearnedResidualMix(
                config.n_layers,
                r_start=config.residual_mix_r_start,
                r_end=config.residual_mix_r_end,
                a_start=config.residual_mix_a_start,
                a_end=config.residual_mix_a_end,
            )
        self._init_weights()
        self.output_head = _CopiedActionHead(
            self.input_encoder.skill_embed.weight[self.action_to_vocab_id],
        )
        self._kv_cache_enabled = False
        self._kv_cache: TransformerKVCache | None = None
        self._runtime_debug = None

    def _init_weights(self) -> None:
        for name, parameter in self.named_parameters():
            if parameter.dim() > 1 and name != "input_encoder.state_reset_proj.weight":
                nn.init.xavier_uniform_(parameter)
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

    def enable_activation_checkpoint_attention(
        self,
        enabled: bool = True,
        *,
        block: bool = False,
    ) -> None:
        """开启或关闭训练时的 Attention activation checkpoint。

        ``block=True`` 时整块 attention（norm、Q/K/V 投影、SDPA、merge、out_proj）
        一起重算：反向多算一遍投影，但省下投影与 attention 输出的全部中间激活。
        """
        for layer in self.encoder.layers:
            layer.set_activation_checkpoint_attention(enabled)
            layer.set_activation_checkpoint_attention_block(block and enabled)

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
        if self.training or not self._kv_cache_enabled:
            with self._debug_stage("encoder"):
                hidden, _, _ = run_causal_encoder(
                    self.encoder,
                    encoded,
                )
                current_hidden = hidden[:, encoded["current_state_position"], :]
        else:
            with self._debug_stage("encoder"):
                with torch.no_grad():
                    hidden, self._kv_cache = encode_with_kv_cache(
                        self.encoder,
                        encoded,
                        self._kv_cache,
                    )
            current_hidden = hidden[:, 0, :]
        with self._debug_stage("lm_head"):
            logits = self._score_current_hidden(current_hidden, batch)
        output: dict[str, torch.Tensor] = {"logits": logits}

        if "label_index" not in batch:
            return output

        label_index = batch["label_index"]
        predictions = logits.argmax(dim=-1)
        top_k = min(3, self.data_spec.num_actions)
        top_indices = logits.topk(top_k, dim=-1).indices
        output.update(
            {
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
        """返回单一因果上下文的逐层 hidden 和逐 head attention。"""
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
    ) -> torch.Tensor:
        """从最新状态 hidden 计算固定动作词表 logits。"""
        return self._score_current_hidden(hidden[:, encoded["current_state_position"], :], batch)

    def compute_action_logits(self, current_hidden: torch.Tensor) -> torch.Tensor:
        """训练、回放和 ONNX 共用独立动作头与 FP32 softcap，不含宿主后处理。"""
        raw_logits = self.output_head(current_hidden)
        cap = self.config.logit_softcap
        return cap * torch.tanh(raw_logits.float() / cap)

    def _score_current_hidden(self, current_hidden, batch):
        """先执行输出头与 softcap，再执行一次既有重复惩罚。"""
        logits = self.compute_action_logits(current_hidden)
        return apply_repetition_penalty(logits, batch, self.repetition)

    @staticmethod
    def checkpoint_model_config(checkpoint: dict[str, object]) -> ModelConfig:
        """只恢复与当前无候选输入契约匹配的模型结构。"""
        ModelInputContract.from_checkpoint(checkpoint)
        payload = checkpoint.get("model_config")
        if not isinstance(payload, Mapping):
            raise ValueError("checkpoint missing model_config")
        if "history_capacity" not in payload:
            raise ValueError("checkpoint missing model.history_capacity")
        for key in ("history_reset_keep", "time_delta_scale"):
            if key not in payload:
                raise ValueError(f"checkpoint missing model.{key}; retrain")
        if "qk_norm_scale" not in payload:
            raise ValueError("checkpoint missing model.qk_norm_scale; retrain")
        if "logit_softcap" not in payload:
            raise ValueError("checkpoint missing model.logit_softcap; retrain")
        for name in ("residual_mix_r_start", "residual_mix_r_end",
                     "residual_mix_a_start", "residual_mix_a_end"):
            if name not in payload:
                raise ValueError(f"checkpoint missing model.{name}; retrain")
        return ModelConfig.from_mapping(payload)
