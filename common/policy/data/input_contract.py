"""模型 checkpoint 的自描述输入契约。"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

from .normalizer import Normalizer
from .schema import TrainingSchema
from .skill_vocab import SkillVocab
from .spec import DataSpec


# 输入或执行语义变化时必须升级，拒绝旧 checkpoint 静默复用。
# 5：黑魔模型移除 retrace、manaward、surecast 候选并更新模型结构，候选布局与
#    输入张量形状均已变化，旧 checkpoint 的输入分布不可复用。
# 6：技能和状态输入移除累计 GCD 索引；状态输入同时移除精确战斗剩余时间
#    及四个冗余 GCD 窗口字段，旧 checkpoint 必须重新训练。
# 7：状态输入移除调用方调度窗口与所有 GCD 单位的时间字段，旧权重不兼容。
# 8：状态输入移除资源 consumed、weave 计数/上限和黑魔残留辅助 Buff。
# 9：移除 CLS token，评分器直接读取候选 hidden，旧模型权重不兼容。
# 10：固定动作输出词表与显式当前状态取代候选输入；旧模型必须重新训练。
# 11：技能和状态拆为独立 d_model 维 token，共享技能词表直接输出；旧融合权重不兼容。
# 12：状态两段改为上一动作后与当前请求快照，历史输入在请求时冻结。
# 13：技能数值特征完全移除绝对时间，状态时间与其余技能字段口径保持。
# 14：历史顺序改为状态在技能之前，因果可见范围变化，旧顺序权重不兼容。
# 15：保存完整输入技能词表；旧 checkpoint 缺少 embedding 行的原始技能身份。
# 16：取消技能、状态、场景的独立 LayerNorm，content 与 role 相加后统一执行
#     一次无参数 RMSNorm；字段、历史布局和状态机执行语义保持不变。
# 17：主干 attention/FFN 子层与最终输出改用无参数 RMSNorm，固定保存主干
#     归一化位置；旧 LayerNorm 和带可学习尺度的 RMSNorm 权重均不兼容。
# 18：普通 Transformer 每层先以可学习 r/a 混合当前 hidden 和初始输入 x0，
#     Full AttnRes 仍使用独立深度残差路径；旧普通残差权重必须重新训练。
# 19：RoPE 后对每个 Q/K head 执行无参数 RMSNorm，并使用 checkpoint 保存的共同尺度；
#     旧的未归一化 Q/K 权重不得静默套用新注意力算法。
# 20：独立无 bias 动作输出头取代共享技能 embedding 点积，并在 FP32 中执行
#     checkpoint 保存尺度的 softcap；旧共享输出权重不得静默套用新读出算法。
# 21：状态使用窗口 ABS 锚点与原始全字段 DELTA，场景时间裁剪后差分，显式保存
#     字段 ABS 重置标识；旧归一化输入和旧 checkpoint 必须重新构建。
INPUT_CONTRACT_VERSION = 21

RESIDUAL_MIX_CONFIG_FIELDS = (
    "residual_mix_r_start", "residual_mix_r_end",
    "residual_mix_a_start", "residual_mix_a_end",
)

BACKBONE_RESIDUAL_CONTRACT = {
    "selection": "model_config.full_attention_residuals",
    "ordinary": {
        "type": "learned_residual_mix",
        "position": "before_each_transformer_layer",
        "formula": "mixed = r[layer] * hidden + a[layer] * x0",
        "attention_and_skip_input": "mixed",
        "x0": "input_encoder.output_after_token_rmsnorm",
        "x0_detached": False,
        "parameters": ["encoder.residual_mix.r", "encoder.residual_mix.a"],
        "parameter_shape": "[model_config.n_layers]",
        "initialization": {
            "r": "linspace(model_config.residual_mix_r_start, model_config.residual_mix_r_end)",
            "a": "linspace(model_config.residual_mix_a_start, model_config.residual_mix_a_end)",
        },
    },
    "full_attention": {
        "type": "full_attention_residual",
        "depth_sources": "initial_tokens_and_previous_sublayer_outputs",
        "learned_residual_mix": False,
    },
}


def residual_composition_contract(model_config: Mapping[str, object]) -> dict[str, object]:
    """从保存的架构字段选择残差机制，禁止给旧 checkpoint 补初始化默认值。"""
    missing = [key for key in RESIDUAL_MIX_CONFIG_FIELDS if key not in model_config]
    if missing:
        raise ValueError("model_config missing residual mix initialization fields: " + ", ".join(missing))
    values = {}
    for key in RESIDUAL_MIX_CONFIG_FIELDS:
        value = model_config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"model_config.{key} must be a finite number")
        values[key] = float(value)
    full_attention = model_config.get("full_attention_residuals")
    if not isinstance(full_attention, bool):
        # 与其他 checkpoint 契约格式错误统一使用 ValueError。
        raise ValueError("model_config.full_attention_residuals must be a saved boolean")  # noqa: TRY004
    if full_attention:
        return deepcopy(BACKBONE_RESIDUAL_CONTRACT["full_attention"])
    descriptor = deepcopy(BACKBONE_RESIDUAL_CONTRACT["ordinary"])
    descriptor["initialization"] = {
        "r": {"start": values["residual_mix_r_start"], "end": values["residual_mix_r_end"]},
        "a": {"start": values["residual_mix_a_start"], "end": values["residual_mix_a_end"]},
        "method": "linspace",
    }
    return descriptor

# 描述固定的输入结构，不作为可调运行参数；d_model 仍由保存的 model_config 提供。
# 数据 bank 的字段与时间语义由 schema 与转换版本负责，不把读取窗口加入 cache 身份。
TOKEN_ENCODING_CONTRACT = {
    "skill": "E[id] + Linear(skill_features)",
    "state": "Linear(state_values) + Linear(null_mask, bias=False) + Linear(state_reset_mask, bias=False)",
    "scene": "Linear_by_scene_type(scene_values)",
    "token_normalization": {
        "type": "RMSNorm",
        "position": "after_content_plus_role",
        "eps": 1e-5,
        "elementwise_affine": False,
        "applications": 1,
    },
    "backbone_normalization": {
        "type": "RMSNorm",
        "positions": [
            "encoder.layers[*].norm1",
            "encoder.layers[*].norm2",
            "encoder.norm",
        ],
        "eps": 1e-5,
        "elementwise_affine": False,
    },
    "backbone_residual": BACKBONE_RESIDUAL_CONTRACT,
    "attention_qk_normalization": {
        "type": "RMSNorm",
        "position": "after_RoPE_before_attention",
        "normalized_shape": "head_dim",
        "eps": None,
        "elementwise_affine": False,
        "scale": "model_config.qk_norm_scale",
        "applies_to": ["query", "key"],
        "key_cache": "normalize_new_keys_once_before_append",
    },
    "role_ids": {"scene": 0, "state": 1, "skill": 2},
    "current_state_encoder": "shared_with_history_state",
    "state_snapshots": ["previous_action_after", "request_state"],
    "history_state_frozen_at": "request",
    "context_encoding": {
        "compute_dtype": "float32_before_activation_cast",
        "state": "first_visible_ABS_then_raw_numeric_DELTA",
        "state_abs_time_origin": "first_visible_request_state.time_seconds",
        "boolean_delta": "minus_one_zero_plus_one",
        "unknown_recovery": "per_field_ABS_reset_mask",
        "delta_linear": "raw_delta / saved_field_divisor_without_clipping",
        "delta_logarithmic": "sign(raw_delta) * log1p(abs(raw_delta))",
        "reset_projection": "zero_initialized_input_encoder.state_reset_proj",
        "scene": "clip_at_state_anchor_then_stable_start_end_type_order",
        "scene_time": "first_anchor_offset_then_separate_start_end_DELTA",
        "scene_duration": "own_clipped_duration",
        "time_scale": "model_config.time_delta_scale",
        "history_window": "block_reset_from_full_history_cursor",
        "overflow_keep": "model_config.history_reset_keep",
        "time_clipping": False,
    },
    "output_projection": {
        "type": "Linear",
        "parameter": "output_head.weight",
        "weight_shape": "[data_spec.num_actions, model_config.d_model]",
        "bias": False,
        "weight_tying": False,
        "initialization": "copy_skill_embed_rows_in_action_to_vocab_id_order",
    },
    "logit_softcap": {
        "formula": "cap * tanh(output_head(hidden).float() / cap)",
        "cap": "model_config.logit_softcap",
        "compute_dtype": "float32",
        "position": "after_output_projection_before_repetition_penalty_and_legal_mask",
    },
    "token_order": "scene, (state_i, skill_i)*H, current_state",
    "history_capacity_unit": "actions",
    "history_tokens_per_action": 2,
}


@dataclass(frozen=True)
class ModelInputContract:
    """描述输入字段、完整技能词表、schema 和归一化规则的不可变契约。"""

    job_tag: str
    data_spec: dict[str, object]
    schema: TrainingSchema
    normalizer_contract: dict[str, object]
    skill_vocab_entries: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        data_spec = DataSpec.from_dict(self.data_spec)
        vocab = self.create_skill_vocab()
        if any(row >= vocab.size() for row in data_spec.action_to_vocab_id):
            raise ValueError("input contract output actions exceed the saved skill vocab")
        if "time_seconds" in self.schema.skill_history_fields:
            raise ValueError("removed skill time_seconds field; rebuild model input")

    @classmethod
    def from_training(cls, *, data_spec, schema: TrainingSchema, normalizer: Normalizer, skill_vocab: SkillVocab):
        if normalizer.configured_job_tag != data_spec.job_tag:
            raise ValueError(
                "normalizer job_tag must match training data spec: "
                f"{normalizer.configured_job_tag!r} != {data_spec.job_tag!r}"
            )
        return cls(
            job_tag=str(data_spec.job_tag),
            data_spec=asdict(data_spec),
            schema=schema,
            normalizer_contract=normalizer.normalization_contract,
            skill_vocab_entries=tuple(skill_vocab),
        )

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping[str, object]) -> "ModelInputContract":
        if not isinstance(checkpoint, Mapping):
            raise ValueError("checkpoint must be a mapping")
        payload = checkpoint.get("input_contract")
        if not isinstance(payload, Mapping):
            raise ValueError("checkpoint missing input_contract")
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ModelInputContract":
        if not isinstance(payload, Mapping):
            raise ValueError("input contract must be a mapping")
        version = payload.get("version")
        if int(version) != INPUT_CONTRACT_VERSION:
            raise ValueError(
                "unsupported input contract version: "
                f"{version!r} != {INPUT_CONTRACT_VERSION}"
            )
        if payload.get("token_encoding") != TOKEN_ENCODING_CONTRACT:
            raise ValueError("input contract token_encoding does not match the fixed token and backbone encoding")
        if not isinstance(payload.get("skill_vocab"), Mapping):
            raise ValueError("input contract missing complete skill_vocab; restore it from the original training metadata")
        skill_vocab = SkillVocab.from_dict(payload["skill_vocab"])
        job_tag = str(payload.get("job_tag", "")).strip()
        if not job_tag:
            raise ValueError("input contract job_tag must not be empty")
        data_spec = payload.get("data_spec")
        if not isinstance(data_spec, Mapping):
            raise ValueError("input contract data_spec must be a mapping")
        normalizer_contract = payload.get("normalizer")
        if not isinstance(normalizer_contract, Mapping):
            raise ValueError("input contract normalizer must be a mapping")
        normalizer = Normalizer.from_contract(normalizer_contract)
        if normalizer.configured_job_tag != job_tag:
            raise ValueError(
                "input contract normalizer job_tag mismatch: "
                f"{normalizer.configured_job_tag!r} != {job_tag!r}"
            )
        normalized_data_spec = dict(data_spec)
        try:
            DataSpec.from_dict(normalized_data_spec)
        except (KeyError, TypeError) as exc:
            raise ValueError("input contract data_spec is missing the fixed action output mapping") from exc
        if str(normalized_data_spec.get("job_tag", "")) != job_tag:
            raise ValueError("input contract data_spec job_tag mismatch")
        return cls(
            job_tag=job_tag,
            data_spec=normalized_data_spec,
            schema=TrainingSchema.from_dict(payload.get("schema")),
            normalizer_contract=dict(normalizer_contract),
            skill_vocab_entries=tuple(skill_vocab),
        )

    def to_dict(self) -> dict[str, object]:
        """转成可直接写入 torch checkpoint 的普通 mapping。"""
        return {
            "version": INPUT_CONTRACT_VERSION,
            "token_encoding": deepcopy(TOKEN_ENCODING_CONTRACT),
            "job_tag": self.job_tag,
            "data_spec": dict(self.data_spec),
            "schema": asdict(self.schema),
            "normalizer": dict(self.normalizer_contract),
            "skill_vocab": self.create_skill_vocab().to_dict(),
        }

    def create_skill_vocab(self) -> SkillVocab:
        """恢复训练时的 embedding 行身份，不读取本机 YAML 或 policy 配置。"""
        return SkillVocab.from_entries(self.skill_vocab_entries)

    def assert_matches_embedding(self, embedding_vocab_size: int) -> None:
        expected = self.create_skill_vocab().size()
        if expected != embedding_vocab_size:
            raise ValueError(
                "checkpoint skill embedding row count differs from the saved complete skill vocab: "
                f"{embedding_vocab_size} != {expected}"
            )

    def create_normalizer(self) -> Normalizer:
        """恢复归一化器并注册 checkpoint 中的状态字段。"""
        normalizer = Normalizer.from_contract(self.normalizer_contract)
        if normalizer.configured_job_tag != self.job_tag:
            raise ValueError("input contract normalizer job_tag mismatch")
        normalizer.register_schema(self.schema)
        return normalizer

    def assert_matches_data_spec(self, data_spec) -> None:
        expected = dict(self.data_spec)
        actual = asdict(data_spec)
        if expected != actual:
            raise ValueError("checkpoint input contract and data_spec mismatch")
