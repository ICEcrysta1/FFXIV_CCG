"""模型 checkpoint 的自描述输入契约。"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass

from .normalizer import Normalizer
from .schema import TrainingSchema
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
INPUT_CONTRACT_VERSION = 13

# 描述固定的输入结构，不作为可调运行参数；d_model 仍由保存的 model_config 提供。
# 数据 bank 的字段与时间语义由 schema 与转换版本负责，不把读取窗口加入 cache 身份。
TOKEN_ENCODING_CONTRACT = {
    "skill": "LayerNorm(E[id] + Linear(skill_features))",
    "state": "LayerNorm(Linear(state_values) + Linear(null_mask, bias=False))",
    "scene": "LayerNorm(Linear_by_scene_type(scene_values))",
    "role_ids": {"scene": 0, "state": 1, "skill": 2},
    "current_state_encoder": "shared_with_history_state",
    "state_snapshots": ["previous_action_after", "request_state"],
    "history_state_frozen_at": "request",
    "output_projection": "hidden @ E[action_to_vocab_id].T",
    "token_order": "scene, (skill_i, state_i)*H, current_state",
    "history_capacity_unit": "actions",
    "history_tokens_per_action": 2,
}


@dataclass(frozen=True)
class ModelInputContract:
    """描述模型输入字段、schema 和归一化规则的不可变契约。"""

    job_tag: str
    data_spec: dict[str, object]
    schema: TrainingSchema
    normalizer_contract: dict[str, object]

    def __post_init__(self) -> None:
        DataSpec.from_dict(self.data_spec)
        if "time_seconds" in self.schema.skill_history_fields:
            raise ValueError("removed skill time_seconds field; rebuild model input")

    @classmethod
    def from_training(cls, *, data_spec, schema: TrainingSchema, normalizer: Normalizer):
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
            raise ValueError("input contract token_encoding does not match independent skill/state tokens")
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
        }

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
