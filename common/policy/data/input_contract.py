"""模型 checkpoint 的自描述输入契约。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

from .normalizer import Normalizer
from .schema import TrainingSchema


# 候选合法性与执行语义变化时必须升级，拒绝旧 checkpoint 静默复用。
# 5：黑魔模型移除 retrace、manaward、surecast 候选并更新模型结构，候选布局与
#    输入张量形状均已变化，旧 checkpoint 的输入分布不可复用。
INPUT_CONTRACT_VERSION = 5


@dataclass(frozen=True)
class ModelInputContract:
    """描述模型输入字段、schema 和归一化规则的不可变契约。"""

    job_tag: str
    data_spec: dict[str, object]
    schema: TrainingSchema
    normalizer_contract: dict[str, object]

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
