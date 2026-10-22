"""训练公共层归一化器。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from common.pytorch.dependencies import import_torch

from .normalization import (
    FIELD_UNITS,
    NormalizerConfig,
    _fp32_limits,
    load_normalizer_config,
)

# 4：字段单位统一声明，分别保存相对时间与绝对时长尺度，并冻结技能状态的注册充能上限。
NORMALIZER_CONTRACT_VERSION = 4


def _normalize_values(values, divisor, *, lower=None, upper=None):
    """所有特征共用 FP32 截断与除法，不将已损失精度的原料静默转型。"""
    torch = import_torch()
    if values.dtype != torch.float32 or (
        isinstance(divisor, torch.Tensor) and divisor.dtype != torch.float32
    ):
        raise ValueError("normalization values and divisors must be FP32")
    if lower is not None or upper is not None:
        values = values.clamp(min=lower, max=upper)
    return values / divisor


class Normalizer:
    """基于 feature key 的状态向量归一化器。"""

    def __init__(
        self,
        config: NormalizerConfig | None = None,
        *,
        config_path: Path | None = None,
    ):
        if config is None:
            config = load_normalizer_config(config_path)
        self._config = config
        self._rules: dict[str, list[_FieldRule]] = {}
        self._resource_limits: dict[str, float] = {}
        self._status_limits: dict[str, float] = {}
        self._skill_charge_limits: dict[str, float] = {}
        self._resource_time_fields: tuple[str, ...] = ()
        self._configured_job_tag: str | None = None

    def register_resource_limits(self, resource_limits: dict[str, float]) -> None:
        """注册职业量谱上限，供 state/skill 两条输入链路共同使用。"""
        fp32 = _fp32_limits()
        normalized_limits: dict[str, float] = {}
        for key, raw_value in resource_limits.items():
            value = float(raw_value)
            if not fp32.tiny <= value <= fp32.max:
                raise ValueError(
                    f"resource {key!r}: normalization max must be within the FP32 normal positive range, got {raw_value!r}"
                )
            normalized_limits[str(key)] = value
        self._resource_limits = normalized_limits
        if self._rules:
            registered_features = {
                group_key: [rule.feature_name for rule in rules]
                for group_key, rules in self._rules.items()
            }
            for group_key, feature_keys in registered_features.items():
                self.register_feature_keys(group_key, feature_keys)

    def configure_job_resources(self, job_tag: str) -> None:
        """缓存量谱单位、Buff 与技能层数上限，公共尺度保持 YAML 配置值。"""
        normalized_job_tag = str(job_tag)
        if self._configured_job_tag == normalized_job_tag:
            return

        from common.game.config import load_project_config

        from ..contracts.policy_actions import load_policy_actions

        project_config = load_project_config(job_tag=normalized_job_tag)
        fp32 = _fp32_limits()
        status_limits = {
            f"{source}.{status_key}": float(status.max_stacks)
            for source, statuses in (
                ("system", project_config.system.statuses),
                ("job", project_config.job.statuses),
            )
            for status_key, status in statuses.items()
        }
        if any(not fp32.tiny <= value <= fp32.max for value in status_limits.values()):
            raise ValueError("status max_stacks must be within the FP32 normal positive range")

        self._resource_time_fields = project_config.job.resource_time_fields
        self._status_limits = status_limits
        self._skill_charge_limits = {
            skill.key: float(skill.charges)
            for skill in (*project_config.system.skills, *project_config.job.skills)
            if skill.enabled
        }
        self._skill_charge_limits.update({action.key: 1.0 for action in load_policy_actions()})
        if any(not 1 <= value <= fp32.max or not value.is_integer()
               for value in self._skill_charge_limits.values()):
            raise ValueError("skill charges must be positive integers within the FP32 normal range")
        self.register_resource_limits(project_config.job.resource_limits)
        self._configured_job_tag = normalized_job_tag

    @property
    def configured_job_tag(self) -> str | None:
        """返回当前归一化上限对应的职业标签。"""
        return self._configured_job_tag

    def ensure_job_resources(self, job_tag: str) -> None:
        """确保职业上限已配置；已从模型契约恢复时不重新读取 YAML。"""
        normalized_job_tag = str(job_tag).strip()
        if not normalized_job_tag:
            raise ValueError("job_tag must not be empty")
        if self._configured_job_tag is None:
            self.configure_job_resources(normalized_job_tag)
            return
        if self._configured_job_tag != normalized_job_tag:
            raise ValueError(
                "normalizer job_tag mismatch: "
                f"configured={self._configured_job_tag!r} requested={normalized_job_tag!r}"
            )

    @property
    def cache_signature(self) -> dict[str, object]:
        """返回会影响 compiled cache 数值的归一化签名。"""
        return {
            "version": NORMALIZER_CONTRACT_VERSION,
            "config": asdict(self._config),
            "resource_limits": dict(sorted(self._resource_limits.items())),
            "status_limits": dict(sorted(self._status_limits.items())),
            "skill_charge_limits": dict(sorted(self._skill_charge_limits.items())),
            "resource_time_fields": sorted(self._resource_time_fields),
        }

    @property
    def normalization_contract(self) -> dict[str, object]:
        """返回可随模型 checkpoint 保存的完整归一化契约。"""
        return {
            **self.cache_signature,
            "job_tag": self._configured_job_tag,
        }

    @classmethod
    def from_contract(cls, contract: Mapping[str, object]) -> Normalizer:
        """从 checkpoint 中的归一化契约恢复，不访问项目 YAML。"""
        if not isinstance(contract, Mapping):
            raise TypeError("normalizer contract must be a mapping")
        version = contract.get("version")
        if int(version) != NORMALIZER_CONTRACT_VERSION:
            raise ValueError(
                "unsupported normalizer contract version: "
                f"{version!r} != {NORMALIZER_CONTRACT_VERSION}"
            )

        config_payload = contract.get("config")
        if not isinstance(config_payload, Mapping):
            raise TypeError("normalizer contract config must be a mapping")
        required_config_keys = tuple(field.name for field in fields(NormalizerConfig))
        missing_config_keys = [
            key for key in required_config_keys if key not in config_payload
        ]
        if missing_config_keys:
            raise ValueError(
                "normalizer contract config is missing: "
                + ", ".join(missing_config_keys)
            )
        extra = config_payload.keys() - set(required_config_keys)
        if extra:
            raise ValueError("normalizer contract config has removed or unknown fields: " + ", ".join(sorted(extra)))
        normalizer_config = NormalizerConfig(**config_payload)
        normalizer = cls(config=normalizer_config)
        normalizer.register_resource_limits(
            _contract_limits(contract, "resource_limits")
        )
        normalizer._status_limits = _contract_limits(contract, "status_limits")
        normalizer._skill_charge_limits = _contract_limits(contract, "skill_charge_limits")
        if any(not value.is_integer() for value in normalizer._skill_charge_limits.values()):
            raise ValueError("normalizer contract skill_charge_limits must contain positive integers")
        time_fields = contract.get("resource_time_fields")
        if (not isinstance(time_fields, (list, tuple))
                or any(not isinstance(key, str) or key not in normalizer._resource_limits for key in time_fields)
                or len(set(time_fields)) != len(time_fields)):
            raise ValueError("normalizer contract resource_time_fields must list unique registered resources")
        normalizer._resource_time_fields = tuple(time_fields)
        job_tag = contract.get("job_tag")
        if job_tag is not None:
            normalized_job_tag = str(job_tag).strip()
            if not normalized_job_tag:
                raise ValueError("normalizer contract job_tag must not be empty")
            normalizer._configured_job_tag = normalized_job_tag
        return normalizer

    def register_feature_keys(self, group_key: str, feature_keys: list[str] | tuple[str, ...]) -> None:
        """注册一个状态分组的 feature key 列表。"""
        self._rules[group_key] = [self._field_rule(name) for name in feature_keys]

    def _field_rule(self, feature_name: str) -> _FieldRule:
        """把字段声明或已注册单位解析成同一份缩放常量。"""
        raw_name = _strip_prefix(feature_name)
        leaf_name = raw_name.rsplit(".", 1)[-1]
        if leaf_name in {"cumulative_potency", "cumulative_dot_potency"}:
            raise ValueError("cumulative potency must not enter model features; rebuild old input artifacts")
        maximum = None
        if leaf_name in self._resource_time_fields:
            unit = "seconds"
        elif leaf_name in self._resource_limits:
            unit, maximum = "count", self._resource_limits[leaf_name]
        elif leaf_name == "stacks" and raw_name.rsplit(".", 1)[0] in self._status_limits:
            unit, maximum = "count", self._status_limits[raw_name.rsplit(".", 1)[0]]
        else:
            unit = FIELD_UNITS.get(raw_name, FIELD_UNITS.get(leaf_name, "identity"))
        divisor = {
            "mp": self._config.mp_max,
            "mp_cost": self._config.mp_max,
            "seconds": self._config.duration_scale,
            "time_anchor": self._config.time_scale,
            "potency": self._config.current_potency_max,
            "target_count": self._config.target_count_max,
            "count": maximum,
        }.get(unit, 1.0)
        bounded = unit in {"mp_cost", "seconds", "potency", "target_count", "count"}
        return _FieldRule(
            feature_name, unit, float(divisor),
            0.0 if bounded else -math.inf,
            float(divisor) if bounded else math.inf,
        )

    def register_schema(self, schema) -> None:
        """从训练 schema 注册全部状态分组。"""
        for group in schema.state_groups:
            if group.encoding == "anchored_delta":
                self.register_feature_keys(group.group_key, list(group.feature_keys))

    def state_encoding_metadata(self, schema) -> StateEncodingMetadata:
        """从已注册规则生成绝对状态与相对时间元数据，不重复推断字段。"""
        self.register_schema(schema)
        rules = [rule for group in schema.state_groups if group.encoding == "anchored_delta"
                 for rule in self._rules[group.group_key]]
        return StateEncodingMetadata(
            tuple(rule.feature_name for rule in rules),
            tuple(rule.divisor for rule in rules),
            tuple(rule.absolute_lower for rule in rules),
            tuple(rule.absolute_upper for rule in rules),
            tuple(rule.unit == "time_anchor" for rule in rules),
        )

    def skill_state_encoding_metadata(self, layout) -> StateEncodingMetadata:
        """按冻结技能表生成绝对值缩放；上限仅来自保存的注册元数据。"""
        divisors, lower, upper = [], [], []
        for feature_key in layout.skill_state_feature_keys:
            action_key, field = _strip_prefix(feature_key).rsplit(".", 1)
            if action_key not in self._skill_charge_limits:
                raise ValueError(f"normalizer contract missing skill charge limit: {action_key}")
            if field == "current_charges":
                divisor = self._skill_charge_limits[action_key]
            elif field == "available":
                divisor = 1.0
            elif field == "next_cooldown_seconds":
                divisor = self.duration_scale
            else:
                raise ValueError(f"unknown registered skill state field: {field}")
            divisors.append(divisor)
            lower.append(0.0)
            upper.append(divisor)
        return StateEncodingMetadata(
            layout.skill_state_feature_keys, tuple(divisors), tuple(lower), tuple(upper),
            (False,) * layout.skill_state_dim,
        )

    @property
    def time_scale(self) -> float:
        """返回战斗时间与场景端点使用的相对时间尺度。"""
        return float(self._config.time_scale)

    @property
    def duration_scale(self) -> float:
        """返回冷却、剩余时间和场景持续时长使用的绝对时长尺度。"""
        return float(self._config.duration_scale)

    @property
    def target_count_max(self) -> float:
        return float(self._config.target_count_max)

    def normalize_skill_features(self, tensor, feature_names):
        """按 skill 字段名归一化技能数值特征。"""
        removed = {"time_seconds", "available_charges", "next_cooldown_seconds"}
        if any(str(name).rsplit(".", 1)[-1] in removed for name in feature_names):
            raise ValueError("removed skill time/charge features; rebuild old input artifacts")
        result = tensor.clone()
        # 分母必须来自尚未归一化的冻结记录，不能受字段排列顺序影响。
        names = tuple(str(name) for name in feature_names)
        if tensor.shape[-1] != len(names):
            raise ValueError("skill feature width must match feature names")
        charge_limits = None
        if "max_charges" in names:
            charge_limits = tensor[..., names.index("max_charges") : names.index("max_charges") + 1]
            if not bool((charge_limits.isfinite() & (charge_limits > 0)).all()):
                raise ValueError("skill max_charges must be finite and positive")
        if tensor.dtype != import_torch().float32:
            raise ValueError("skill normalization values must be FP32")
        for index, feature_name in enumerate(feature_names):
            if feature_name == "max_charges":
                result[..., index : index + 1] = _normalize_values(
                    tensor[..., index : index + 1], charge_limits,
                )
                continue
            rule = self._field_rule(str(feature_name))
            result[..., index : index + 1] = _normalize_values(
                tensor[..., index : index + 1], rule.divisor,
                lower=rule.absolute_lower, upper=rule.absolute_upper,
            )
        result[result.isnan()] = 0.0
        return result


def _strip_prefix(name: str) -> str:
    for prefix in ("previous_action_after.", "request_state."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name
@dataclass(frozen=True)
class StateEncodingMetadata:
    """按完整状态向量字段排列的固定归一化常量。"""

    feature_keys: tuple[str, ...]
    divisors: tuple[float, ...]
    absolute_lower_bounds: tuple[float, ...]
    absolute_upper_bounds: tuple[float, ...]
    time_anchors: tuple[bool, ...]


@dataclass(frozen=True)
class _FieldRule:
    feature_name: str
    unit: str
    divisor: float
    absolute_lower: float
    absolute_upper: float


def _contract_limits(contract: Mapping[str, object], key: str) -> dict[str, float]:
    payload = contract.get(key)
    if not isinstance(payload, Mapping):
        raise TypeError(f"normalizer contract {key} must be a mapping")
    fp32 = _fp32_limits()
    limits: dict[str, float] = {}
    for raw_name, raw_value in payload.items():
        name = str(raw_name)
        value = float(raw_value)
        if not name or not fp32.tiny <= value <= fp32.max:
            raise ValueError(
                f"normalizer contract {key} contains invalid FP32 limit: {raw_name!r}={raw_value!r}"
            )
        limits[name] = value
    return limits
