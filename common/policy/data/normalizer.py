"""训练公共层归一化器。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path

from .normalization import NormalizerConfig, load_normalizer_config


# 3：统一公共尺度、按冻结上限编码充能，并移除累计威力输入。
NORMALIZER_CONTRACT_VERSION = 3


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
        self._resource_time_fields: tuple[str, ...] = ()
        self._configured_job_tag: str | None = None

    def register_resource_limits(self, resource_limits: dict[str, float]) -> None:
        """注册职业量谱上限，供 state/skill 两条输入链路共同使用。"""
        normalized_limits: dict[str, float] = {}
        for key, raw_value in resource_limits.items():
            value = float(raw_value)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"resource {key!r}: normalization max must be positive and finite, got {raw_value!r}"
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
        """缓存职业量谱单位和 Buff 层数上限，公共尺度保持 YAML 配置值。"""
        normalized_job_tag = str(job_tag)
        if self._configured_job_tag == normalized_job_tag:
            return

        from common.game.config import load_project_config

        project_config = load_project_config(job_tag=normalized_job_tag)
        status_limits = {
            f"{source}.{status_key}": float(status.max_stacks)
            for source, statuses in (
                ("system", project_config.system.statuses),
                ("job", project_config.job.statuses),
            )
            for status_key, status in statuses.items()
        }
        if any(not math.isfinite(value) or value <= 0.0 for value in status_limits.values()):
            raise ValueError("status max_stacks must be positive and finite")

        self._resource_time_fields = project_config.job.resource_time_fields
        self._status_limits = status_limits
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
    def from_contract(cls, contract: Mapping[str, object]) -> "Normalizer":
        """从 checkpoint 中的归一化契约恢复，不访问项目 YAML。"""
        if not isinstance(contract, Mapping):
            raise ValueError("normalizer contract must be a mapping")
        version = contract.get("version")
        if int(version) != NORMALIZER_CONTRACT_VERSION:
            raise ValueError(
                "unsupported normalizer contract version: "
                f"{version!r} != {NORMALIZER_CONTRACT_VERSION}"
            )

        config_payload = contract.get("config")
        if not isinstance(config_payload, Mapping):
            raise ValueError("normalizer contract config must be a mapping")
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
        rules: list[_FieldRule] = []
        for feature_name in feature_keys:
            raw_name = _strip_prefix(feature_name)
            rule_type = _infer_rule_type(
                raw_name,
                self._resource_limits,
                self._status_limits,
                self._resource_time_fields,
            )
            rules.append(
                _FieldRule(
                    feature_name=feature_name,
                    rule_type=rule_type,
                    max_value=_max_value_for_feature(
                        raw_name,
                        self._resource_limits,
                        self._status_limits,
                    ),
                )
            )
        self._rules[group_key] = rules

    def register_schema(self, schema) -> None:
        """从训练 schema 注册全部状态分组。"""
        for group in schema.state_groups:
            if group.encoding == "anchored_delta":
                self.register_feature_keys(group.group_key, list(group.feature_keys))

    def state_encoding_metadata(self, schema) -> StateEncodingMetadata:
        """从已注册规则生成向量化 ABS/有符号 DELTA 元数据，不重复推断字段。"""
        self.register_schema(schema)
        keys, divisors, lower_bounds, upper_bounds = [], [], [], []
        for group in schema.state_groups:
            if group.encoding != "anchored_delta":
                continue
            for rule in self._rules[group.group_key]:
                divisor, lower, upper = 1.0, -math.inf, math.inf
                kind = rule.rule_type
                if kind in {"divide_mp_max", "divide_max_mp"}:
                    divisor = self._config.mp_max
                elif kind == "clip_divide_seconds_max":
                    divisor, lower, upper = self._config.time_scale, 0.0, self._config.time_scale
                elif kind == "divide_time_scale":
                    divisor = self._config.time_scale
                elif kind in {"clip_divide_resource_max", "clip_divide_status_max"}:
                    if rule.max_value is None or rule.max_value <= 0:
                        raise ValueError(f"state field {rule.feature_name!r} requires a positive max_value")
                    divisor, lower, upper = rule.max_value, 0.0, rule.max_value
                elif kind == "divide_current_potency_max":
                    divisor, lower, upper = self._config.current_potency_max, 0.0, self._config.current_potency_max
                keys.append(rule.feature_name)
                divisors.append(float(divisor))
                lower_bounds.append(float(lower))
                upper_bounds.append(float(upper))
        return StateEncodingMetadata(
            tuple(keys), tuple(divisors), tuple(lower_bounds), tuple(upper_bounds),
        )

    @property
    def time_scale(self) -> float:
        """返回普通时间字段使用的公共秒数尺度。"""
        return float(self._config.time_scale)

    @property
    def target_count_max(self) -> float:
        return float(self._config.target_count_max)

    def normalize_skill_features(self, tensor, feature_names):
        """按 skill 字段名归一化技能数值特征。"""
        if any(str(name).split(".")[-1] == "time_seconds" for name in feature_names):
            raise ValueError("skill features must not include time_seconds; rebuild old input artifacts")
        result = tensor.clone()
        # 分母必须来自尚未归一化的冻结记录，不能受字段排列顺序影响。
        names = tuple(str(name) for name in feature_names)
        if tensor.shape[-1] != len(names):
            raise ValueError("skill feature width must match feature names")
        charge_limits = None
        if "available_charges" in names or "max_charges" in names:
            if "max_charges" not in names:
                raise ValueError("available_charges requires the frozen max_charges field")
            charge_limits = tensor[..., names.index("max_charges") : names.index("max_charges") + 1]
            if not bool((charge_limits.isfinite() & (charge_limits > 0)).all()):
                raise ValueError("skill max_charges must be finite and positive")
        for index, feature_name in enumerate(feature_names):
            if feature_name in {"available_charges", "max_charges"}:
                values = result[..., index : index + 1]
                if not bool((values.isfinite() & (values >= 0) & (values <= charge_limits)).all()):
                    raise ValueError("skill charges must be finite and within the frozen max_charges")
                values.div_(charge_limits)
                continue
            _apply_skill_normalize_inplace(
                result[..., index : index + 1],
                str(feature_name),
                self._config,
                resource_limits=self._resource_limits,
                resource_time_fields=self._resource_time_fields,
            )
        result[result.isnan()] = 0.0
        return result

def _apply_skill_normalize_inplace(
    values,
    feature_name: str,
    config: NormalizerConfig,
    *,
    resource_limits: dict[str, float],
    resource_time_fields: tuple[str, ...],
) -> None:
    """把 compiled cache 中的动态 skill 数值压到模型可比较的尺度。"""
    leaf_name = feature_name.split(".")[-1]
    if leaf_name in {"is_legal", "kind"}:
        return
    if leaf_name in {"cumulative_potency", "cumulative_dot_potency"}:
        raise ValueError("cumulative potency must not enter model features; rebuild old input artifacts")
    if leaf_name == "potency":
        values.clamp_(min=0.0, max=config.current_potency_max).div_(config.current_potency_max)
        return
    if leaf_name in {"actual_mp_cost", "mp_cost"}:
        values.clamp_(min=0.0, max=config.mp_max).div_(config.mp_max)
        return
    if leaf_name in resource_time_fields:
        values.clamp_(min=0.0, max=config.time_scale).div_(config.time_scale)
        return
    resource_max = resource_limits.get(leaf_name)
    if resource_max is not None:
        values.clamp_(min=0.0, max=resource_max).div_(resource_max)
        return
    if feature_name.endswith(".seconds") or leaf_name.endswith("_seconds"):
        values.clamp_(min=0.0, max=config.time_scale).div_(config.time_scale)
        return
    if leaf_name.endswith("_timer"):
        values.clamp_(min=0.0, max=config.time_scale).div_(config.time_scale)


def _infer_rule_type(
    raw_name: str,
    resource_limits: dict[str, float],
    status_limits: dict[str, float],
    resource_time_fields: tuple[str, ...],
) -> str:
    """根据字段名推断归一化规则。"""
    leaf_name = _leaf_name(raw_name)
    if leaf_name in {"cumulative_potency", "cumulative_dot_potency"}:
        raise ValueError("cumulative potency must not enter model features; rebuild old input artifacts")

    if leaf_name in _BINARY_FIELDS:
        return "keep"
    if raw_name.endswith(".active"):
        return "keep"
    if raw_name.endswith(".stacks"):
        if _status_key_from_stacks_feature(raw_name) in status_limits:
            return "clip_divide_status_max"
        return "keep"
    if leaf_name.endswith("_ready"):
        return "keep"
    if leaf_name == "mp_ratio":
        return "keep"
    if leaf_name == "mp":
        return "divide_mp_max"
    if leaf_name == "max_mp":
        return "divide_max_mp"
    if leaf_name == "time_seconds":
        return "divide_time_scale"
    if leaf_name in resource_time_fields:
        return "clip_divide_seconds_max"
    if leaf_name in resource_limits:
        return "clip_divide_resource_max"
    if leaf_name.endswith("_seconds"):
        return "clip_divide_seconds_max"
    if leaf_name == "current_potency":
        return "divide_current_potency_max"
    if leaf_name == "current_gcd_dot_potency":
        return "divide_current_potency_max"
    if leaf_name.endswith("_timer"):
        return "clip_divide_seconds_max"
    return "skip"


_BINARY_FIELDS = frozenset(
    {
        "boss_targetable",
        "is_moving",
    }
)


def _strip_prefix(name: str) -> str:
    for prefix in ("previous_action_after.", "request_state."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _leaf_name(raw_name: str) -> str:
    return raw_name.split(".")[-1]


def _status_key_from_stacks_feature(raw_name: str) -> str:
    """把 `job.triplecast.stacks` 映射为状态定义键。"""
    suffix = ".stacks"
    if not raw_name.endswith(suffix):
        return ""
    return raw_name[: -len(suffix)]


def _max_value_for_feature(
    raw_name: str,
    resource_limits: dict[str, float],
    status_limits: dict[str, float],
) -> float | None:
    if raw_name.endswith(".stacks"):
        return status_limits.get(_status_key_from_stacks_feature(raw_name))
    return resource_limits.get(_leaf_name(raw_name))


@dataclass(frozen=True)
class StateEncodingMetadata:
    """按完整状态向量字段排列的固定归一化常量。"""

    feature_keys: tuple[str, ...]
    divisors: tuple[float, ...]
    absolute_lower_bounds: tuple[float, ...]
    absolute_upper_bounds: tuple[float, ...]


@dataclass(frozen=True)
class _FieldRule:
    feature_name: str
    rule_type: str
    max_value: float | None = None


def _contract_limits(contract: Mapping[str, object], key: str) -> dict[str, float]:
    payload = contract.get(key)
    if not isinstance(payload, Mapping):
        raise ValueError(f"normalizer contract {key} must be a mapping")
    limits: dict[str, float] = {}
    for raw_name, raw_value in payload.items():
        name = str(raw_name)
        value = float(raw_value)
        if not name or not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"normalizer contract {key} contains invalid limit: {raw_name!r}={raw_value!r}"
            )
        limits[name] = value
    return limits
