"""训练公共层归一化器。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path

from common.torch_dependencies import import_torch

from .normalization import NormalizerConfig, load_normalizer_config


NORMALIZER_CONTRACT_VERSION = 1


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
        self._feature_dims: dict[str, int] = {}
        self._resource_limits: dict[str, float] = {}
        self._status_limits: dict[str, float] = {}
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
        """缓存职业配置中的量谱、Buff 层数上限和最大技能冷却。"""
        normalized_job_tag = str(job_tag)
        if self._configured_job_tag == normalized_job_tag:
            return

        from common.config import load_project_config

        project_config = load_project_config(job_tag=normalized_job_tag)
        max_cooldown = max(
            (
                float(skill.cooldown)
                for skill in (*project_config.system.skills, *project_config.job.skills)
                if skill.enabled and float(skill.cooldown) > 0.0
            ),
            default=float(self._config.remaining_seconds_max),
        )
        if not math.isfinite(max_cooldown) or max_cooldown <= 0.0:
            raise ValueError(
                f"job {normalized_job_tag!r} must define a positive skill cooldown "
                "for remaining-seconds normalization"
            )

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

        self._config = replace(self._config, remaining_seconds_max=max_cooldown)
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
        required_config_keys = tuple(asdict(NormalizerConfig()))
        missing_config_keys = [
            key for key in required_config_keys if key not in config_payload
        ]
        if missing_config_keys:
            raise ValueError(
                "normalizer contract config is missing: "
                + ", ".join(missing_config_keys)
            )
        normalizer_config = NormalizerConfig(
            mp_max=float(config_payload["mp_max"]),
            remaining_seconds_max=float(config_payload["remaining_seconds_max"]),
            fight_time_max=float(config_payload["fight_time_max"]),
            target_count_max=float(config_payload["target_count_max"]),
            remaining_gcds_max=float(config_payload["remaining_gcds_max"]),
            current_potency_max=float(config_payload["current_potency_max"]),
            cumulative_potency_mode=str(config_payload["cumulative_potency_mode"]),
        )
        normalizer = cls(config=normalizer_config)
        normalizer.register_resource_limits(
            _contract_limits(contract, "resource_limits")
        )
        normalizer._status_limits = _contract_limits(contract, "status_limits")
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
        for index, feature_name in enumerate(feature_keys):
            raw_name = _strip_prefix(feature_name)
            rule_type = _infer_rule_type(
                raw_name,
                self._resource_limits,
                self._status_limits,
                self._config.cumulative_potency_mode,
            )
            rules.append(
                _FieldRule(
                    index=index,
                    feature_name=feature_name,
                    raw_name=raw_name,
                    rule_type=rule_type,
                    max_value=_max_value_for_feature(
                        raw_name,
                        self._resource_limits,
                        self._status_limits,
                    ),
                )
            )
        self._rules[group_key] = rules
        self._feature_dims[group_key] = len(rules)

    def register_schema(self, schema) -> None:
        """从训练 schema 注册全部状态分组。"""
        for group_key, feature_keys in schema.state_group_feature_keys.items():
            self.register_feature_keys(group_key, list(feature_keys))

    @property
    def registered_groups(self) -> list[str]:
        return list(self._rules.keys())

    @property
    def fight_time_max(self) -> float:
        return float(self._config.fight_time_max)

    @property
    def remaining_seconds_max(self) -> float:
        """返回 skill/状态剩余秒数特征使用的归一化上限。"""
        return float(self._config.remaining_seconds_max)

    @property
    def target_count_max(self) -> float:
        return float(self._config.target_count_max)

    def normalize_scene_time(self, time_seconds: float) -> float:
        """把绝对 scene 时间按统一战斗时长归一化到 [0, 1]。"""
        fight_time_max = self.fight_time_max
        if fight_time_max <= 0.0:
            raise ValueError("fight_time_max must be positive")
        return max(0.0, min(float(time_seconds), fight_time_max)) / fight_time_max

    def normalize_scene_target_count(self, target_count: float) -> float:
        """按 scene token 的统一目标数量上限归一化目标数量。"""
        target_count_max = self.target_count_max
        if target_count_max <= 0.0:
            raise ValueError("target_count_max must be positive")
        return max(0.0, min(float(target_count), target_count_max)) / target_count_max

    def denormalize_scene_target_count(self, normalized_target_count: float) -> int:
        """把归一化后的 scene 目标数量恢复为状态机使用的整数。"""
        target_count_max = self.target_count_max
        if target_count_max <= 0.0:
            raise ValueError("target_count_max must be positive")
        normalized = max(0.0, min(float(normalized_target_count), 1.0))
        return max(0, int(round(normalized * target_count_max)))

    def feature_dim(self, group_key: str) -> int:
        return self._feature_dims.get(group_key, 0)

    def normalize(self, tensor, group_key: str, *, null_mask=None):
        """归一化一个状态分组的向量。null 位置会被安全置零。"""
        torch = import_torch()
        rules = self._rules.get(group_key)
        if rules is None:
            raise KeyError(f"group_key {group_key!r} not registered")

        if tensor.shape[-1] != len(rules):
            raise ValueError(
                f"tensor last dim {tensor.shape[-1]} != registered feature count {len(rules)} "
                f"for group {group_key!r}"
            )

        result = tensor.clone()
        if null_mask is None:
            null_mask = torch.zeros_like(result, dtype=torch.bool)
        else:
            null_mask = null_mask.to(dtype=torch.bool)
            if null_mask.shape != result.shape:
                raise ValueError(
                    f"null_mask shape {tuple(null_mask.shape)} != tensor shape {tuple(result.shape)}"
                )

        for rule in rules:
            if rule.rule_type in ("keep", "skip"):
                result[..., rule.index].masked_fill_(null_mask[..., rule.index], -1.0)
                continue

            values = result[..., rule.index : rule.index + 1]
            mask = null_mask[..., rule.index : rule.index + 1]
            values.masked_fill_(mask, 0.0)
            _apply_normalize_inplace(
                values,
                rule.rule_type,
                self._config,
                max_value=rule.max_value,
            )
            values.masked_fill_(mask, -1.0)

        result[torch.isnan(result)] = 0.0
        result.masked_fill_(null_mask, -1.0)
        return result

    def normalize_skill_features(self, tensor, feature_names):
        """按 skill 字段名归一化技能数值特征。"""
        result = tensor.clone()
        for index, feature_name in enumerate(feature_names):
            _apply_skill_normalize_inplace(
                result[..., index : index + 1],
                str(feature_name),
                self._config,
                resource_limits=self._resource_limits,
            )
        result[result.isnan()] = 0.0
        return result

    def normalize_scene_tokens(self, tensor, scene_types, schema):
        """按窗口类型归一化 scene 的绝对战斗时间字段。"""
        torch = import_torch()
        if tensor.ndim != 2:
            raise ValueError(f"scene tensor must be 2D, got shape {tuple(tensor.shape)}")
        if scene_types.ndim != 1 or scene_types.shape[0] != tensor.shape[0]:
            raise ValueError(
                "scene type shape must be [N] and match scene tensor rows: "
                f"{tuple(scene_types.shape)} != {tensor.shape[0]}"
            )
        scene_width = schema.scene_feature_dim()
        if tensor.shape[-1] < scene_width:
            raise ValueError(
                "scene tensor width is smaller than schema scene width: "
                f"{tensor.shape[-1]} < {scene_width}"
            )
        if self._config.fight_time_max <= 0.0:
            raise ValueError("fight_time_max must be positive")
        if self._config.target_count_max <= 0.0:
            raise ValueError("target_count_max must be positive")

        result = tensor.clone()
        known_mask = torch.zeros(scene_types.shape, dtype=torch.bool, device=scene_types.device)
        fight_time_max = float(self._config.fight_time_max)
        for window_schema in schema.scene_windows:
            type_mask = scene_types == window_schema.scene_type_id
            known_mask |= type_mask
            if not bool(type_mask.any().item()):
                continue

            rows = result[type_mask].clone()
            start = rows[:, window_schema.start_offset_index].clamp(
                min=0.0,
                max=fight_time_max,
            ) / fight_time_max
            end = rows[:, window_schema.end_offset_index].clamp(
                min=0.0,
                max=fight_time_max,
            ) / fight_time_max
            rows[:, window_schema.start_offset_index] = start
            rows[:, window_schema.end_offset_index] = end
            rows[:, window_schema.duration_index] = (end - start).clamp_min(0.0)
            if "target_count" in window_schema.feature_keys:
                target_count_index = window_schema.feature_keys.index("target_count")
                rows[:, target_count_index] = rows[:, target_count_index].clamp(
                    min=0.0,
                    max=float(self._config.target_count_max),
                ) / float(self._config.target_count_max)
            result[type_mask] = rows

        if bool((~known_mask).any().item()):
            unknown_types = torch.unique(scene_types[~known_mask]).tolist()
            raise ValueError(f"unknown scene type ids: {unknown_types}")
        return result

    def normalize_value(self, group_key: str, feature_name: str, value: float) -> float:
        """对单个标量应用归一化。"""
        rules = self._rules.get(group_key)
        if rules is None:
            raise KeyError(f"group_key {group_key!r} not registered")
        for rule in rules:
            if rule.feature_name == feature_name:
                return _apply_rule_to_scalar(
                    value,
                    rule.rule_type,
                    self._config,
                    max_value=rule.max_value,
                )
        raise ValueError(f"feature {feature_name!r} not found in group {group_key!r}")

    def inverse(self, group_key: str, feature_name: str, normalized_value: float) -> float:
        """归一化逆变换，仅用于调试。"""
        rules = self._rules.get(group_key)
        if rules is None:
            raise KeyError(f"group_key {group_key!r} not registered")
        for rule in rules:
            if rule.feature_name == feature_name:
                return _inverse_rule(
                    normalized_value,
                    rule.rule_type,
                    self._config,
                    max_value=rule.max_value,
                )
        raise ValueError(f"feature {feature_name!r} not found in group {group_key!r}")


def _apply_normalize_inplace(
    values,
    rule_type: str,
    config: NormalizerConfig,
    *,
    max_value: float | None = None,
) -> None:
    if rule_type == "divide_mp_max":
        values.div_(config.mp_max)
    elif rule_type == "divide_max_mp":
        values.div_(config.mp_max)
    elif rule_type == "clip_divide_seconds_max":
        values.clamp_(min=0.0, max=config.remaining_seconds_max).div_(config.remaining_seconds_max)
    elif rule_type == "clip_divide_fight_time_max":
        values.clamp_(min=0.0, max=config.fight_time_max).div_(config.fight_time_max)
    elif rule_type == "divide_potency_by_fight_time_max":
        values.clamp_(min=0.0).div_(config.fight_time_max)
    elif rule_type == "clip_divide_resource_max":
        if max_value is None or max_value <= 0.0:
            raise ValueError("resource normalization requires a positive max_value")
        values.clamp_(min=0.0, max=max_value).div_(max_value)
    elif rule_type == "clip_divide_status_max":
        if max_value is None or max_value <= 0.0:
            raise ValueError("status normalization requires a positive max_value")
        values.clamp_(min=0.0, max=max_value).div_(max_value)
    elif rule_type == "clip_divide_gcds_max":
        values.clamp_(min=0.0, max=config.remaining_gcds_max).div_(config.remaining_gcds_max)
    elif rule_type == "log1p_potency":
        values.clamp_(min=0.0)
        values.log1p_()
    elif rule_type == "divide_current_potency_max":
        values.clamp_(min=0.0, max=config.current_potency_max).div_(config.current_potency_max)


def _apply_skill_normalize_inplace(
    values,
    feature_name: str,
    config: NormalizerConfig,
    *,
    resource_limits: dict[str, float],
) -> None:
    """把 compiled cache 中的动态 skill 数值压到模型可比较的尺度。"""
    leaf_name = feature_name.split(".")[-1]
    if leaf_name in {"is_legal", "max_charges", "available_charges", "gcd_index"}:
        return
    if leaf_name == "potency":
        values.clamp_(min=0.0, max=config.current_potency_max).div_(config.current_potency_max)
        return
    if leaf_name in {"actual_mp_cost", "mp_cost"}:
        values.clamp_(min=0.0, max=config.mp_max).div_(config.mp_max)
        return
    if feature_name.endswith(".gcds"):
        values.clamp_(min=0.0, max=config.remaining_gcds_max).div_(config.remaining_gcds_max)
        return
    if leaf_name == "time_seconds":
        _apply_normalize_inplace(values, "clip_divide_fight_time_max", config)
        return
    resource_max = resource_limits.get(leaf_name)
    if resource_max is not None:
        _apply_normalize_inplace(
            values,
            "clip_divide_resource_max",
            config,
            max_value=resource_max,
        )
        return
    if feature_name.endswith(".seconds") or leaf_name.endswith("_seconds"):
        values.clamp_(min=0.0, max=config.remaining_seconds_max).div_(config.remaining_seconds_max)
        return
    if leaf_name.endswith("_timer"):
        values.clamp_(min=0.0, max=config.remaining_seconds_max).div_(config.remaining_seconds_max)


def _infer_rule_type(
    raw_name: str,
    resource_limits: dict[str, float],
    status_limits: dict[str, float],
    cumulative_potency_mode: str,
) -> str:
    """根据字段名推断归一化规则。"""
    leaf_name = _leaf_name(raw_name)

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
    if leaf_name in ("ogcds_weaved", "max_ogcd_per_window", "gcd_index", "mp_ratio"):
        return "keep"
    if leaf_name == "mp":
        return "divide_mp_max"
    if leaf_name == "max_mp":
        return "divide_max_mp"
    if leaf_name in ("time_seconds", "fight_remaining_seconds"):
        return "clip_divide_fight_time_max"
    if leaf_name in resource_limits:
        return "clip_divide_resource_max"
    if leaf_name.endswith("_seconds"):
        return "clip_divide_seconds_max"
    if leaf_name.endswith("_gcds"):
        return "clip_divide_gcds_max"
    if leaf_name == "current_potency":
        return "divide_current_potency_max"
    if leaf_name == "current_gcd_dot_potency":
        return "divide_current_potency_max"
    if leaf_name in ("cumulative_potency", "cumulative_dot_potency"):
        if cumulative_potency_mode == "log1p":
            return "log1p_potency"
        if cumulative_potency_mode == "divide":
            return "divide_potency_by_fight_time_max"
        raise ValueError(
            "unsupported cumulative_potency_mode: "
            f"{cumulative_potency_mode!r}; expected 'log1p' or 'divide'"
        )
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
    for prefix in ("before.", "after.", "consumed."):
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


def _apply_rule_to_scalar(
    value: float,
    rule_type: str,
    config: NormalizerConfig,
    *,
    max_value: float | None = None,
) -> float:
    if rule_type == "divide_mp_max":
        return value / config.mp_max
    if rule_type == "divide_max_mp":
        return value / config.mp_max
    if rule_type in ("keep", "skip"):
        return value
    if rule_type == "clip_divide_seconds_max":
        return max(0.0, min(value, config.remaining_seconds_max)) / config.remaining_seconds_max
    if rule_type == "clip_divide_fight_time_max":
        return max(0.0, min(value, config.fight_time_max)) / config.fight_time_max
    if rule_type == "divide_potency_by_fight_time_max":
        return max(0.0, value) / config.fight_time_max
    if rule_type == "clip_divide_resource_max":
        if max_value is None or max_value <= 0.0:
            raise ValueError("resource normalization requires a positive max_value")
        return max(0.0, min(value, max_value)) / max_value
    if rule_type == "clip_divide_status_max":
        if max_value is None or max_value <= 0.0:
            raise ValueError("status normalization requires a positive max_value")
        return max(0.0, min(value, max_value)) / max_value
    if rule_type == "clip_divide_gcds_max":
        return max(0.0, min(value, config.remaining_gcds_max)) / config.remaining_gcds_max
    if rule_type == "log1p_potency":
        return math.log1p(max(0.0, value))
    if rule_type == "divide_current_potency_max":
        return max(0.0, min(value, config.current_potency_max)) / config.current_potency_max
    return value


def _inverse_rule(
    normalized: float,
    rule_type: str,
    config: NormalizerConfig,
    *,
    max_value: float | None = None,
) -> float:
    if rule_type == "divide_mp_max":
        return normalized * config.mp_max
    if rule_type == "divide_max_mp":
        return normalized * config.mp_max
    if rule_type in ("keep", "skip"):
        return normalized
    if rule_type == "clip_divide_seconds_max":
        return normalized * config.remaining_seconds_max
    if rule_type == "clip_divide_fight_time_max":
        return normalized * config.fight_time_max
    if rule_type == "divide_potency_by_fight_time_max":
        return normalized * config.fight_time_max
    if rule_type == "clip_divide_resource_max":
        if max_value is None or max_value <= 0.0:
            raise ValueError("resource normalization requires a positive max_value")
        return normalized * max_value
    if rule_type == "clip_divide_status_max":
        if max_value is None or max_value <= 0.0:
            raise ValueError("status normalization requires a positive max_value")
        return normalized * max_value
    if rule_type == "clip_divide_gcds_max":
        return normalized * config.remaining_gcds_max
    if rule_type == "log1p_potency":
        return math.expm1(normalized)
    if rule_type == "divide_current_potency_max":
        return normalized * config.current_potency_max
    return normalized


@dataclass(frozen=True)
class _FieldRule:
    index: int
    feature_name: str
    raw_name: str
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
