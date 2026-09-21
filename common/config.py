"""项目配置加载层。

这个文件专门负责读取 YAML 配置，整理成 Python 数据类。
它不处理战斗逻辑，只负责把静态配置装配好给状态机和输出层使用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .models import ActionKind, SkillDefinition, StatusDefinition
from common.project_config import resolve_project_job_tag, resolve_project_path
from common.torch_dependencies import import_torch
from common.yaml_config import load_yaml_mapping

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
PRECISION_CONFIG_PATH = PROJECT_ROOT / "config" / "precision.yaml"


@dataclass(frozen=True)
class RuntimeConfig:
    job_tag: str
    strict_validation: bool
    system_config_path: Path
    job_config_path: Path


@dataclass(frozen=True)
class EngineTimingConfig:
    # 容量一动作队列窗口：状态机据此保留一个待执行动作并在就绪时刻自动提交。
    action_queue_window_seconds: float


@dataclass(frozen=True)
class EngineResourceConfig:
    max_mp: int


@dataclass(frozen=True)
class SystemPotencyConfig:
    burst_potion_multiplier: float
    raid_buff_window_multiplier: float


@dataclass(frozen=True)
class SystemMpRecoveryConfig:
    tick_interval_seconds: float
    in_combat_amount: int


@dataclass(frozen=True)
class JobConfig:
    key: str
    name: str
    timing: dict[str, float]
    resource_limits: dict[str, float]
    statuses: dict[str, StatusDefinition]
    skills: tuple[SkillDefinition, ...]

    @property
    def default_fight_remaining(self) -> float:
        return self.timing["default_fight_remaining"]

    def timing_value(self, key: str) -> float:
        return self.timing[key]


@dataclass(frozen=True)
class SystemConfig:
    base_gcd: float
    skill_table_base_gcd: float
    mp_recovery: SystemMpRecoveryConfig
    potency: SystemPotencyConfig
    raid_buff_window_marker_skills: tuple[str, ...]
    raid_buff_window_duration: float
    statuses: dict[str, StatusDefinition]
    skills: tuple[SkillDefinition, ...]


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    version: str
    runtime: RuntimeConfig
    engine_timing: EngineTimingConfig
    engine_resources: EngineResourceConfig
    system: SystemConfig
    job: JobConfig


@dataclass(frozen=True)
class PrecisionConfig:
    """统一 tensor 输出精度配置。"""

    int_dtype: str
    float_dtype: str

    def resolve_int_dtype(self):
        return _resolve_torch_dtype(
            self.int_dtype,
            field_name="int_dtype",
            allowed_names=_INT_DTYPE_NAMES,
            category_name="integer",
        )

    def resolve_float_dtype(self):
        return _resolve_torch_dtype(
            self.float_dtype,
            field_name="float_dtype",
            allowed_names=_FLOAT_DTYPE_NAMES,
            category_name="floating-point",
        )


_INT_DTYPE_ALIASES = {
    "int32": "int32",
    "int64": "int64",
}

_FLOAT_DTYPE_ALIASES = {
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "fp32": "float32",
    "fp16": "float16",
}

_INT_DTYPE_NAMES = frozenset(_INT_DTYPE_ALIASES.values())
_FLOAT_DTYPE_NAMES = frozenset(_FLOAT_DTYPE_ALIASES.values())


def load_precision_config(path: Path | None = None) -> PrecisionConfig:
    """加载统一精度配置。"""
    config_path = path or PRECISION_CONFIG_PATH
    raw = load_yaml_mapping(config_path)
    payload = raw.get("precision", {})
    resolved_int = _resolve_precision_alias(
        payload.get("int_dtype", "int32"),
        field_name="int_dtype",
        aliases=_INT_DTYPE_ALIASES,
    )
    resolved_float = _resolve_precision_alias(
        payload.get("float_dtype", "float32"),
        field_name="float_dtype",
        aliases=_FLOAT_DTYPE_ALIASES,
    )
    return PrecisionConfig(int_dtype=resolved_int, float_dtype=resolved_float)


def _resolve_precision_alias(raw_value: object, *, field_name: str, aliases: dict[str, str]) -> str:
    raw_text = str(raw_value).lower()
    resolved = aliases.get(raw_text)
    if resolved is None:
        supported = ", ".join(sorted(aliases))
        raise ValueError(
            f"precision.yaml: {field_name}={raw_value!r} is invalid; supported aliases: {supported}"
        )
    return resolved


def _resolve_torch_dtype(
    dtype_name: str,
    *,
    field_name: str,
    allowed_names: frozenset[str],
    category_name: str,
):
    torch = import_torch()

    try:
        dtype = getattr(torch, dtype_name)
    except AttributeError:
        raise ValueError(
            f"unsupported torch dtype for {field_name}: {dtype_name!r}; "
            f"allowed canonical names: {', '.join(sorted(allowed_names))}"
        ) from None

    if dtype_name not in allowed_names:
        raise ValueError(
            f"{field_name} must use {category_name} torch dtype, got {dtype_name!r}; "
            f"allowed canonical names: {', '.join(sorted(allowed_names))}"
        )
    return dtype


def load_project_config(
    path: Path | None = None,
    *,
    job_tag: str | None = None,
) -> ProjectConfig:
    """加载项目配置和当前职业配置。"""
    config_path = path or DEFAULT_CONFIG_PATH
    payload = load_yaml_mapping(config_path)

    runtime_data = payload["runtime"]
    requested_job_tag = resolve_project_job_tag(
        project_root=PROJECT_ROOT,
        explicit=job_tag,
    )
    engine_timing_data = payload["engine_timing"]
    engine_resource_data = payload["engine_resources"]
    system_config_path = resolve_project_path(
        runtime_data["system_config"],
        project_root=PROJECT_ROOT,
    )
    job_config_path = _resolve_job_config_path(
        runtime_data,
        requested_job_tag=requested_job_tag,
    )
    system_payload = load_yaml_mapping(system_config_path)
    job_payload = load_yaml_mapping(job_config_path)

    runtime = RuntimeConfig(
        job_tag=requested_job_tag,
        strict_validation=bool(runtime_data.get("strict_validation", True)),
        system_config_path=system_config_path,
        job_config_path=job_config_path,
    )
    engine_timing = EngineTimingConfig(
        action_queue_window_seconds=float(
            engine_timing_data["action_queue_window_seconds"]
        ),
    )
    engine_resources = EngineResourceConfig(
        max_mp=int(engine_resource_data["max_mp"]),
    )
    system_timing = system_payload["timing"]
    system_potency_data = system_payload.get("potency", {})
    raid_buff_window_data = system_payload["raid_buff_window"]
    raid_buff_window_marker_skills = tuple(
        str(skill_key) for skill_key in raid_buff_window_data["marker_skills"]
    )
    if not raid_buff_window_marker_skills:
        raise ValueError("raid_buff_window.marker_skills must not be empty")
    if len(set(raid_buff_window_marker_skills)) != len(raid_buff_window_marker_skills):
        raise ValueError("raid_buff_window.marker_skills must not contain duplicates")
    raid_buff_window_duration = float(raid_buff_window_data["duration_seconds"])
    if raid_buff_window_duration <= 0:
        raise ValueError("raid_buff_window.duration_seconds must be positive")
    system = SystemConfig(
        base_gcd=float(system_timing["base_gcd"]),
        skill_table_base_gcd=float(system_timing["skill_table_base_gcd"]),
        mp_recovery=SystemMpRecoveryConfig(
            tick_interval_seconds=float(system_payload["mp_recovery"]["tick_interval_seconds"]),
            in_combat_amount=int(system_payload["mp_recovery"]["in_combat_amount"]),
        ),
        potency=SystemPotencyConfig(
            burst_potion_multiplier=float(system_potency_data.get("burst_potion_multiplier", 1.05)),
            raid_buff_window_multiplier=float(system_potency_data.get("raid_buff_window_multiplier", 1.10)),
        ),
        raid_buff_window_marker_skills=raid_buff_window_marker_skills,
        raid_buff_window_duration=raid_buff_window_duration,
        statuses=_build_status_definitions(system_payload.get("statuses", {})),
        skills=_build_skill_definitions(system_payload.get("skills", {})),
    )

    timing = job_payload["timing"]
    job = JobConfig(
        key=job_payload["job"]["key"],
        name=job_payload["job"]["name"],
        timing={key: float(value) for key, value in timing.items()},
        resource_limits=_build_resource_limits(job_payload.get("resources", {})),
        statuses=_build_status_definitions(job_payload.get("statuses", {})),
        skills=_build_skill_definitions(job_payload.get("skills", {})),
    )
    _validate_timing_resource_contracts(job)
    _validate_status_definitions(system, job)
    _validate_status_conflicts(system, job)
    _validate_skill_conflicts(system, job)
    _validate_skill_status_references(system, job)

    return ProjectConfig(
        name=payload["project"]["name"],
        version=str(payload["project"]["version"]),
        runtime=runtime,
        engine_timing=engine_timing,
        engine_resources=engine_resources,
        system=system,
        job=job,
    )


def _resolve_job_config_path(runtime_data: dict, *, requested_job_tag: str) -> Path:
    """按职业标签解析已注册的职业 YAML。"""
    configured_paths = runtime_data.get("job_configs", {})
    if not isinstance(configured_paths, dict):
        raise ValueError("runtime.job_configs must be a mapping")

    configured_path = configured_paths.get(requested_job_tag)
    if configured_path is None:
        supported = ", ".join(sorted(str(key) for key in configured_paths))
        raise ValueError(
            f"no job config registered for {requested_job_tag!r}; supported={supported}"
        )
    return resolve_project_path(configured_path, project_root=PROJECT_ROOT)


def _build_status_definitions(raw_statuses: dict) -> dict[str, StatusDefinition]:
    definitions: dict[str, StatusDefinition] = {}
    for key, payload in raw_statuses.items():
        definitions[key] = StatusDefinition(
            key=key,
            game_id=int(payload["game_id"]),
            duration=float(payload["duration"]),
            max_stacks=int(payload.get("max_stacks", 1)),
        )
    return definitions


def _build_resource_limits(raw_resources: object) -> dict[str, float]:
    """读取职业量谱上限，供状态机和训练归一化共同使用。"""
    if raw_resources is None:
        return {}
    if not isinstance(raw_resources, dict):
        raise ValueError("job resources must be a mapping")

    limits: dict[str, float] = {}
    for key, payload in raw_resources.items():
        if not isinstance(payload, dict):
            raise ValueError(f"job resource {key!r} must be a mapping")
        if "max_value" not in payload:
            raise ValueError(f"job resource {key!r} is missing max_value")
        max_value = float(payload["max_value"])
        if not math.isfinite(max_value) or max_value <= 0.0:
            raise ValueError(
                f"job resource {key!r}: max_value must be positive and finite, got {max_value!r}"
            )
        limits[str(key)] = max_value
    return limits


def _validate_timing_resource_contracts(job: JobConfig) -> None:
    """校验需要同时参与时序和资源归一化的职业配置，避免双源静默漂移。"""
    wildfire_duration = job.timing.get("wildfire_duration")
    wildfire_remaining_max = job.resource_limits.get("wildfire_remaining")
    if wildfire_duration is None or wildfire_remaining_max is None:
        return
    if not math.isclose(wildfire_duration, wildfire_remaining_max, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "job config wildfire_duration and resources.wildfire_remaining.max_value "
            f"must match, got {wildfire_duration} and {wildfire_remaining_max}"
        )


def _build_skill_definitions(raw_skills: dict) -> tuple[SkillDefinition, ...]:
    skills: list[SkillDefinition] = []
    for key, payload in raw_skills.items():
        raw_mp_cost = payload.get("mp_cost", 0)
        mp_cost = raw_mp_cost if isinstance(raw_mp_cost, str) else int(raw_mp_cost)
        if isinstance(mp_cost, str) and mp_cost not in {"full"}:
            raise ValueError(f"unsupported mp_cost semantic for skill {key}: {mp_cost}")
        if isinstance(mp_cost, str):
            mp_cost_floor = int(payload.get("mp_cost_floor", 0))
        else:
            mp_cost_floor = int(payload.get("mp_cost_floor", mp_cost))
        aoe_secondary_reduction = float(payload.get("aoe_secondary_reduction", 1.0))
        if not 0.0 <= aoe_secondary_reduction <= 1.0:
            raise ValueError(
                f"skill {key}: aoe_secondary_reduction must be between 0 and 1, "
                f"got {aoe_secondary_reduction}"
            )
        value = float(payload.get("value", 1.0))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"skill {key}: value must be a positive finite number, got {value}")
        skills.append(
            SkillDefinition(
                key=key,
                game_id=int(payload["game_id"]),
                name=payload["name"],
                kind=ActionKind(payload["kind"]),
                behavior=payload["behavior"],
                potency=int(payload.get("potency", 0)),
                value=value,
                cast_time=float(payload.get("cast_time", 0.0)),
                recast_time=float(payload.get("recast_time", 2.5)),
                mp_cost=mp_cost,
                mp_cost_floor=mp_cost_floor,
                cooldown=float(payload.get("cooldown", 0.0)),
                charges=int(payload.get("charges", 1)),
                enabled=bool(payload.get("enabled", True)),
                max_targets=int(payload.get("max_targets", 1)),
                aoe_secondary_reduction=aoe_secondary_reduction,
                dot_potency=int(payload.get("dot_potency", 0)),
                dot_duration=float(payload.get("dot_duration", 0.0)),
                dot_key=(str(payload["dot_key"]) if payload.get("dot_key") is not None else None),
                requires_target=bool(
                    payload.get(
                        "requires_target",
                        int(payload.get("potency", 0)) > 0 or int(payload.get("dot_potency", 0)) > 0,
                    )
                ),
                applies_statuses=tuple(payload.get("applies_statuses", ())),
                tags=tuple(payload.get("tags", ())),
            )
        )
    return tuple(skills)


def _validate_status_conflicts(system: SystemConfig, job: JobConfig) -> None:
    duplicate_keys = sorted(set(system.statuses) & set(job.statuses))
    if duplicate_keys:
        joined = ", ".join(duplicate_keys)
        raise ValueError(f"duplicate status keys across system/job config: {joined}")

    system_game_ids = {definition.game_id: key for key, definition in system.statuses.items()}
    for key, definition in job.statuses.items():
        system_key = system_game_ids.get(definition.game_id)
        if system_key is None:
            continue
        raise ValueError(
            "duplicate status game_id across system/job config: "
            f"{definition.game_id} ({system_key}, {key})"
        )


def _validate_skill_conflicts(system: SystemConfig, job: JobConfig) -> None:
    duplicate_keys = sorted({skill.key for skill in system.skills} & {skill.key for skill in job.skills})
    if duplicate_keys:
        joined = ", ".join(duplicate_keys)
        raise ValueError(f"duplicate skill keys across system/job config: {joined}")

    system_game_ids = {skill.game_id: skill.key for skill in system.skills}
    for skill in job.skills:
        system_key = system_game_ids.get(skill.game_id)
        if system_key is None:
            continue
        raise ValueError(
            "duplicate skill game_id across system/job config: "
            f"{skill.game_id} ({system_key}, {skill.key})"
        )


def _validate_skill_status_references(system: SystemConfig, job: JobConfig) -> None:
    status_keys = set(system.statuses) | set(job.statuses)
    for source, skills in (("system", system.skills), ("job", job.skills)):
        for skill in skills:
            if skill.behavior == "grant_status" and not skill.applies_statuses:
                raise ValueError(
                    f"{source} skill {skill.key} uses grant_status but applies_statuses is empty"
                )
            for status_key in skill.applies_statuses:
                if status_key in status_keys:
                    continue
                raise ValueError(
                    f"{source} skill {skill.key} references unregistered status: {status_key}"
                )


def _validate_status_definitions(system: SystemConfig, job: JobConfig) -> None:
    """验证状态定义的 max_stacks 合法。"""
    for source, statuses in (("system", system.statuses), ("job", job.statuses)):
        for key, definition in statuses.items():
            if definition.max_stacks < 1:
                raise ValueError(
                    f"{source} status {key}: max_stacks must be >= 1, got {definition.max_stacks}"
                )
