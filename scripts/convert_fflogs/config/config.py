"""convert_fflogs 的职业和 GCD 检测配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.project_config import load_root_dotenv, resolve_project_job_tag
from common.yaml_config import load_yaml_mapping


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONVERT_CONFIG_ROOT = _PROJECT_ROOT / "config" / "convert_fflogs"
CONVERT_DEFAULT_CONFIG_PATH = CONVERT_CONFIG_ROOT / "default.yaml"
CONVERT_JOB_CONFIG_DIR = CONVERT_CONFIG_ROOT / "jobs"


@dataclass(frozen=True)
class GcdDetectionConfig:
    """原始日志的基础 GCD 估算配置。"""

    probe_skill_game_id: int
    haste_excluded_buff_game_ids: tuple[int, ...]
    fallback_seconds: float
    min_probe_casts: int
    min_gap_samples: int
    histogram_lower_ms: int
    histogram_upper_ms: int
    histogram_bin_width_ms: int
    probe_skill_cast_time_seconds: float | None = None


@dataclass(frozen=True)
class ConvertFflogsConfig:
    """convert_fflogs 全局配置。"""

    default_worker_count: int
    gcd_detection_defaults: GcdDetectionConfig


@dataclass(frozen=True)
class ConvertFflogsJobConfig:
    """convert_fflogs 职业级配置。"""

    job_tag: str
    gcd_detection: GcdDetectionConfig


def load_convert_fflogs_config(path: Path | None = None) -> ConvertFflogsConfig:
    config_path = path or CONVERT_DEFAULT_CONFIG_PATH
    payload = load_yaml_mapping(config_path, description="convert_fflogs config")
    section = _require_mapping(payload, "convert_fflogs", source=str(config_path))
    gcd_defaults = _require_mapping(section, "gcd_detection_defaults", source=str(config_path))
    worker_count = int(section.get("default_worker_count", 6))
    if worker_count < 1:
        raise ValueError(f"{config_path}: default_worker_count must be >= 1")
    return ConvertFflogsConfig(
        default_worker_count=worker_count,
        gcd_detection_defaults=_build_gcd_detection_config(
            gcd_defaults,
            source=str(config_path),
        ),
    )


def load_convert_fflogs_job_config(
    job_tag: str,
    *,
    default_path: Path | None = None,
    job_dir: Path | None = None,
) -> ConvertFflogsJobConfig:
    default_config = load_convert_fflogs_config(default_path)
    job_path = (job_dir or CONVERT_JOB_CONFIG_DIR) / f"{job_tag}.yaml"
    payload = load_yaml_mapping(job_path, description="convert_fflogs config")
    job_section = _require_mapping(payload, "job", source=str(job_path))
    configured_job_tag = str(job_section.get("key", ""))
    if configured_job_tag != job_tag:
        raise ValueError(
            f"convert_fflogs job config key mismatch: expected {job_tag!r}, got {configured_job_tag!r}"
        )
    overrides = _require_mapping(payload, "gcd_detection", source=str(job_path))
    defaults = default_config.gcd_detection_defaults
    merged = {
        "fallback_seconds": defaults.fallback_seconds,
        "min_probe_casts": defaults.min_probe_casts,
        "min_gap_samples": defaults.min_gap_samples,
        "histogram_lower_ms": defaults.histogram_lower_ms,
        "histogram_upper_ms": defaults.histogram_upper_ms,
        "histogram_bin_width_ms": defaults.histogram_bin_width_ms,
        **overrides,
    }
    gcd_detection = _build_gcd_detection_config(merged, source=str(job_path))
    if gcd_detection.probe_skill_game_id <= 0:
        raise ValueError(f"{job_path}: gcd_detection.probe_skill_game_id must be > 0")
    return ConvertFflogsJobConfig(
        job_tag=job_tag,
        gcd_detection=gcd_detection,
    )


def resolve_convert_fflogs_job_tag(explicit_job_tag: str | None) -> str:
    return resolve_project_job_tag(
        project_root=_PROJECT_ROOT,
        explicit=explicit_job_tag,
    )


def load_convert_fflogs_dotenv(*, override: bool = False) -> Path | None:
    return load_root_dotenv(_PROJECT_ROOT, override=override)


def _require_mapping(payload: dict[str, object], key: str, *, source: str) -> dict[str, object]:
    value = payload.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"{source}: {key} must be a mapping")
    return value


def _build_gcd_detection_config(
    payload: dict[str, object],
    *,
    source: str,
) -> GcdDetectionConfig:
    if "probe_skill_game_id" not in payload:
        raise ValueError(f"{source}: gcd_detection.probe_skill_game_id is required")
    excluded_ids = payload.get("haste_excluded_buff_game_ids", ())
    if not isinstance(excluded_ids, (list, tuple)):
        raise ValueError(
            f"{source}: gcd_detection.haste_excluded_buff_game_ids must be a list"
        )
    probe_cast_time = payload.get("probe_skill_cast_time_seconds")
    if probe_cast_time is not None:
        probe_cast_time = float(probe_cast_time)
        if probe_cast_time <= 0.0:
            raise ValueError(
                f"{source}: gcd_detection.probe_skill_cast_time_seconds must be > 0"
            )
    return GcdDetectionConfig(
        probe_skill_game_id=int(payload["probe_skill_game_id"]),
        haste_excluded_buff_game_ids=tuple(int(item) for item in excluded_ids),
        fallback_seconds=float(payload.get("fallback_seconds", 2.5)),
        min_probe_casts=int(payload.get("min_probe_casts", 4)),
        min_gap_samples=int(payload.get("min_gap_samples", 3)),
        histogram_lower_ms=int(payload.get("histogram_lower_ms", 1800)),
        histogram_upper_ms=int(payload.get("histogram_upper_ms", 2600)),
        histogram_bin_width_ms=int(payload.get("histogram_bin_width_ms", 10)),
        probe_skill_cast_time_seconds=probe_cast_time,
    )
