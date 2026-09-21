"""ONNX 导出、发布门禁与回放共享的根目录 `.env` 配置。"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path

from common.project_config import load_root_dotenv, resolve_project_path
from common.policy.config import (
    PROJECT_ROOT,
    resolve_policy_checkpoint_path,
    resolve_policy_model_config_path,
)

from ..release.policy import (
    RELEASE_EMPTY_MIN_GCDS,
    minimum_empty_action_budget,
)
from ..runtime.ort_runtime import ORT_PROVIDER_CUDA
from ..runtime.precision import PRECISION_BF16, SUPPORTED_PRECISIONS


# 导出与回放共用同一模型来源、部署包、ORT Provider 和实战 scene。
AUTOREGRESSIVE_REPLAY_CHECKPOINT_ENV = "AUTOREGRESSIVE_REPLAY_CHECKPOINT"
AUTOREGRESSIVE_REPLAY_ONNX_PACKAGE_ENV = "AUTOREGRESSIVE_REPLAY_ONNX_PACKAGE"
AUTOREGRESSIVE_REPLAY_ORT_PROVIDER_ENV = "AUTOREGRESSIVE_REPLAY_ORT_PROVIDER"
AUTOREGRESSIVE_REPLAY_SCENE_JSON_ENV = "AUTOREGRESSIVE_REPLAY_SCENE_JSON"

ONNX_EXPORT_DEPLOYMENT_PROFILE_ENV = "ONNX_EXPORT_DEPLOYMENT_PROFILE"
ONNX_EXPORT_OPSET_ENV = "ONNX_EXPORT_OPSET"
ONNX_EXPORT_PRECISION_ENV = "ONNX_EXPORT_PRECISION"
ONNX_EXPORT_VALIDATION_DEVICES_ENV = "ONNX_EXPORT_VALIDATION_DEVICES"
ONNX_EXPORT_OVERWRITE_ENV = "ONNX_EXPORT_OVERWRITE"

ONNX_PARITY_EMPTY_MAX_STEPS_ENV = "ONNX_PARITY_EMPTY_MAX_STEPS"
ONNX_PARITY_EMPTY_MAX_GCDS_ENV = "ONNX_PARITY_EMPTY_MAX_GCDS"
ONNX_PARITY_EMPTY_REPORT_ENV = "ONNX_PARITY_EMPTY_REPORT"
ONNX_PARITY_SCENE_MAX_STEPS_ENV = "ONNX_PARITY_SCENE_MAX_STEPS"
ONNX_PARITY_SCENE_REPORT_ENV = "ONNX_PARITY_SCENE_REPORT"
ONNX_PARITY_TOLERANCE_ENV = "ONNX_PARITY_TOLERANCE"


@dataclass(frozen=True)
class OnnxExportConfig:
    """一次 ONNX 图导出的完整配置。"""

    checkpoint_path: Path
    output_dir: Path
    deployment_profile_path: Path | None
    opset: int
    precision: str
    ort_provider: str
    validation_devices: tuple[str, ...]
    overwrite: bool


@dataclass(frozen=True)
class OnnxParityConfig:
    """两项正式 PT/ORT parity 门禁的运行参数。"""

    empty_max_steps: int
    empty_max_gcds: int
    empty_report_path: Path
    scene_max_steps: int
    scene_report_path: Path
    tolerance: float | None


def load_export_config(
    *,
    checkpoint: Path | None = None,
    output_dir: Path | None = None,
    deployment_profile: Path | None = None,
    opset: int | None = None,
    precision: str | None = None,
    ort_provider: str | None = None,
    validation_devices: str | tuple[str, ...] | None = None,
    overwrite: bool | None = None,
) -> OnnxExportConfig:
    """合并 CLI 临时覆盖和根目录 `.env`；CLI 始终优先。"""
    load_root_dotenv(PROJECT_ROOT)
    checkpoint_path = _resolve_checkpoint(checkpoint)
    resolved_precision = (
        precision or os.environ.get(ONNX_EXPORT_PRECISION_ENV, PRECISION_BF16)
    ).strip().lower()
    if resolved_precision not in SUPPORTED_PRECISIONS:
        raise ValueError(
            f"{ONNX_EXPORT_PRECISION_ENV} must be one of "
            f"{', '.join(SUPPORTED_PRECISIONS)}, got {resolved_precision!r}"
        )
    resolved_opset = _positive_int(
        opset,
        ONNX_EXPORT_OPSET_ENV,
        18,
    )
    resolved_provider = (
        ort_provider
        or os.environ.get(AUTOREGRESSIVE_REPLAY_ORT_PROVIDER_ENV, ORT_PROVIDER_CUDA)
    ).strip()
    if not resolved_provider:
        raise ValueError(f"{AUTOREGRESSIVE_REPLAY_ORT_PROVIDER_ENV} must not be empty")

    return OnnxExportConfig(
        checkpoint_path=checkpoint_path,
        output_dir=resolve_onnx_package_path(
            explicit=output_dir,
            checkpoint_path=checkpoint_path,
        ),
        deployment_profile_path=_optional_project_path(
            deployment_profile,
            ONNX_EXPORT_DEPLOYMENT_PROFILE_ENV,
        ),
        opset=resolved_opset,
        precision=resolved_precision,
        ort_provider=resolved_provider,
        validation_devices=_validation_devices(validation_devices),
        overwrite=_boolean(overwrite, ONNX_EXPORT_OVERWRITE_ENV, False),
    )


def load_parity_config() -> OnnxParityConfig:
    """读取正式空场景与实战 parity 门禁参数。"""
    load_root_dotenv(PROJECT_ROOT)
    empty_max_gcds = _positive_int(
        None,
        ONNX_PARITY_EMPTY_MAX_GCDS_ENV,
        RELEASE_EMPTY_MIN_GCDS,
    )
    return OnnxParityConfig(
        empty_max_steps=_positive_int(
            None,
            ONNX_PARITY_EMPTY_MAX_STEPS_ENV,
            minimum_empty_action_budget(empty_max_gcds),
        ),
        empty_max_gcds=empty_max_gcds,
        empty_report_path=_env_project_path(
            ONNX_PARITY_EMPTY_REPORT_ENV,
            "artifacts/onnx_parity_bf16_empty_128gcd.json",
        ),
        scene_max_steps=_positive_int(
            None,
            ONNX_PARITY_SCENE_MAX_STEPS_ENV,
            100,
        ),
        scene_report_path=_env_project_path(
            ONNX_PARITY_SCENE_REPORT_ENV,
            "artifacts/onnx_parity_bf16_scene_100.json",
        ),
        tolerance=_optional_nonnegative_float(ONNX_PARITY_TOLERANCE_ENV),
    )


def resolve_onnx_package_path(
    *,
    explicit: Path | str | None,
    checkpoint_path: Path,
) -> Path:
    """解析共用部署包路径；未配置时按 checkpoint 目录结构自动推导。"""
    raw = explicit or _optional_text(
        os.environ.get(AUTOREGRESSIVE_REPLAY_ONNX_PACKAGE_ENV)
    )
    if raw:
        path = resolve_project_path(raw, project_root=PROJECT_ROOT)
        return path.parent if path.suffix.lower() == ".onnx" else path
    return derive_onnx_output_dir(checkpoint_path, project_root=PROJECT_ROOT)


def derive_onnx_output_dir(checkpoint_path: Path, *, project_root: Path) -> Path:
    """把 artifacts/checkpoints/<job>/<run> 映射到 artifacts/exports/<job>/<run>。"""
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint_root = (Path(project_root) / "artifacts" / "checkpoints").resolve()
    try:
        relative_parent = checkpoint_path.relative_to(checkpoint_root).parent
    except ValueError as exc:
        raise ValueError(
            f"cannot derive ONNX output from checkpoint outside {checkpoint_root}; "
            f"set {AUTOREGRESSIVE_REPLAY_ONNX_PACKAGE_ENV} explicitly"
        ) from exc
    if len(relative_parent.parts) < 2:
        raise ValueError(
            "checkpoint path must follow artifacts/checkpoints/<job>/<run>/<file>.pt "
            f"or set {AUTOREGRESSIVE_REPLAY_ONNX_PACKAGE_ENV} explicitly"
        )
    return (Path(project_root) / "artifacts" / "exports" / relative_parent).resolve()


def _resolve_checkpoint(explicit: Path | None) -> Path:
    if explicit is not None:
        return resolve_project_path(explicit, project_root=PROJECT_ROOT)
    raw = _optional_text(os.environ.get(AUTOREGRESSIVE_REPLAY_CHECKPOINT_ENV))
    if raw:
        return resolve_project_path(raw, project_root=PROJECT_ROOT)
    model_config_path = resolve_policy_model_config_path()
    return resolve_policy_checkpoint_path(model_config_path).resolve()


def _optional_project_path(explicit: Path | None, env_name: str) -> Path | None:
    raw = explicit or _optional_text(os.environ.get(env_name))
    if not raw:
        return None
    return resolve_project_path(raw, project_root=PROJECT_ROOT)


def _env_project_path(env_name: str, default: str) -> Path:
    raw = _optional_text(os.environ.get(env_name)) or default
    return resolve_project_path(raw, project_root=PROJECT_ROOT)


def _validation_devices(
    explicit: str | tuple[str, ...] | None,
) -> tuple[str, ...]:
    raw: str | tuple[str, ...] = (
        explicit
        if explicit is not None
        else os.environ.get(ONNX_EXPORT_VALIDATION_DEVICES_ENV, "cuda")
    )
    values = (
        tuple(value.strip() for value in raw.split(",") if value.strip())
        if isinstance(raw, str)
        else tuple(str(value).strip() for value in raw if str(value).strip())
    )
    if not values or any(value not in {"cpu", "cuda"} for value in values):
        raise ValueError(
            f"{ONNX_EXPORT_VALIDATION_DEVICES_ENV} only supports comma-separated "
            f"cpu,cuda, got {raw!r}"
        )
    return values


def _positive_int(explicit: int | None, env_name: str, default: int) -> int:
    raw = explicit if explicit is not None else os.environ.get(env_name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{env_name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{env_name} must be >= 1, got {value}")
    return value


def _boolean(explicit: bool | None, env_name: str, default: bool) -> bool:
    if explicit is not None:
        return bool(explicit)
    raw = _optional_text(os.environ.get(env_name))
    if raw is None:
        return default
    normalized = raw.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{env_name} must be true or false, got {raw!r}")


def _optional_nonnegative_float(env_name: str) -> float | None:
    raw = _optional_text(os.environ.get(env_name))
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{env_name} must be a number, got {raw!r}") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{env_name} must be finite and >= 0, got {value}")
    return value


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
