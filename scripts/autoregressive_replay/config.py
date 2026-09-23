"""自回归回放的命令行和根目录 `.env` 配置。"""

from __future__ import annotations

import os
from pathlib import Path

from common.project_config import (
    PROJECT_JOB_TAG_ENV,
    load_root_dotenv,
    resolve_project_path,
)
from common.policy.config import (
    PROJECT_ROOT,
    load_policy_config,
    load_model_config,
    resolve_policy_cache_dir,
    resolve_policy_checkpoint_path,
    resolve_policy_device,
    resolve_policy_model_config_path,
    resolve_policy_model_job_tag,
    resolve_policy_model_variant,
    validate_policy_model_variant,
)
from common.policy.replay import AutoregressiveReplayConfig
from common.torch_serialization import safe_torch_load
from scripts.onnx_export.config.config import (
    AUTOREGRESSIVE_REPLAY_CHECKPOINT_ENV,
    AUTOREGRESSIVE_REPLAY_ORT_PROVIDER_ENV,
    AUTOREGRESSIVE_REPLAY_SCENE_JSON_ENV,
    resolve_onnx_package_path,
)
from scripts.onnx_export.runtime.ort_runtime import ORT_PROVIDER_CUDA
from scripts.onnx_export.runtime.precision import SUPPORTED_PRECISIONS
AUTOREGRESSIVE_REPLAY_BACKEND_ENV = "AUTOREGRESSIVE_REPLAY_BACKEND"
AUTOREGRESSIVE_REPLAY_OUTPUT_ENV = "AUTOREGRESSIVE_REPLAY_OUTPUT"
AUTOREGRESSIVE_REPLAY_SCENE_MODE_ENV = "AUTOREGRESSIVE_REPLAY_SCENE_MODE"
AUTOREGRESSIVE_REPLAY_SCENE_SAMPLE_ENV = "AUTOREGRESSIVE_REPLAY_SCENE_SAMPLE"
AUTOREGRESSIVE_REPLAY_MAX_STEPS_ENV = "AUTOREGRESSIVE_REPLAY_MAX_STEPS"
AUTOREGRESSIVE_REPLAY_MAX_GCDS_ENV = "AUTOREGRESSIVE_REPLAY_MAX_GCDS"
AUTOREGRESSIVE_REPLAY_TOP_K_ENV = "AUTOREGRESSIVE_REPLAY_TOP_K"
AUTOREGRESSIVE_REPLAY_TOP_P_ENV = "AUTOREGRESSIVE_REPLAY_TOP_P"
AUTOREGRESSIVE_REPLAY_TEMPERATURE_ENV = "AUTOREGRESSIVE_REPLAY_TEMPERATURE"
AUTOREGRESSIVE_REPLAY_DEVICE_ENV = "AUTOREGRESSIVE_REPLAY_DEVICE"
AUTOREGRESSIVE_REPLAY_INITIAL_ACTION_ENV = "AUTOREGRESSIVE_REPLAY_INITIAL_ACTION"
AUTOREGRESSIVE_REPLAY_INITIAL_TIME_ENV = "AUTOREGRESSIVE_REPLAY_INITIAL_TIME"
AUTOREGRESSIVE_REPLAY_BASE_GCD_ENV = "AUTOREGRESSIVE_REPLAY_BASE_GCD"
AUTOREGRESSIVE_REPLAY_USE_KV_CACHE_ENV = "AUTOREGRESSIVE_REPLAY_USE_KV_CACHE"


def load_replay_config(
    *,
    checkpoint: Path | None = None,
    backend: str | None = None,
    onnx_package: Path | None = None,
    ort_provider: str | None = None,
    output: Path | None = None,
    scene_json: Path | None = None,
    scene_mode: str | None = None,
    scene_sample_index: int | None = None,
    max_steps: int | None = None,
    max_gcds: int | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    temperature: float | None = None,
    max_history: int | None = None,
    device: str | None = None,
    use_kv_cache: bool | None = None,
    policy_precision: str | None = None,
) -> AutoregressiveReplayConfig:
    """从命令行覆盖和根目录 `.env` 合并回放配置。"""
    # backend 决定后续走 checkpoint 还是 ONNX manifest，必须先加载根目录
    # `.env` 再解析；否则无参数 CLI 会在 dotenv 尚未加载时错误回退到 PyTorch。
    load_root_dotenv(PROJECT_ROOT)
    backend_name = _backend_name(
        backend or os.environ.get(AUTOREGRESSIVE_REPLAY_BACKEND_ENV, "pytorch")
    )
    configured_job_tag = _optional_text(os.environ.get(PROJECT_JOB_TAG_ENV))
    model_config_path = resolve_policy_model_config_path()
    model_config = load_model_config(model_config_path)
    model_job_tag = resolve_policy_model_job_tag(model_config_path)
    model_variant = resolve_policy_model_variant(model_config_path)
    source_config = load_policy_config(model_config_path)
    training_raw = source_config.get("training", {}) or {}
    if not isinstance(training_raw, dict):
        raise ValueError("training config section must be a mapping")
    raw_root = resolve_project_path(
        source_config.get("raw_data_dir", ""),
        project_root=PROJECT_ROOT,
    )
    cache_shard_size = int(training_raw.get("compiled_cache_shard_size", 512))
    cache_max_shards = int(training_raw.get("compiled_cache_max_shards", 8))
    model_history_capacity = model_config.history_capacity
    if configured_job_tag and configured_job_tag != model_job_tag:
        raise ValueError(
            f"{PROJECT_JOB_TAG_ENV}={configured_job_tag!r} does not match policy model "
            f"job {model_job_tag!r}"
        )
    if backend_name == "onnxruntime":
        onnx_package_path = _resolve_onnx_package(onnx_package)
        onnx_metadata = _load_onnx_metadata(onnx_package_path)
        job_tag = str(onnx_metadata["job_tag"])
        onnx_model_variant = str(onnx_metadata["model_variant"])
        if configured_job_tag and configured_job_tag != job_tag:
            raise ValueError(
                f"ONNX job_tag {job_tag!r} does not match configured job_tag "
                f"{configured_job_tag!r}"
            )
        if model_job_tag != job_tag:
            raise ValueError(
                f"ONNX job_tag {job_tag!r} does not match training model config job "
                f"{model_job_tag!r}"
            )
        if onnx_model_variant != model_variant:
            raise ValueError(
                f"ONNX model_variant {onnx_model_variant!r} does not match training "
                f"model variant {model_variant!r}"
            )
        checkpoint_path = None
        model_history_capacity = int(onnx_metadata["history_capacity"])
        if model_config.history_capacity != model_history_capacity:
            raise ValueError(
                "ONNX history capacity differs from training replay config: "
                f"{model_history_capacity} != {model_config.history_capacity}"
            )
    else:
        onnx_package_path = None
        checkpoint_path = _resolve_checkpoint(model_config_path, checkpoint)
        checkpoint_payload = _load_checkpoint_metadata(checkpoint_path)
        checkpoint_job_tag = checkpoint_payload.get("job_tag") or checkpoint_payload[
            "data_spec"
        ].get("job_tag")
        checkpoint_job_tag = _optional_text(checkpoint_job_tag)
        validate_policy_model_variant(
            checkpoint_payload,
            model_variant,
            artifact_name="checkpoint",
        )
        job_tag = model_job_tag
        if checkpoint_job_tag and job_tag and checkpoint_job_tag != job_tag:
            raise ValueError(
                f"checkpoint job_tag {checkpoint_job_tag!r} does not match configured "
                f"job_tag {job_tag!r}"
            )
        if job_tag and model_job_tag != job_tag:
            raise ValueError(
                f"configured job_tag {job_tag!r} does not match training model job "
                f"{model_job_tag!r}"
            )
        job_tag = model_job_tag

    resolved_scene_json = _resolve_scene_json(scene_json, raw_root=raw_root)
    resolved_scene_mode = (
        scene_mode
        or os.environ.get(AUTOREGRESSIVE_REPLAY_SCENE_MODE_ENV, "cache")
    ).strip().lower()
    if resolved_scene_mode not in {"cache", "empty"}:
        raise ValueError(
            f"{AUTOREGRESSIVE_REPLAY_SCENE_MODE_ENV} must be cache or empty, "
            f"got {resolved_scene_mode!r}"
        )
    result = AutoregressiveReplayConfig(
        checkpoint_path=checkpoint_path,
        output_path=_resolve_path(
            output,
            AUTOREGRESSIVE_REPLAY_OUTPUT_ENV,
            "artifacts/autoregressive_rollout.md",
        ),
        scene_json_path=resolved_scene_json,
        cache_dir=resolve_policy_cache_dir(job_tag),
        model_history_capacity=model_history_capacity,
        cache_shard_size=cache_shard_size,
        cache_max_shards=cache_max_shards,
        scene_mode=resolved_scene_mode,
        scene_sample_index=_positive_int(
            scene_sample_index,
            AUTOREGRESSIVE_REPLAY_SCENE_SAMPLE_ENV,
            0,
            minimum=0,
        ),
        max_steps=_positive_int(
            max_steps,
            AUTOREGRESSIVE_REPLAY_MAX_STEPS_ENV,
            100,
            minimum=1,
        ),
        max_gcds=_optional_positive_int(
            max_gcds,
            AUTOREGRESSIVE_REPLAY_MAX_GCDS_ENV,
        ),
        top_k=_positive_int(
            top_k,
            AUTOREGRESSIVE_REPLAY_TOP_K_ENV,
            8,
            minimum=1,
        ),
        top_p=_probability_float(
            top_p,
            AUTOREGRESSIVE_REPLAY_TOP_P_ENV,
            1.0,
        ),
        temperature=_nonnegative_float(
            temperature,
            AUTOREGRESSIVE_REPLAY_TEMPERATURE_ENV,
            0.0,
        ),
        # 未显式做截断实验时，PT 以 checkpoint 为准，ORT 以 manifest 为准。
        max_history=(
            model_history_capacity if max_history is None else int(max_history)
        ),
        device=(
            "cpu"
            if backend_name == "onnxruntime"
            else (
                device
                or os.environ.get(AUTOREGRESSIVE_REPLAY_DEVICE_ENV)
                or resolve_policy_device()
            ).strip().lower()
        ),
        job_tag=job_tag,
        model_variant=model_variant,
        initial_action=_optional_text(
            os.environ.get(AUTOREGRESSIVE_REPLAY_INITIAL_ACTION_ENV, "fire_iii")
        ),
        initial_time_seconds=_optional_float(
            os.environ.get(AUTOREGRESSIVE_REPLAY_INITIAL_TIME_ENV)
        ),
        base_gcd=_optional_positive_float(
            os.environ.get(AUTOREGRESSIVE_REPLAY_BASE_GCD_ENV)
        ),
        use_kv_cache=_boolean(
            use_kv_cache,
            AUTOREGRESSIVE_REPLAY_USE_KV_CACHE_ENV,
            backend_name == "pytorch",
        ),
        backend=backend_name,
        onnx_package_path=onnx_package_path,
        ort_provider=(
            ort_provider
            or os.environ.get(
                AUTOREGRESSIVE_REPLAY_ORT_PROVIDER_ENV,
                ORT_PROVIDER_CUDA,
            )
        ).strip(),
        policy_precision=_optional_precision(policy_precision),
    )
    if result.device not in {"cuda", "cpu"}:
        raise ValueError(
            f"{AUTOREGRESSIVE_REPLAY_DEVICE_ENV} must be cuda or cpu, got {result.device!r}"
        )
    if result.backend == "onnxruntime" and result.use_kv_cache:
        raise ValueError("ONNX Runtime replay v1 requires KV cache to be disabled")
    if result.backend == "onnxruntime" and result.policy_precision is not None:
        raise ValueError(
            "PyTorch policy precision override is unavailable for ONNX Runtime replay; "
            "the manifest precision is authoritative"
        )
    if result.max_history > result.model_history_capacity:
        raise ValueError(
            "autoregressive replay max_history exceeds model capacity: "
            f"{result.max_history} > {result.model_history_capacity}"
        )
    if result.max_history < 0:
        raise ValueError(
            f"autoregressive replay max_history must be >= 0, got {result.max_history}"
        )
    if not result.ort_provider:
        raise ValueError(f"{AUTOREGRESSIVE_REPLAY_ORT_PROVIDER_ENV} must not be empty")
    return result


def _backend_name(value: object) -> str:
    normalized = str(value).strip().lower()
    aliases = {"pytorch": "pytorch", "torch": "pytorch", "onnxruntime": "onnxruntime", "ort": "onnxruntime"}
    result = aliases.get(normalized)
    if result is None:
        raise ValueError(
            f"{AUTOREGRESSIVE_REPLAY_BACKEND_ENV} must be pytorch or onnxruntime, "
            f"got {value!r}"
        )
    return result


def _resolve_onnx_package(explicit: Path | None) -> Path:
    # 只接受 CLI 显式传入的部署包；旧 .env 的 AUTOREGRESSIVE_REPLAY_ONNX_PACKAGE
    # 已不再读取，未显式指定时按 checkpoint 推导，避免回放读错模型的部署包。
    if explicit is not None:
        path = resolve_onnx_package_path(explicit=explicit)
    else:
        checkpoint_path = _resolve_checkpoint(
            resolve_policy_model_config_path(),
            None,
        )
        path = resolve_onnx_package_path(
            explicit=None,
            checkpoint_path=checkpoint_path,
        )
    package_dir = path if path.is_dir() else path.parent
    if not (package_dir / "manifest.json").is_file():
        raise FileNotFoundError(f"ONNX deployment manifest not found: {package_dir}")
    return path


def _load_onnx_metadata(package_path: Path) -> dict[str, object]:
    """只解析路由字段；backend 会在建 session 前执行完整 manifest 验证。"""
    import json

    package_dir = package_path if package_path.is_dir() else package_path.parent
    payload = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    try:
        contract = payload["contract"]
        model = payload["model"]
        return {
            "job_tag": str(contract["job_tag"]),
            "model_variant": str(model["model_variant"]),
            "history_capacity": int(contract["capacity"]["history_capacity"]),
        }
    except KeyError as exc:
        missing = str(exc.args[0]) if exc.args else "?"
        if missing != "model_variant":
            raise ValueError(
                f"ONNX manifest is missing replay routing field {missing!r}"
            ) from exc
        raise ValueError(
            "ONNX manifest is missing model_variant; re-export the package"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("ONNX manifest is missing replay routing fields") from exc


def _resolve_checkpoint(model_config_path: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = resolve_project_path(explicit, project_root=PROJECT_ROOT)
    else:
        raw = os.environ.get(AUTOREGRESSIVE_REPLAY_CHECKPOINT_ENV)
        path = (
            resolve_project_path(raw, project_root=PROJECT_ROOT)
            if raw
            else resolve_policy_checkpoint_path(model_config_path)
        )
    if not path.is_file():
        raise FileNotFoundError(f"autoregressive checkpoint not found: {path}")
    return path


def _resolve_scene_json(explicit: Path | None, *, raw_root: Path) -> Path:
    raw = explicit or os.environ.get(AUTOREGRESSIVE_REPLAY_SCENE_JSON_ENV)
    if raw:
        path = resolve_project_path(raw, project_root=PROJECT_ROOT)
    else:
        candidates = sorted(Path(raw_root).rglob("*.json"))
        if not candidates:
            raise FileNotFoundError(f"no scene raw JSON found: {raw_root}")
        path = candidates[0]
    if not path.is_file():
        raise FileNotFoundError(f"autoregressive scene raw JSON not found: {path}")
    return path


def _resolve_path(explicit: Path | None, env_name: str, default: str) -> Path:
    raw = explicit or os.environ.get(env_name, default)
    return resolve_project_path(raw, project_root=PROJECT_ROOT)


def _positive_int(
    explicit: int | None,
    env_name: str,
    default: int,
    *,
    minimum: int,
) -> int:
    value = explicit if explicit is not None else int(os.environ.get(env_name, default))
    if value < minimum:
        raise ValueError(f"{env_name} must be >= {minimum}, got {value}")
    return int(value)


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_precision(value: object) -> str | None:
    precision = _optional_text(value)
    if precision is None:
        return None
    normalized = precision.lower()
    if normalized not in SUPPORTED_PRECISIONS:
        raise ValueError(
            f"policy precision must be one of {SUPPORTED_PRECISIONS}, got {value!r}"
        )
    return normalized


def _optional_positive_int(explicit: int | None, env_name: str) -> int | None:
    raw = explicit if explicit is not None else _optional_text(os.environ.get(env_name))
    if raw is None:
        return None
    value = int(raw)
    if value < 1:
        raise ValueError(f"{env_name} must be >= 1, got {value}")
    return value


def _optional_float(value: object) -> float | None:
    text = _optional_text(value)
    if text is None:
        return None
    result = float(text)
    if result > 0.0:
        raise ValueError(
            f"{AUTOREGRESSIVE_REPLAY_INITIAL_TIME_ENV} must be <= 0, got {result}"
        )
    return result


def _optional_positive_float(value: object) -> float | None:
    text = _optional_text(value)
    if text is None:
        return None
    result = float(text)
    if result <= 0.0:
        raise ValueError(f"{AUTOREGRESSIVE_REPLAY_BASE_GCD_ENV} must be > 0, got {result}")
    return result


def _nonnegative_float(
    explicit: float | None,
    env_name: str,
    default: float,
) -> float:
    value = explicit if explicit is not None else float(os.environ.get(env_name, default))
    if value < 0.0:
        raise ValueError(f"{env_name} must be >= 0, got {value}")
    return float(value)


def _probability_float(
    explicit: float | None,
    env_name: str,
    default: float,
) -> float:
    value = explicit if explicit is not None else float(os.environ.get(env_name, default))
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{env_name} must be > 0 and <= 1, got {value}")
    return float(value)


def _boolean(explicit: bool | None, env_name: str, default: bool) -> bool:
    """读取可选布尔配置，供部署时关闭推理缓存。"""
    if explicit is not None:
        return bool(explicit)
    raw = os.environ.get(env_name)
    if raw is None:
        return bool(default)
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{env_name} must be a boolean, got {raw!r}")


def _load_checkpoint_metadata(path: Path) -> dict[str, object]:
    payload = safe_torch_load(path)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must be a mapping: {path}")
    if not isinstance(payload.get("data_spec"), dict):
        raise ValueError(f"checkpoint missing data_spec: {path}")
    return payload
