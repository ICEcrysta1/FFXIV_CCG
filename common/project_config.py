"""跨模块复用的项目配置、环境变量和路径解析工具。"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_JOB_TAG_ENV = "FFXIV_JOB_TAG"
PROJECT_MODEL_VARIANT_ENV = "FFXIV_MODEL_VARIANT"
TENSORBOARD_PORT_ENV = "TENSORBOARD_PORT"
DEFAULT_TENSORBOARD_PORT = 6006


def resolve_project_path(value: object, *, project_root: Path) -> Path:
    """把绝对路径原样保留，相对路径解析到项目根目录。"""
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def load_root_dotenv(project_root: Path, *, override: bool = False) -> Path | None:
    """加载项目根目录 `.env`，已有进程环境变量默认优先。"""
    env_path = Path(project_root) / ".env"
    if not env_path.is_file():
        return None

    load_dotenv(env_path, override=override)
    return env_path


def resolve_tensorboard_port(
    *,
    project_root: Path,
    explicit: int | str | None = None,
) -> int:
    """解析 TensorBoard Web 端口；进程环境优先，其次项目 `.env`，默认 6006。"""
    value = explicit
    if value is None:
        load_root_dotenv(project_root)
        value = os.environ.get(TENSORBOARD_PORT_ENV)
    if value is None or (isinstance(value, str) and not value.strip()):
        return DEFAULT_TENSORBOARD_PORT
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(
            f"{TENSORBOARD_PORT_ENV} must be an integer from 1 to 65535, "
            f"got {value!r}"
        )

    try:
        port = int(value)
    except ValueError as exc:
        raise ValueError(
            f"{TENSORBOARD_PORT_ENV} must be an integer from 1 to 65535, "
            f"got {value!r}"
        ) from exc
    if not 1 <= port <= 65535:
        raise ValueError(
            f"{TENSORBOARD_PORT_ENV} must be between 1 and 65535, got {port}"
        )
    return port


def resolve_project_job_tag(
    *,
    project_root: Path,
    explicit: str | None = None,
) -> str:
    """解析项目统一职业标签；显式参数优先，其次读取根目录 `.env`。"""
    value = explicit
    if value is None:
        load_root_dotenv(project_root)
        value = os.environ.get(PROJECT_JOB_TAG_ENV)
    if value is None or not str(value).strip():
        raise ValueError(
            f"missing {PROJECT_JOB_TAG_ENV}; set it in the project .env"
        )
    return str(value).strip()


def resolve_project_model_variant(
    *,
    project_root: Path,
    explicit: str | None = None,
) -> str:
    """解析策略模型变体目录名；显式参数优先，其次读取根目录 `.env`。"""
    if explicit is None:
        load_root_dotenv(project_root)
        value = os.environ.get(PROJECT_MODEL_VARIANT_ENV)
    else:
        value = explicit
    if value is None or not str(value).strip():
        raise ValueError(
            f"missing {PROJECT_MODEL_VARIANT_ENV}; set it in the project .env"
        )

    variant = str(value).strip()
    variant_path = Path(variant)
    if (
        variant in {".", ".."}
        or variant_path.is_absolute()
        or len(variant_path.parts) != 1
    ):
        raise ValueError(
            f"{PROJECT_MODEL_VARIANT_ENV} must be a single model variant directory name, "
            f"got {variant!r}"
        )
    return variant


def resolve_registered_job_tags(project_root: Path) -> tuple[str, ...]:
    """从项目总配置读取全部已注册职业 tag（配置驱动）。

    与状态机注册层解耦：职业是否可用由 `config/default.yaml` 的
    `runtime.job_configs` 声明，不依赖运行时状态机注册中心。
    """
    from .yaml_config import load_yaml_mapping

    payload = load_yaml_mapping(project_root / "config" / "default.yaml")
    job_configs = payload.get("runtime", {}).get("job_configs", {})
    if not isinstance(job_configs, dict):
        return ()
    return tuple(str(key) for key in job_configs)
