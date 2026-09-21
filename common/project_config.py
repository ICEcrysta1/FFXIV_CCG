"""跨模块复用的项目配置、环境变量和路径解析工具。"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_JOB_TAG_ENV = "FFXIV_JOB_TAG"


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


def resolve_project_job_tag(
    *,
    project_root: Path,
    explicit: str | None = None,
) -> str:
    """解析项目统一职业标签；显式参数优先，其次读取根目录 `.env`。"""
    if explicit:
        return str(explicit).strip()
    load_root_dotenv(project_root)
    value = os.environ.get(PROJECT_JOB_TAG_ENV)
    if value:
        return str(value).strip()
    raise ValueError(
        f"missing {PROJECT_JOB_TAG_ENV}; set it in the project .env"
    )


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
