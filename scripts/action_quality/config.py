"""复用项目环境变量和模型 YAML，解析桥接运行参数。"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from common.policy.config import (
    load_policy_config,
    resolve_policy_model_config_path,
)
from common.project_config import (
    load_root_dotenv,
    resolve_project_path,
)
from common.yaml_config import load_yaml_mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BridgeConfig:
    analyzer_root: Path
    node_modules: Path
    node: str
    timeout: float
    max_workers: int


def load_bridge_config() -> BridgeConfig:
    """机器路径可由 .env 覆盖，运行时参数以 YAML 为默认权威来源。"""
    load_root_dotenv(PROJECT_ROOT)
    raw = load_yaml_mapping(PROJECT_ROOT / "config/action_quality.yaml")["bridge"]
    analyzer = resolve_project_path(raw["analyzer_root"], project_root=PROJECT_ROOT)
    modules = resolve_project_path(
        os.environ.get("ACTION_QUALITY_NODE_MODULES") or analyzer / "node_modules",
        project_root=PROJECT_ROOT,
    )
    timeout = float(raw["timeout_seconds"])
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("bridge.timeout_seconds must be positive and finite")
    max_workers = raw["max_workers"]
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("bridge.max_workers must be a positive integer")
    return BridgeConfig(
        analyzer, modules, os.environ.get("ACTION_QUALITY_NODE") or "node", timeout,
        max_workers,
    )


def default_raw_root() -> Path:
    """从模型输入阶段定位同级 raw，避免重复评估 annotated 文件。"""
    config = load_policy_config(resolve_policy_model_config_path())
    source = resolve_project_path(config["raw_data_dir"], project_root=PROJECT_ROOT)
    annotated = next((part for part in (source, *source.parents) if part.name == "annotated"), None)
    if annotated is None:
        return source
    return annotated.with_name("raw") / source.relative_to(annotated)
