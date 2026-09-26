"""复用项目环境变量和模型 YAML，解析桥接运行参数。"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from common.policy.config import load_policy_config, resolve_policy_model_config_path
from common.project_config import (
    load_root_dotenv,
    resolve_project_model_variant,
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
    return BridgeConfig(analyzer, modules, os.environ.get("ACTION_QUALITY_NODE") or "node", timeout)


def default_raw_root() -> Path:
    """不重复维护 raw 路径，从现有策略模型配置读取。"""
    config = load_policy_config(resolve_policy_model_config_path())
    return resolve_project_path(config["raw_data_dir"], project_root=PROJECT_ROOT)


def severity_weights(job_tag: str) -> dict[str, float]:
    """按输入职业选择权重；没有配置时保留等级，不借用其他职业配置。"""
    if not os.environ.get("FFXIV_MODEL_VARIANT", "").strip():
        return {}
    variant = resolve_project_model_variant(project_root=PROJECT_ROOT)
    manifest = PROJECT_ROOT / "config/models" / job_tag / variant / "config.yaml"
    if not manifest.is_file():
        return {}
    quality = load_policy_config(manifest).get("action_quality", {})
    weights = quality.get("severity_weights", {})
    if not weights:
        return {}
    result = {}
    for level in ("minor", "medium", "major"):
        value = weights.get(level)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"invalid action quality severity weight: {level}")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid action quality severity weight: {level}")
        result[level] = float(value)
    return result
