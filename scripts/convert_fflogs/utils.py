"""通用工具函数 —— 后端构造、静态配置、时间合并、路径消毒。"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from .config.constants import _round_time
from scripts.common.inprocess_backend import InProcessBackend

# 确保项目根目录在 sys.path 中，以解析 common 配置/模型导入
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.config import load_project_config  # noqa: E402
from common.skills import SkillBook  # noqa: E402


def load_job_project_config(job_tag: str):
    """加载职业项目配置（静态数据，不依赖状态机后端）。"""
    return load_project_config(job_tag=job_tag)


def build_skill_book(project_config):
    """从项目配置构建技能索引（静态数据，系统技能 + 职业技能合并）。"""
    return SkillBook.from_project_config(project_config)


def build_backend(job_tag: str, *, max_history: int | None = None) -> InProcessBackend:
    """构造进程内 C# 状态机后端，不启动 SidecarHost 子进程。"""
    return InProcessBackend(job_tag, max_history=max_history)


def merge_timestamps_to_windows(
    timestamps: list[float],
    *,
    gap_seconds: float,
) -> list[tuple[float, float]]:
    """把离得很近的一组时间点合并成窗口。"""
    if not timestamps:
        return []

    ordered = sorted(timestamps)
    windows: list[tuple[float, float]] = []
    current_start = ordered[0]
    current_end = ordered[0]

    for timestamp in ordered[1:]:
        if timestamp - current_end <= gap_seconds:
            current_end = timestamp
        else:
            windows.append((current_start, current_end))
            current_start = timestamp
            current_end = timestamp
    windows.append((current_start, current_end))
    return windows


def _sanitize_path_fragment(value: str) -> str:
    """清理字符串以安全用于文件/目录名。"""
    cleaned = re.sub(r"[^\w一-鿿-]+", "_", value, flags=re.UNICODE).strip("_")
    return cleaned or "unnamed"
